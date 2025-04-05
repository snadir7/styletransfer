"""Neural style transfer (https://arxiv.org/abs/1508.06576) in PyTorch."""

import copy
from dataclasses import dataclass
from functools import partial
import time
import warnings
import torch

import numpy as np
from PIL import Image
import torch
from torch import optim, nn
from torch.nn import functional as F
from torchvision import models, transforms
from torchvision.transforms import functional as TF

from . import sqrtm

# Helper functions (can be placed outside the class or as static methods)

def rgb_to_ycbc(image_tensor):
    """Converts a batch of RGB tensors (NCHW, float [0,1]) to YCbCr."""
    # Ensure tensor is on CPU for numpy conversion if needed, or use torch operations
    if image_tensor.is_cuda:
        image_tensor = image_tensor.cpu() # Simple approach, could be optimized

    # Using standard conversion matrices (like JPEG)
    # Alternative: Use skimage.color.rgb2ycbcr or similar if allowed
    r = image_tensor[:, 0, :, :]
    g = image_tensor[:, 1, :, :]
    b = image_tensor[:, 2, :, :]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5 # Offset to keep in [0, 1] range approx
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5 # Offset to keep in [0, 1] range approx

    # Clamp just in case to maintain [0, 1] range roughly for Cb, Cr after offset
    # Note: True YCbCr ranges differ, but this keeps things simple for tensor combination
    return torch.stack([y, cb.clamp(0, 1), cr.clamp(0, 1)], dim=1)


def ycbcr_to_rgb(image_tensor):
    """Converts a batch of YCbCr tensors (NCHW, float [0,1]) back to RGB."""
    if image_tensor.is_cuda:
        image_tensor = image_tensor.cpu()

    y = image_tensor[:, 0, :, :]
    cb = image_tensor[:, 1, :, :] - 0.5 # Remove offset
    cr = image_tensor[:, 2, :, :] - 0.5 # Remove offset

    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb

    # Stack and clamp to valid [0, 1] RGB range
    return torch.stack([r, g, b], dim=1).clamp(0, 1)

def pil_rgb_to_ycbcr_channels(pil_image):
    """Converts a PIL RGB image to separate Y, Cb, Cr tensors [0, 1]."""
    rgb_tensor = TF.to_tensor(pil_image).unsqueeze(0) # Add batch dim
    ycbcr_tensor = rgb_to_ycbc(rgb_tensor)
    return ycbcr_tensor[0, 0:1, :, :], ycbcr_tensor[0, 1:2, :, :], ycbcr_tensor[0, 2:3, :, :] # Return Y, Cb, Cr as separate tensors

def combine_y_cbcr_to_pil(y_tensor, cb_tensor, cr_tensor):
    """Combines Y, Cb, Cr tensors into an RGB PIL image."""
    # Ensure tensors have batch dimension if needed by ycbcr_to_rgb
    if y_tensor.dim() == 3: y_tensor = y_tensor.unsqueeze(0)
    if cb_tensor.dim() == 3: cb_tensor = cb_tensor.unsqueeze(0)
    if cr_tensor.dim() == 3: cr_tensor = cr_tensor.unsqueeze(0)

    ycbcr_tensor = torch.cat([y_tensor, cb_tensor, cr_tensor], dim=1)
    rgb_tensor = ycbcr_to_rgb(ycbcr_tensor)
    return TF.to_pil_image(rgb_tensor.squeeze(0).cpu()) # Remove batch dim


class VGGFeatures(nn.Module):
    poolings = {'max': nn.MaxPool2d, 'average': nn.AvgPool2d, 'l2': partial(nn.LPPool2d, 2)}
    pooling_scales = {'max': 1., 'average': 2., 'l2': 0.78}

    def __init__(self, layers, pooling='max'):
        super().__init__()
        self.layers = sorted(set(layers))

        # The PyTorch pre-trained VGG-19 expects sRGB inputs in the range [0, 1] which are then
        # normalized according to this transform, unlike Simonyan et al.'s original model.
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # The PyTorch pre-trained VGG-19 has different parameters from Simonyan et al.'s original
        # model.
        self.model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:self.layers[-1] + 1]
        self.devices = [torch.device('cpu')] * len(self.model)

        # Reduces edge artifacts.
        self.model[0] = self._change_padding_mode(self.model[0], 'replicate')

        pool_scale = self.pooling_scales[pooling]
        for i, layer in enumerate(self.model):
            if pooling != 'max' and isinstance(layer, nn.MaxPool2d):
                # Changing the pooling type from max results in the scale of activations
                # changing, so rescale them. Gatys et al. (2015) do not do this.
                self.model[i] = Scale(self.poolings[pooling](2), pool_scale)

        self.model.eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _change_padding_mode(conv, padding_mode):
        new_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                             stride=conv.stride, padding=conv.padding,
                             padding_mode=padding_mode)
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight)
            new_conv.bias.copy_(conv.bias)
        return new_conv

    @staticmethod
    def _get_min_size(layers):
        last_layer = max(layers)
        min_size = 1
        for layer in [4, 9, 18, 27, 36]:
            if last_layer < layer:
                break
            min_size *= 2
        return min_size

    def distribute_layers(self, devices):
        for i, layer in enumerate(self.model):
            if i in devices:
                device = torch.device(devices[i])
            self.model[i] = layer.to(device)
            self.devices[i] = device

    def forward(self, input, layers=None):
        layers = self.layers if layers is None else sorted(set(layers))
        h, w = input.shape[2:4]
        min_size = self._get_min_size(layers)
        if min(h, w) < min_size:
            raise ValueError(f'Input is {h}x{w} but must be at least {min_size}x{min_size}')
        feats = {'input': input}
        input = self.normalize(input)
        for i in range(max(layers) + 1):
            input = self.model[i](input.to(self.devices[i]))
            if i in layers:
                feats[i] = input
        return feats


class ScaledMSELoss(nn.Module):
    """Computes MSE scaled such that its gradient L1 norm is approximately 1.
    This differs from Gatys at al. (2015) and Johnson et al."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.register_buffer('eps', torch.tensor(eps))

    def extra_repr(self):
        return f'eps={self.eps:g}'

    def forward(self, input, target):
        diff = input - target
        return diff.pow(2).sum() / diff.abs().sum().add(self.eps)


class ContentLoss(nn.Module):
    def __init__(self, target, eps=1e-8):
        super().__init__()
        self.register_buffer('target', target)
        self.loss = ScaledMSELoss(eps=eps)

    def forward(self, input):
        return self.loss(input, self.target)


class ContentLossMSE(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.register_buffer('target', target)
        self.loss = nn.MSELoss()

    def forward(self, input):
        return self.loss(input, self.target)


class StyleLoss(nn.Module):
    def __init__(self, target, eps=1e-8):
        super().__init__()
        self.register_buffer('target', target)
        self.loss = ScaledMSELoss(eps=eps)

    @staticmethod
    def get_target(target):
        mat = target.flatten(-2)
        # The Gram matrix normalization differs from Gatys et al. (2015) and Johnson et al.
        return mat @ mat.transpose(-2, -1) / mat.shape[-1]

    def forward(self, input):
        return self.loss(self.get_target(input), self.target)


def eye_like(x):
    return torch.eye(x.shape[-2], x.shape[-1], dtype=x.dtype, device=x.device).expand_as(x)


class StyleLossW2(nn.Module):
    """Wasserstein-2 style loss."""

    def __init__(self, target, eps=1e-4):
        super().__init__()
        self.sqrtm = partial(sqrtm.sqrtm_ns_lyap, num_iters=12)
        mean, srm = target
        cov = self.srm_to_cov(mean, srm) + eye_like(srm) * eps
        self.register_buffer('mean', mean)
        self.register_buffer('cov', cov)
        self.register_buffer('cov_sqrt', self.sqrtm(cov))
        self.register_buffer('eps', mean.new_tensor(eps))

    @staticmethod
    def get_target(target):
        """Compute the mean and second raw moment of the target activations.
        Unlike the covariance matrix, these are valid to combine linearly."""
        mean = target.mean([-2, -1])
        srm = torch.einsum('...chw,...dhw->...cd', target, target) / (target.shape[-2] * target.shape[-1])
        return mean, srm

    @staticmethod
    def srm_to_cov(mean, srm):
        """Compute the covariance matrix from the mean and second raw moment."""
        return srm - torch.einsum('...c,...d->...cd', mean, mean)

    def forward(self, input):
        mean, srm = self.get_target(input)
        cov = self.srm_to_cov(mean, srm) + eye_like(srm) * self.eps
        mean_diff = torch.mean((mean - self.mean) ** 2)
        sqrt_term = self.sqrtm(self.cov_sqrt @ cov @ self.cov_sqrt)
        cov_diff = torch.diagonal(self.cov + cov - 2 * sqrt_term, dim1=-2, dim2=-1).mean()
        return mean_diff + cov_diff


class TVLoss(nn.Module):
    """L2 total variation loss (nine point stencil)."""

    def forward(self, input):
        input = F.pad(input, (1, 1, 1, 1), 'replicate')
        s1, s2 = slice(1, -1), slice(2, None)
        s3, s4 = slice(None, -1), slice(1, None)
        d1 = (input[..., s1, s2] - input[..., s1, s1]).pow(2).mean() / 3
        d2 = (input[..., s2, s1] - input[..., s1, s1]).pow(2).mean() / 3
        d3 = (input[..., s4, s4] - input[..., s3, s3]).pow(2).mean() / 12
        d4 = (input[..., s4, s3] - input[..., s3, s4]).pow(2).mean() / 12
        return 2 * (d1 + d2 + d3 + d4)


class SumLoss(nn.ModuleList):
    def __init__(self, losses, verbose=False):
        super().__init__(losses)
        self.verbose = verbose

    def forward(self, *args, **kwargs):
        losses = [loss(*args, **kwargs) for loss in self]
        if self.verbose:
            for i, loss in enumerate(losses):
                print(f'({i}): {loss.item():g}')
        return sum(loss.to(losses[-1].device) for loss in losses)


class Scale(nn.Module):
    def __init__(self, module, scale):
        super().__init__()
        self.module = module
        self.register_buffer('scale', torch.tensor(scale))

    def extra_repr(self):
        return f'(scale): {self.scale.item():g}'

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs) * self.scale


class LayerApply(nn.Module):
    def __init__(self, module, layer):
        super().__init__()
        self.module = module
        self.layer = layer

    def extra_repr(self):
        return f'(layer): {self.layer!r}'

    def forward(self, input):
        return self.module(input[self.layer])


class EMA(nn.Module):
    """A bias-corrected exponential moving average, as in Kingma et al. (Adam)."""

    def __init__(self, input, decay):
        super().__init__()
        self.register_buffer('value', torch.zeros_like(input))
        self.register_buffer('decay', torch.tensor(decay))
        self.register_buffer('accum', torch.tensor(1.))
        self.update(input)

    def get(self):
        return self.value / (1 - self.accum)

    def update(self, input):
        self.accum *= self.decay
        self.value *= self.decay
        self.value += (1 - self.decay) * input


def size_to_fit(size, max_dim, scale_up=False):
    w, h = size
    if not scale_up and max(h, w) <= max_dim:
        return w, h
    new_w, new_h = max_dim, max_dim
    if h > w:
        new_w = round(max_dim * w / h)
    else:
        new_h = round(max_dim * h / w)
    return new_w, new_h


def gen_scales(start, end):
    scale = end
    i = 0
    scales = set()
    while scale >= start:
        scales.add(scale)
        i += 1
        scale = round(end / pow(2, i/2))
    return sorted(scales)


def interpolate(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return F.interpolate(*args, **kwargs)


def scale_adam(state, shape):
    """Prepares a state dict to warm-start the Adam optimizer at a new scale."""
    state = copy.deepcopy(state)
    for group in state['state'].values():
        exp_avg, exp_avg_sq = group['exp_avg'], group['exp_avg_sq']
        group['exp_avg'] = interpolate(exp_avg, shape, mode='bicubic')
        group['exp_avg_sq'] = interpolate(exp_avg_sq, shape, mode='bilinear').relu_()
        if 'max_exp_avg_sq' in group:
            max_exp_avg_sq = group['max_exp_avg_sq']
            group['max_exp_avg_sq'] = interpolate(max_exp_avg_sq, shape, mode='bilinear').relu_()
    return state


@dataclass
class STIterate:
    w: int
    h: int
    i: int
    i_max: int
    loss: float
    time: float
    gpu_ram: int


class StyleTransfer:
    def __init__(self, devices=['cpu'], pooling='max'):
        self.devices = [torch.device(device) for device in devices]
        self.image = None
        self.average = None

        # The default content and style layers follow Gatys et al. (2015).
        self.content_layers = [22]
        self.style_layers = [1, 6, 11, 20, 29]

        # The weighting of the style layers differs from Gatys et al. (2015) and Johnson et al.
        style_weights = [256, 64, 16, 4, 1]
        weight_sum = sum(abs(w) for w in style_weights)
        self.style_weights = [w / weight_sum for w in style_weights]

        self.model = VGGFeatures(self.style_layers + self.content_layers, pooling=pooling)

        if len(self.devices) == 1:
            device_plan = {0: self.devices[0]}
        elif len(self.devices) == 2:
            device_plan = {0: self.devices[0], 5: self.devices[1]}
        else:
            raise ValueError('Only 1 or 2 devices are supported.')

        self.model.distribute_layers(device_plan)

    def get_image_tensor(self):
        return self.average.get().detach()[0].clamp(0, 1)

    def get_image(self, image_type='pil'):
        if self.average is not None:
            image = self.get_image_tensor()
            if image_type.lower() == 'pil':
                return TF.to_pil_image(image)
            elif image_type.lower() == 'np_uint16':
                arr = image.cpu().movedim(0, 2).numpy()
                return np.uint16(np.round(arr * 65535))
            else:
                raise ValueError("image_type must be 'pil' or 'np_uint16'")

    def stylize(self, content_image, style_images, *,
                style_weights=None,
                content_weight: float = 0.015,
                tv_weight: float = 2.,
                optimizer: str = 'adam',
                min_scale: int = 128,
                end_scale: int = 512,
                iterations: int = 500,
                initial_iterations: int = 1000,
                step_size: float = 0.02,
                avg_decay: float = 0.99,
                init: str = 'content',
                style_scale_fac: float = 1.,
                style_size: int = None,
                color_preservation: str = 'luminance',
                callback=None):

        min_scale = min(min_scale, end_scale)
        content_weights = [content_weight / len(self.content_layers)] * len(self.content_layers)

        if style_weights is None:
            style_weights = [1 / len(style_images)] * len(style_images)
        else:
            weight_sum = sum(abs(w) for w in style_weights)
            style_weights = [weight / weight_sum for weight in style_weights]
        if len(style_images) != len(style_weights):
            raise ValueError('style_images and style_weights must have the same length')

        tv_loss = Scale(LayerApply(TVLoss(), 'input'), tv_weight)

        # === COLOR PRESERVATION SETUP ===
        original_content_pil = content_image.copy() # Keep original for chrominance
        content_y_target = None
        content_cb_orig = None
        content_cr_orig = None

        if color_preservation == 'luminance':
            print("Using Luminance Transfer for Color Preservation.")
            # Convert content image and keep original Cb, Cr
            content_y_target, content_cb_orig, content_cr_orig = pil_rgb_to_ycbcr_channels(original_content_pil)
            # The 'content_image' variable will now conceptually represent the Luminance channel PIL equivalent
            # We might need to convert Y channel tensor back to PIL temporarily if resizing logic relies on it,
            # or adjust resizing logic. Let's assume resizing happens on tensor.
            content_input_pil_or_tensor = content_y_target # Use Y channel tensor for sizing/resizing logic
            content_is_tensor = True
        else:
            print("Using RGB space for Style Transfer (no color preservation).")
            content_input_pil_or_tensor = content_image
            content_is_tensor = False

        # Determine scales
        scales = gen_scales(min_scale, end_scale)

        # Determine initial size based on the input (might need adjustment based on above)
        if content_is_tensor:
            ch_in, cw_in = content_input_pil_or_tensor.shape[-2:]
        else: # PIL
            cw_in, ch_in = content_input_pil_or_tensor.size
        cw, ch = size_to_fit((cw_in, ch_in), scales[0], scale_up=True)

        # === INITIAL IMAGE SETUP ===
        if init == 'content':
            if color_preservation == 'luminance':
                # Initialize with content's luminance, resized
                self.image = interpolate(content_y_target.unsqueeze(0), (ch, cw), mode='bicubic').clamp(0, 1)
            else:
                # Original RGB initialization
                self.image = TF.to_tensor(content_image.resize((cw, ch), Image.BICUBIC))[None]
        # ... (other init options need careful handling for luminance)
        # E.g., 'gray', 'uniform', 'normal' should probably initialize a single-channel tensor
        elif init in ('gray', 'uniform', 'normal'):
            if color_preservation == 'luminance':
                if init == 'gray':
                    self.image = torch.rand([1, 1, ch, cw]) / 255 + 0.5 # Single channel
                elif init == 'uniform':
                    self.image = torch.rand([1, 1, ch, cw]) # Single channel
                elif init == 'normal':
                    self.image = torch.empty([1, 1, ch, cw]) # Single channel
                    nn.init.trunc_normal_(self.image, mean=0.5, std=0.25, a=0, b=1)
            else: # Original RGB init
                if init == 'gray':
                    self.image = torch.rand([1, 3, ch, cw]) / 255 + 0.5
                elif init == 'uniform':
                    self.image = torch.rand([1, 3, ch, cw])
                elif init == 'normal':
                    self.image = torch.empty([1, 3, ch, cw])
                    nn.init.trunc_normal_(self.image, mean=0.5, std=0.25, a=0, b=1)
        # ... handle 'style_stats' init similarly (calculate stats on Y channel if preserving color) ...
        else:
            raise ValueError("init must be one of 'content', 'gray', 'uniform', 'style_mean'") # Or 'normal' was missing before? Added it based on code.

        self.image = self.image.to(self.devices[0])
        opt = None

        # === MULTI-SCALE LOOP ===
        for scale in scales:
            # ... (clear cache) ...

            # --- Resize Content Target ---
            if content_is_tensor:
                ch_in, cw_in = content_input_pil_or_tensor.shape[-2:]
            else: # PIL
                cw_in, ch_in = content_input_pil_or_tensor.size
            cw, ch = size_to_fit((cw_in, ch_in), scale, scale_up=True)

            if color_preservation == 'luminance':
                # Resize the target Y channel tensor
                content_y_scaled = interpolate(content_y_target.unsqueeze(0), (ch, cw), mode='bicubic').clamp(0, 1)
                # VGG expects 3 channels, so replicate the Y channel
                content_vgg_input = content_y_scaled.repeat(1, 3, 1, 1).to(self.devices[0])
            else:
                # Original RGB resizing
                content = TF.to_tensor(content_image.resize((cw, ch), Image.BICUBIC))[None]
                content_vgg_input = content.to(self.devices[0])

            # --- Resize Current Image ---
            # self.image will be (1, 1, H, W) if preserving color, (1, 3, H, W) otherwise
            self.image = interpolate(self.image.detach(), (ch, cw), mode='bicubic').clamp(0, 1)
            self.average = EMA(self.image, avg_decay)
            self.image.requires_grad_()

            print(f'Processing content features ({cw}x{ch})...')
            # Content loss uses the VGG input prepared above
            content_feats = self.model(content_vgg_input, layers=self.content_layers)
            content_losses = []
            for layer, weight in zip(self.content_layers, content_weights):
                target = content_feats[layer].detach() # Detach targets
                # ContentLossMSE expects input and target of same shape.
                # The model 'feats' output will have features derived from the 3-channel replicated Y.
                content_losses.append(Scale(LayerApply(ContentLossMSE(target), layer), weight))

            # --- Prepare Style Targets ---
            style_targets, style_losses = {}, []
            for i, style_pil_image in enumerate(style_images):
                if color_preservation == 'luminance':
                    # Convert style to Y channel, resize, replicate for VGG
                    style_y, _, _ = pil_rgb_to_ycbcr_channels(style_pil_image)
                    if style_size is None:
                        sw, sh = size_to_fit(style_pil_image.size, round(scale * style_scale_fac))
                    else:
                        sw, sh = size_to_fit(style_pil_image.size, style_size)
                    style_y_scaled = interpolate(style_y.unsqueeze(0), (sh, sw), mode='bicubic').clamp(0, 1)
                    style_vgg_input = style_y_scaled.repeat(1, 3, 1, 1).to(self.devices[0])
                else:
                    # Original RGB style processing
                    if style_size is None:
                        sw, sh = size_to_fit(style_pil_image.size, round(scale * style_scale_fac))
                    else:
                        sw, sh = size_to_fit(style_pil_image.size, style_size)
                    style = TF.to_tensor(style_pil_image.resize((sw, sh), Image.BICUBIC))[None]
                    style_vgg_input = style.to(self.devices[0])

                print(f'Processing style features ({sw}x{sh})...')
                # Style loss uses the VGG input prepared above
                style_feats = self.model(style_vgg_input, layers=self.style_layers)

                for layer in self.style_layers:
                    # Style Loss calculation remains the same structurally, but operates on features from Y channel
                    target_mean, target_cov = StyleLossW2.get_target(style_feats[layer].detach()) # Detach targets
                    target_mean *= style_weights[i]
                    target_cov *= style_weights[i]
                    # ... (rest of style target aggregation is the same) ...
                    if layer not in style_targets:
                        style_targets[layer] = target_mean, target_cov
                    else:
                        style_targets[layer][0].add_(target_mean)
                        style_targets[layer][1].add_(target_cov)

            for layer, weight in zip(self.style_layers, self.style_weights):
                target = style_targets[layer]
                # StyleLossW2 input will be features derived from the optimized Y channel (replicated)
                style_losses.append(Scale(LayerApply(StyleLossW2(target), layer), weight))

            crit = SumLoss([*content_losses, *style_losses, tv_loss])

            # --- Optimizer Setup (Adam part needs adjustment if image changes channels) ---
            if optimizer == 'adam':
                opt2 = optim.Adam([self.image], lr=step_size, betas=(0.9, 0.99))
                if scale != scales[0] and opt is not None: # Check opt exists
                    # Check if channels changed (e.g., switching modes - unlikely but possible)
                    # If channels are consistent within a run, scale_adam might work as is,
                    # otherwise it might need adjustment if channels differ. Assuming consistent channels here.
                    try:
                        opt_state = scale_adam(opt.state_dict(), (ch, cw))
                        opt2.load_state_dict(opt_state)
                    except RuntimeError as e:
                        print(f"Warning: Could not load optimizer state at scale {scale}. Reinitializing Adam. Error: {e}")
                        opt2 = optim.Adam([self.image], lr=step_size, betas=(0.9, 0.99)) # Reinitialize fully
                opt = opt2
            # ... (LBFGS remains similar conceptually) ...
            elif optimizer == 'lbfgs':
                opt = optim.LBFGS([self.image], max_iter=1, history_size=10)
            else:
                 raise ValueError("optimizer must be one of 'adam', 'lbfgs'")

            # ... (clear cache) ...

            # --- Optimization Loop ---
            def closure():
                opt.zero_grad()
                # === IMPORTANT: Replicate Y channel image for VGG input ===
                if color_preservation == 'luminance':
                    model_input = self.image.repeat(1, 3, 1, 1) # Replicate optimized Y
                else:
                    model_input = self.image # Use RGB image directly

                feats = self.model(model_input) # Pass appropriate input to VGG
                loss = crit(feats)
                loss.backward()
                # Gradient should flow back to the single-channel self.image if preserving color
                return loss

            actual_its = initial_iterations if scale == scales[0] else iterations
            for i in range(1, actual_its + 1):
                # ... (opt.step, clamping, average update are okay) ...
                # Clamping should still be [0, 1] as Y is also in this range
                loss = opt.step(closure)
                if optimizer != 'lbfgs':
                    with torch.no_grad():
                        self.image.clamp_(0, 1)
                self.average.update(self.image)
                # ... (callback is okay) ...
                if callback is not None:
                    # ... (callback logic) ...

                    # --- Prepare for Next Scale ---
                    with torch.no_grad():
                        self.image.copy_(self.average.get())
                        # self.image remains single channel if preserving color

        # === FINAL IMAGE GENERATION ===
        if color_preservation == 'luminance':
            final_y = self.average.get().detach().cpu() # Final optimized Y channel (1, 1, H, W)
            # Resize original Cb, Cr channels to match the final Y size
            final_h, final_w = final_y.shape[-2:]
            # Resize requires PIL or careful tensor interpolation. Let's use PIL temporarily.
            orig_cbcr_pil = combine_y_cbcr_to_pil(torch.zeros_like(content_cb_orig), content_cb_orig, content_cr_orig) # Dummy Y needed
            resized_cbcr_pil = orig_cbcr_pil.resize((final_w, final_h), Image.BICUBIC)
            _, final_cb, final_cr = pil_rgb_to_ycbcr_channels(resized_cbcr_pil) # Extract resized Cb, Cr

            # Combine final Y with original (resized) Cb, Cr
            # Ensure device consistency if needed before combining
            final_image_pil = combine_y_cbcr_to_pil(final_y, final_cb, final_cr)
            return final_image_pil
        else:
            # Original RGB output generation
            return self.get_image() # Uses self.average which is RGB
