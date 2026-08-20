import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.library import register_fake
from train_log.IFNet_helpers import (
    TimestepFold,
    fold_scale_state_dict,
    fold_timestep_state_dict,
    fold_warp_normalization_state_dict,
    prune_lastconv_state_dict,
)
# from train_log.refine import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def warp(tenInput, tenFlow):
    tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
        1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
    tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
        1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
    g = (torch.cat([tenHorizontal, tenVertical], 1).to(device) + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

def downsample_even_scale(tenInput, scale):
    return F.interpolate(
        tenInput,
        scale_factor=1.0 / scale,
        mode="bilinear",
        align_corners=False,
    )

@torch.library.custom_op("rife::warp_downsample8", mutates_args=())
def warp_downsample8(tenInput: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    return downsample_even_scale(warp(tenInput, tenFlow), 8)

@torch.library.custom_op("rife::warp_downsample4", mutates_args=())
def warp_downsample4(tenInput: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    return downsample_even_scale(warp(tenInput, tenFlow), 4)

@torch.library.custom_op("rife::warp_downsample2", mutates_args=())
def warp_downsample2(tenInput: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    return downsample_even_scale(warp(tenInput, tenFlow), 2)

def _warp_downsample_fake_output(
    tenInput: torch.Tensor,
    tenFlow: torch.Tensor,
    scale: int,
) -> torch.Tensor:
    _ = tenFlow
    return torch.empty(
        (
            tenInput.shape[0],
            tenInput.shape[1],
            tenInput.shape[2] // scale,
            tenInput.shape[3] // scale,
        ),
        dtype=tenInput.dtype,
        device=tenInput.device,
    )

@register_fake("rife::warp_downsample8")
def _warp_downsample8_fake(tenInput: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    return _warp_downsample_fake_output(tenInput, tenFlow, 8)

@register_fake("rife::warp_downsample4")
def _warp_downsample4_fake(tenInput: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    return _warp_downsample_fake_output(tenInput, tenFlow, 4)

@register_fake("rife::warp_downsample2")
def _warp_downsample2_fake(tenInput: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    return _warp_downsample_fake_output(tenInput, tenFlow, 2)

def _make_warp_downsample_setup(scale):
    def setup_context(ctx, inputs, output):
        _ = output
        ctx.scale = scale
        ctx.save_for_backward(inputs[0], inputs[1])

    return setup_context

def _make_warp_downsample_backward(scale):
    def backward(ctx, grad_output):
        tenInput, tenFlow = ctx.saved_tensors
        input_requires_grad = tenInput.is_floating_point()
        flow_requires_grad = tenFlow.is_floating_point()
        if not input_requires_grad and not flow_requires_grad:
            return None, None, None

        with torch.enable_grad():
            tenInput_req = tenInput.detach()
            tenFlow_req = tenFlow.detach()
            if input_requires_grad:
                tenInput_req.requires_grad_(True)
            if flow_requires_grad:
                tenFlow_req.requires_grad_(True)
            result = downsample_even_scale(
                warp(tenInput_req, tenFlow_req),
                scale,
            )
            grad_input, grad_flow = torch.autograd.grad(
                result,
                (tenInput_req, tenFlow_req),
                grad_output,
                allow_unused=True,
            )
        return grad_input, grad_flow

    return backward

for _warp_downsample_scale in (8, 4, 2):
    torch.library.register_autograd(
        f"rife::warp_downsample{_warp_downsample_scale}",
        _make_warp_downsample_backward(_warp_downsample_scale),
        setup_context=_make_warp_downsample_setup(_warp_downsample_scale),
    )

def warp_downsample_for_scale(tenInput, tenFlow, scale):
    if scale == 8:
        return torch.ops.rife.warp_downsample8.default(tenInput, tenFlow)
    if scale == 4:
        return torch.ops.rife.warp_downsample4.default(tenInput, tenFlow)
    if scale == 2:
        return torch.ops.rife.warp_downsample2.default(tenInput, tenFlow)
    raise ValueError(f"Unsupported warp downsample scale: {scale}")

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),        
        nn.LeakyReLU(0.2, True)
    )

def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, True)
    )

class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x, feat=False):
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        if feat:
            return [x0, x1, x2, x3]
        return x3

class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.relu = nn.LeakyReLU(0.2, True)
        with torch.no_grad():
            self._add_residual_identity_(self.conv.weight)

    @staticmethod
    def _add_residual_identity_(weight):
        channels = weight.shape[0]
        center_h = weight.shape[2] // 2
        center_w = weight.shape[3] // 2
        channel_idx = torch.arange(channels, device=weight.device)
        weight[channel_idx, channel_idx, center_h, center_w] += 1.0

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        beta_key = prefix + "beta"
        weight_key = prefix + "conv.weight"
        bias_key = prefix + "conv.bias"
        if beta_key in state_dict and weight_key in state_dict:
            beta = state_dict.pop(beta_key).reshape(-1)
            fused_weight = state_dict[weight_key].clone()
            fused_weight *= beta.reshape(-1, 1, 1, 1)
            self._add_residual_identity_(fused_weight)
            state_dict[weight_key] = fused_weight
            if bias_key in state_dict:
                state_dict[bias_key] = state_dict[bias_key].clone() * beta
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        return self.relu(self.conv(x))

class IFBlock(nn.Module):
    _version = 3

    timestep_fold = TimestepFold.TIMESTEP_CONCAT
    fixed_timestep = 0.5

    def __init__(
        self,
        in_planes,
        c=64,
        timestep_insert_channel=14,
        scale=1,
        output_feat=True,
        input_flow_channels=0,
    ):
        super(IFBlock, self).__init__()
        self.output_feat = output_feat
        self.scale = scale
        self.input_flow_channels = input_flow_channels
        self.timestep_insert_channel = timestep_insert_channel
        self._timestep_scale = scale
        self._timestep_input_height = 768
        self._timestep_input_width = 384
        self._flow_input_height = 768
        self._flow_input_width = 384
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * (13 if output_feat else 5), 4, 2, 1),
            nn.PixelShuffle(2)
        )
        if self.timestep_fold == TimestepFold.TIMESTEP_ADD:
            height = self._timestep_input_height // scale
            width = self._timestep_input_width // scale
            self.register_buffer(
                "conv0_timestep_bias",
                torch.zeros((1, c // 2, (height + 1) // 2, (width + 1) // 2)),
            )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        weight_key = prefix + "conv0.0.0.weight"
        if weight_key in state_dict:
            fold_timestep_state_dict(self, state_dict, prefix, weight_key)
        if not self.output_feat:
            prune_lastconv_state_dict(self, state_dict, prefix)
        version = local_metadata.get("version")
        if version is None or version < 2:
            fold_scale_state_dict(self, state_dict, prefix)
        if version is None or version < self._version:
            fold_warp_normalization_state_dict(self, state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _interpolate_for_scale(self, x, scale):
        if scale == 16:
            x = F.interpolate(
                x, scale_factor=1.0 / 8.0, mode="bilinear", align_corners=False
            )
            return F.interpolate(
                x, scale_factor=1.0 / 2.0, mode="bilinear", align_corners=False
            )
        else:      
            return F.interpolate(
                x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
            )

    def _prepare_input(self, x, scale, native_scale_inputs=(), trailing_inputs=()):
        if not isinstance(x, tuple):
            x = (x,)
        resized_inputs = tuple(self._interpolate_for_scale(part, scale) for part in x)
        resized_trailing = tuple(
            self._interpolate_for_scale(part, scale) for part in trailing_inputs
        )
        return torch.cat((*resized_inputs, *native_scale_inputs, *resized_trailing), 1)

    @staticmethod
    def _resize_output_for_scale(x, current_scale, output_scale):
        scale_factor = float(current_scale) / float(output_scale)
        if scale_factor == 1.0:
            return x
        return F.interpolate(
            x,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        x,
        native_scale_inputs=(),
        trailing_inputs=(),
        output_scale=1,
    ):
        scale = self.scale
        x = self._prepare_input(x, scale, native_scale_inputs, trailing_inputs)
        if self.timestep_fold == TimestepFold.TIMESTEP_CONCAT:
            timestep = x.new_full(
                (x.shape[0], 1, x.shape[2], x.shape[3]), self.fixed_timestep
            )
            insert_channel = self.timestep_insert_channel
            x = torch.cat((x[:, :insert_channel], timestep, x[:, insert_channel:]), 1)
        if self.timestep_fold == TimestepFold.TIMESTEP_ADD:
            feat = self.conv0[0][0](x) + self.conv0_timestep_bias
            feat = self.conv0[0][1](feat)
            feat = self.conv0[1](feat)
        else:
            feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        flow = F.interpolate(
            tmp[:, :4], scale_factor=scale, mode="bilinear", align_corners=False
        )
        mask = self._resize_output_for_scale(tmp[:, 4:5], scale, output_scale)
        feat = self._resize_output_for_scale(tmp[:, 5:], scale, output_scale)
        return flow, mask, feat
class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7+8, c=192, scale=16)
        self.block1 = IFBlock(8+4+8+8, c=128, scale=8, input_flow_channels=4)
        self.block2 = IFBlock(8+4+8+8, c=96, scale=4, input_flow_channels=4)
        self.block3 = IFBlock(8+4+8+8, c=64, scale=2, input_flow_channels=4)
        self.block4 = IFBlock(8+4+8+8, c=32, scale=1, output_feat=False, input_flow_channels=4)
        self.encode = Head()


    def forward(self, x, timestep=0.5, scale_list=[8, 4, 2, 1]):
        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]

        # Encode input to flow
        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])

        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None

        # --- Block 0 ---
        flow, mask, feat = self.block0(
            (img0[:, :3], img1[:, :3], f0, f1),
            output_scale=self.block1.scale,
        )

        warped_img0 = warp_downsample_for_scale(img0, flow[:, :2], self.block1.scale)
        warped_img1 = warp_downsample_for_scale(img1, flow[:, 2:4], self.block1.scale)
        wf0 = warp_downsample_for_scale(f0, flow[:, :2], self.block1.scale)
        wf1 = warp_downsample_for_scale(f1, flow[:, 2:4], self.block1.scale)

        # --- Block 1 ---
        fd, mask, feat = self.block1(
            (),
            native_scale_inputs=(
                warped_img0[:, :3], warped_img1[:, :3], wf0, wf1, mask, feat
            ),
            trailing_inputs=(flow,),
            output_scale=self.block2.scale,
        )
        flow = flow + fd

        warped_img0 = warp_downsample_for_scale(img0, flow[:, :2], self.block2.scale)
        warped_img1 = warp_downsample_for_scale(img1, flow[:, 2:4], self.block2.scale)
        wf0 = warp_downsample_for_scale(f0, flow[:, :2], self.block2.scale)
        wf1 = warp_downsample_for_scale(f1, flow[:, 2:4], self.block2.scale)

        # --- Block 2 ---
        fd, mask, feat = self.block2(
            (),
            native_scale_inputs=(
                warped_img0[:, :3], warped_img1[:, :3], wf0, wf1, mask, feat
            ),
            trailing_inputs=(flow,),
            output_scale=self.block3.scale,
        )
        flow = flow + fd

        warped_img0 = warp_downsample_for_scale(img0, flow[:, :2], self.block3.scale)
        warped_img1 = warp_downsample_for_scale(img1, flow[:, 2:4], self.block3.scale)
        wf0 = warp_downsample_for_scale(f0, flow[:, :2], self.block3.scale)
        wf1 = warp_downsample_for_scale(f1, flow[:, 2:4], self.block3.scale)

        # --- Block 3 ---
        fd, mask, feat = self.block3(
            (),
            native_scale_inputs=(
                warped_img0[:, :3], warped_img1[:, :3], wf0, wf1, mask, feat
            ),
            trailing_inputs=(flow,),
            output_scale=self.block4.scale,
        )
        flow = flow + fd

        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        wf0 = warp(f0, flow[:, :2])
        wf1 = warp(f1, flow[:, 2:4])

        # --- Block 4 ---
        fd, mask, feat = self.block4(
            (warped_img0[:, :3], warped_img1[:, :3], wf0, wf1),
            native_scale_inputs=(mask, feat),
            trailing_inputs=(flow,),
        )
        flow = flow + fd
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])

        # Linear interpolation between warped img0 and warped img1
        mask = torch.sigmoid(mask)
        result = (warped_img0 * mask + warped_img1 * (1 - mask))

        return None, None, [result]
