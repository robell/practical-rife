from enum import Enum

import torch
import torch.nn.functional as F


class TimestepFold(Enum):
    TIMESTEP_ADD = "add"
    TIMESTEP_CONCAT = "concat"
    TIMESTEP_BIAS = "bias"


def as_timestep_fold(timestep_fold):
    if isinstance(timestep_fold, TimestepFold):
        return timestep_fold
    if isinstance(timestep_fold, str) and timestep_fold in TimestepFold.__members__:
        return TimestepFold[timestep_fold]
    return TimestepFold(timestep_fold)


def fold_timestep_state_dict(block, state_dict, prefix, weight_key):
    if block.timestep_fold == TimestepFold.TIMESTEP_ADD:
        _fold_timestep_to_add(block, state_dict, prefix, weight_key)
    elif block.timestep_fold == TimestepFold.TIMESTEP_BIAS:
        _fold_timestep_to_bias(block, state_dict, prefix, weight_key)
    else:
        _fold_timestep_to_concat(block, state_dict, prefix, weight_key)


def prune_lastconv_state_dict(block, state_dict, prefix):
    weight_key = prefix + "lastconv.0.weight"
    bias_key = prefix + "lastconv.0.bias"
    if weight_key in state_dict:
        expected_channels = block.lastconv[0].weight.shape[1]
        if state_dict[weight_key].shape[1] > expected_channels:
            state_dict[weight_key] = state_dict[weight_key][
                :, :expected_channels
            ].clone()
    if bias_key in state_dict:
        expected_channels = block.lastconv[0].bias.shape[0]
        if state_dict[bias_key].shape[0] > expected_channels:
            state_dict[bias_key] = state_dict[bias_key][:expected_channels].clone()


def fold_scale_state_dict(block, state_dict, prefix):
    scale = float(block.scale)
    if scale == 1.0:
        return
    if block.input_flow_channels:
        weight_key = prefix + "conv0.0.0.weight"
        if weight_key in state_dict:
            weight = state_dict[weight_key].clone()
            weight[:, -block.input_flow_channels:] /= scale
            state_dict[weight_key] = weight
    weight_key = prefix + "lastconv.0.weight"
    bias_key = prefix + "lastconv.0.bias"
    flow_output_channels = 4 * block.pixel_shuffle_factor**2
    if weight_key in state_dict:
        weight = state_dict[weight_key].clone()
        weight[:, :flow_output_channels] *= scale
        state_dict[weight_key] = weight
    if bias_key in state_dict:
        bias = state_dict[bias_key].clone()
        bias[:flow_output_channels] *= scale
        state_dict[bias_key] = bias


def _recover_constant_timestep_weight(timestep_bias, timestep):
    bias = timestep_bias[0] / timestep
    interior = bias[:, 1, 1]
    top = bias[:, 0, 1]
    left = bias[:, 1, 0]
    corner = bias[:, 0, 0]
    weight = timestep_bias.new_zeros((bias.shape[0], 1, 3, 3))
    weight[:, 0, 0, 0] = interior - top - left + corner
    weight[:, 0, 0, 1] = left - corner
    weight[:, 0, 1, 0] = top - corner
    weight[:, 0, 1, 1] = corner
    return weight


def _remove_timestep_weight(state_dict, weight_key, timestep_channel):
    weight = state_dict[weight_key].clone()
    timestep_weight = weight[:, timestep_channel:timestep_channel + 1]
    state_dict[weight_key] = torch.cat(
        (weight[:, :timestep_channel], weight[:, timestep_channel + 1:]), 1
    )
    return timestep_weight


def _fold_timestep_to_add(block, state_dict, prefix, weight_key):
    timestep_bias_key = prefix + "conv0_timestep_bias"
    if state_dict[weight_key].shape[1] == block.conv0[0][0].weight.shape[1] + 1:
        timestep_weight = _remove_timestep_weight(
            state_dict, weight_key, block.timestep_insert_channel
        )
        timestep_input = torch.full(
            (
                1,
                1,
                block._timestep_input_height,
                block._timestep_input_width,
            ),
            block.fixed_timestep,
            dtype=timestep_weight.dtype,
            device=timestep_weight.device,
        )
        timestep_input = F.interpolate(
            timestep_input,
            scale_factor=1.0 / block._timestep_scale,
            mode="bilinear",
            align_corners=False,
        )
        state_dict[timestep_bias_key] = F.conv2d(
            timestep_input, timestep_weight, stride=2, padding=1
        )
    elif timestep_bias_key not in state_dict:
        state_dict[timestep_bias_key] = block.conv0_timestep_bias


def _fold_timestep_to_bias(block, state_dict, prefix, weight_key):
    bias_key = prefix + "conv0.0.0.bias"
    timestep_bias_key = prefix + "conv0_timestep_bias"
    if state_dict[weight_key].shape[1] == block.conv0[0][0].weight.shape[1] + 1:
        timestep_weight = _remove_timestep_weight(
            state_dict, weight_key, block.timestep_insert_channel
        )
        if bias_key in state_dict:
            state_dict[bias_key] = (
                state_dict[bias_key].clone()
                + timestep_weight.sum(dim=(1, 2, 3)) * block.fixed_timestep
            )
    if timestep_bias_key in state_dict:
        if bias_key in state_dict:
            state_dict[bias_key] = (
                state_dict[bias_key].clone()
                + state_dict[timestep_bias_key][0, :, 1, 1]
            )
        state_dict.pop(timestep_bias_key)


def _fold_timestep_to_concat(block, state_dict, prefix, weight_key):
    timestep_bias_key = prefix + "conv0_timestep_bias"
    weight = state_dict[weight_key]
    expected_channels = block.conv0[0][0].weight.shape[1]
    if weight.shape[1] == expected_channels - 1:
        if timestep_bias_key in state_dict:
            timestep_weight = _recover_constant_timestep_weight(
                state_dict.pop(timestep_bias_key),
                block.fixed_timestep,
            )
        else:
            timestep_weight = weight.new_zeros((weight.shape[0], 1, 3, 3))
        insert_channel = block.timestep_insert_channel
        state_dict[weight_key] = torch.cat(
            (
                weight[:, :insert_channel],
                timestep_weight,
                weight[:, insert_channel:],
            ),
            1,
        )
    elif timestep_bias_key in state_dict:
        state_dict.pop(timestep_bias_key)
