import numpy as np
import torch

_shuffle_info = {}


def _get_shuffle_info(input, upscale_factor):
    pixel_dim = input.dim() - 2
    if isinstance(upscale_factor, int):
        upscale_factor = [upscale_factor] * pixel_dim
    else:
        assert len(upscale_factor) == pixel_dim
        if isinstance(upscale_factor, tuple):
            upscale_factor = list(upscale_factor)
    key = (input.size(), *upscale_factor)

    if key not in _shuffle_info:
        size = list(input.size())
        assert size[1] % np.prod(upscale_factor) == 0
        extented_shape = [size[0]] + upscale_factor + [-1] + size[2:]

        permute_dim = [0, pixel_dim+1]
        for dim in range(2, 2+pixel_dim):
            permute_dim += [dim+pixel_dim, dim-1]

        reshape_size = [size[0], -1]
        for shape, scale in zip(size[2:], upscale_factor):
            reshape_size.append(shape*scale)

        _shuffle_info[key] = (extented_shape, permute_dim, reshape_size)

    return _shuffle_info[key]


def pixel_shuffle(input, upscale_factor):
    """pixel_shuffle"""
    extented_shape, permute_dim, reshape_size = _get_shuffle_info(
        input, upscale_factor)
    extented = input.reshape(*extented_shape)
    transposed = extented.permute(*permute_dim)
    shuffled = transposed.reshape(*reshape_size)
    return shuffled


def depth_to_space(input, block_size):
    """depth_to_space, alias of pixel_shuffle"""
    return pixel_shuffle(input, block_size)


_unshuffle_info = {}


def _get_unshuffle_info(input, block_size):
    pixel_dim = input.dim() - 2
    if isinstance(block_size, int):
        block_size = [block_size] * pixel_dim
    else:
        assert len(block_size) == pixel_dim
        if isinstance(block_size, tuple):
            block_size = list(block_size)
    key = (input.size(), *block_size)

    if key not in _unshuffle_info:
        size = list(input.size())
        reshape_size = [size[0], -1]
        for shape, scale in zip(size[2:], block_size):
            assert shape % scale == 0
            reshape_size += [shape//scale, scale]

        permute_dim = [0]
        for dim in range(pixel_dim+1, 1, -1):
            permute_dim += [dim, dim+pixel_dim]
        permute_dim.insert(pixel_dim+1, 1)

        merge_shape = [size[0], -1] + \
            reshape_size[slice(2, 2+pixel_dim+1, pixel_dim)]

        _unshuffle_info[key] = (reshape_size, permute_dim, merge_shape)

    return _unshuffle_info[key]


def pixel_unshuffle(input, block_size):
    """pixel_unshuffle"""
    reshape_size, permute_dim, merge_shape = _get_unshuffle_info(
        input, block_size)
    extented = input.reshape(*reshape_size)
    transposed = extented.permute(*permute_dim)
    merged = transposed.reshape(*merge_shape)
    return merged


def space_to_depth(input, block_size):
    """space_to_depth, alias of pixel_unshuffle"""
    return pixel_unshuffle(input, block_size)
