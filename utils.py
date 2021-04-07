"""Utility functions."""

import base64
import os
import subprocess
import cv2
import numpy as np

import torch

from models import MODEL_ZOO
from models import build_generator
from models import parse_gan_type

__all__ = ['postprocess', 'load_generator', 'factorize_weight',
           'HtmlPageVisualizer']

CHECKPOINT_DIR = 'checkpoints'


def to_tensor(array):
    """Converts a `numpy.ndarray` to `torch.Tensor`.

    Args:
      array: The input array to convert.

    Returns:
      A `torch.Tensor` with dtype `torch.FloatTensor` on cuda device.
    """
    assert isinstance(array, np.ndarray)
    return torch.from_numpy(array).type(torch.FloatTensor).cuda()


def postprocess(images, min_val=-1.0, max_val=1.0):
    """Post-processes images from `torch.Tensor` to `numpy.ndarray`.

    Args:
        images: A `torch.Tensor` with shape `NCHW` to process.
        min_val: The minimum value of the input tensor. (default: -1.0)
        max_val: The maximum value of the input tensor. (default: 1.0)

    Returns:
        A `numpy.ndarray` with shape `NHWC` and pixel range [0, 255].
    """
    assert isinstance(images, torch.Tensor)
    images = images.detach().cpu().numpy()
    images = (images - min_val) * 255 / (max_val - min_val)
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    return images


def load_generator(model_name):
    """Loads pre-trained generator.

    Args:
        model_name: Name of the model. Should be a key in `models.MODEL_ZOO`.

    Returns:
        A generator, which is a `torch.nn.Module`, with pre-trained weights
            loaded.

    Raises:
        KeyError: If the input `model_name` is not in `models.MODEL_ZOO`.
    """
    if model_name not in MODEL_ZOO:
        raise KeyError(f'Unknown model name `{model_name}`!')

    model_config = MODEL_ZOO[model_name].copy()
    url = model_config.pop('url')  # URL to download model if needed.

    # Build generator.
    print(f'Building generator for model `{model_name}` ...')
    generator = build_generator(**model_config)
    print(f'Finish building generator.')

    # Load pre-trained weights.
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, model_name + '.pth')
    print(f'Loading checkpoint from `{checkpoint_path}` ...')
    if not os.path.exists(checkpoint_path):
        print(f'  Downloading checkpoint from `{url}` ...')
        subprocess.call(['wget', '--quiet', '-O', checkpoint_path, url])
        print(f'  Finish downloading checkpoint.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'generator_smooth' in checkpoint:
        generator.load_state_dict(checkpoint['generator_smooth'])
    else:
        generator.load_state_dict(checkpoint['generator'])
    generator = generator.cuda()
    generator.eval()
    print(f'Finish loading checkpoint.')
    return generator


def parse_indices(obj, min_val=None, max_val=None):
    """Parses indices.

    The input can be a list or a tuple or a string, which is either a comma
    separated list of numbers 'a, b, c', or a dash separated range 'a - c'.
    Space in the string will be ignored.

    Args:
        obj: The input object to parse indices from.
        min_val: If not `None`, this function will check that all indices are
            equal to or larger than this value. (default: None)
        max_val: If not `None`, this function will check that all indices are
            equal to or smaller than this value. (default: None)

    Returns:
        A list of integers.

    Raises:
        If the input is invalid, i.e., neither a list or tuple, nor a string.
    """
    if obj is None or obj == '':
        indices = []
    elif isinstance(obj, int):
        indices = [obj]
    elif isinstance(obj, (list, tuple, np.ndarray)):
        indices = list(obj)
    elif isinstance(obj, str):
        indices = []
        splits = obj.replace(' ', '').split(',')
        for split in splits:
            numbers = list(map(int, split.split('-')))
            if len(numbers) == 1:
                indices.append(numbers[0])
            elif len(numbers) == 2:
                indices.extend(list(range(numbers[0], numbers[1] + 1)))
            else:
                raise ValueError(f'Unable to parse the input!')

    else:
        raise ValueError(f'Invalid type of input: `{type(obj)}`!')

    assert isinstance(indices, list)
    indices = sorted(list(set(indices)))
    for idx in indices:
        assert isinstance(idx, int)
        if min_val is not None:
            assert idx >= min_val, f'{idx} is smaller than min val `{min_val}`!'
        if max_val is not None:
            assert idx <= max_val, f'{idx} is larger than max val `{max_val}`!'

    return indices

def get_weights(generator, layer_idx='all', apply_norm=True):
    """Obtains weight matrix from specified generator and layer selection. Adapted from `factorize_weights`

    Args:
        generator: Generator to get.
        layer_idx: Indices of layers to interpret, especially for StyleGAN and
            StyleGAN2. (default: `all`)

    Returns:
        A weight matrix.

    Raises:
        ValueError: If the generator type is not supported.
    """

    # Get GAN type.
    gan_type = parse_gan_type(generator)

    # Get layers.
    if gan_type in ['stylegan', 'stylegan2']:
        if layer_idx == 'all':
            layers = list(range(generator.num_layers))
        else:
            layers = parse_indices(layer_idx,
                                   min_val=0,
                                   max_val=generator.num_layers - 1)

    # Factorize semantics from weight.
    weights = []
    for idx in layers:
        layer_name = f'layer{idx}'
        if gan_type == 'stylegan2' and idx == generator.num_layers - 1:
            layer_name = f'output{idx // 2}'
        if gan_type in ['stylegan', 'stylegan2']:
            weight = generator.synthesis.__getattr__(layer_name).style.weight.T
        weights.append(weight.cpu().detach().numpy())
    weight = np.concatenate(weights, axis=1).astype(np.float32)
    if apply_norm:
        weight = weight / np.linalg.norm(weight, axis=0, keepdims=True) # Q: is normalizing the weight values here necessary?
    return weight

def factorize_weight(generator, layer_idx='all'):
    """Factorizes the generator weight to get semantics boundaries.

    Args:
        generator: Generator to factorize.
        layer_idx: Indices of layers to interpret, especially for StyleGAN and
            StyleGAN2. (default: `all`)

    Returns:
        A tuple of (layers_to_interpret, semantic_boundaries, eigen_values).

    Raises:
        ValueError: If the generator type is not supported.
    """
    # Get GAN type.
    gan_type = parse_gan_type(generator)

    # Get layers.
    if gan_type == 'pggan':
        layers = [0]
    elif gan_type in ['stylegan', 'stylegan2']:
        if layer_idx == 'all':
            layers = list(range(generator.num_layers))
        else:
            layers = parse_indices(layer_idx,
                                   min_val=0,
                                   max_val=generator.num_layers - 1)

    # Factorize semantics from weight.
    weights = []
    for idx in layers:
        layer_name = f'layer{idx}'
        if gan_type == 'stylegan2' and idx == generator.num_layers - 1:
            layer_name = f'output{idx // 2}'
        if gan_type == 'pggan':
            weight = generator.__getattr__(layer_name).weight
            weight = weight.flip(2, 3).permute(1, 0, 2, 3).flatten(1)
        elif gan_type in ['stylegan', 'stylegan2']:
            weight = generator.synthesis.__getattr__(layer_name).style.weight.T
        weights.append(weight.cpu().detach().numpy())
    weight = np.concatenate(weights, axis=1).astype(np.float32)
    weight = weight / np.linalg.norm(weight, axis=0, keepdims=True)
    eigen_values, eigen_vectors = np.linalg.eig(weight.dot(weight.T))

    return layers, eigen_vectors.T, eigen_values


def sample(generator, gan_type, num=1, seed=0):
  """Samples latent codes."""
  torch.manual_seed(seed)
  codes = torch.randn(num, generator.z_space_dim).cuda()
  if gan_type == 'pggan':
    codes = generator.layer0.pixel_norm(codes)
  elif gan_type == 'stylegan':
    codes = generator.mapping(codes)['w']
    codes = generator.truncation(codes, trunc_psi=0.7, trunc_layers=8)
  elif gan_type == 'stylegan2':
    codes = generator.mapping(codes)['w']
    codes = generator.truncation(codes, trunc_psi=0.5, trunc_layers=18)
  codes = codes.detach().cpu().numpy()
  return codes


def synthesize(generator, gan_type, codes):
  """Synthesizes images with the give codes."""
  if gan_type == 'pggan':
    images = generator(to_tensor(codes))['image']
  elif gan_type in ['stylegan', 'stylegan2']:
    images = generator.synthesis(to_tensor(codes))['image']
  images = postprocess(images)
  return images


def imshow(images, col, viz_size=256):
  """Shows images in one figure."""
  num, height, width, channels = images.shape
  assert num % col == 0
  row = num // col

  fused_image = np.zeros((viz_size * row, viz_size * col, channels), dtype=np.uint8)

  for idx, image in enumerate(images):
    i, j = divmod(idx, col)
    y = i * viz_size
    x = j * viz_size
    if height != viz_size or width != viz_size:
      image = cv2.resize(image, (viz_size, viz_size))
    fused_image[y:y + viz_size, x:x + viz_size] = image

  fused_image = np.asarray(fused_image, dtype=np.uint8)
  data = io.BytesIO()
  PIL.Image.fromarray(fused_image).save(data, 'jpeg')
  im_data = data.getvalue()
  disp = IPython.display.display(IPython.display.Image(im_data))
  return disp