import torch
import torch.nn as nn
import torch.nn.functional as F
from anyfig import get_config


def calc_loss(outputs, soft_mask, styled_img, warmup=False):
  content_losses = content_loss(
    outputs['content'],
    outputs['styled_content'],
  )
  # dist_loss = distance_loss(soft_mask, styled_img)

  if warmup:
    return {**content_losses}
    # return {**content_losses, **dist_loss}

  style_losses = style_loss(
    outputs['style'],
    outputs['styled_content'],
  )

  return {**style_losses, **content_losses}
  # return {**style_losses, **content_losses, **dist_loss}


def style_loss(style, styled_content):
  config = get_config()
  losses = dict()
  shared_keys = set(style).intersection(set(styled_content))
  assert shared_keys, 'No common layers'

  layer_weights = sum([config.style_weights[l] for l in shared_keys])
  loss_scale = config.style_loss_weight / layer_weights
  for layer in shared_keys:
    x1, x2 = style[layer], styled_content[layer]
    loss = F.mse_loss(gram_matrix(x1),
                      gram_matrix(x2)) * config.style_weights[layer]
    losses[f'style-{layer}'] = loss * loss_scale

  return losses


def content_loss(content, styled_content):
  config = get_config()
  losses = dict()
  shared_keys = set(content).intersection(set(styled_content))
  assert shared_keys, 'No common layers'

  layer_weights = sum([config.content_weights[l] for l in shared_keys])
  loss_scale = config.content_loss_weight / layer_weights
  for layer in shared_keys:
    x1, x2 = content[layer], styled_content[layer]
    loss = F.mse_loss(x1, x2) * config.content_weights[layer]
    losses[f'content-{layer}'] = loss * loss_scale

  return losses


# Gram Matrix
def gram_matrix(tensor):
  B, C, H, W = tensor.shape
  x = tensor.view(B, C, H * W)
  x_t = x.transpose(1, 2)
  return torch.bmm(x, x_t) / (C * H * W)


def distance_loss(soft_mask, styled_content):
  config = get_config()
  masked_content = soft_mask * styled_content
  loss = F.mse_loss(masked_content, torch.zeros_like(masked_content))
  scaled_loss = loss * config.distance_loss_weight
  return dict(distance=scaled_loss)
