import torch
import torch.nn as nn
import torch.nn.functional as F
from anyfig import get_config


def calc_loss(outputs):
  losses = dict()
  losses['style'] = style_loss(
    outputs['style'],
    outputs['styled_content'],
  )

  losses['content'] = content_loss(
    outputs['content'],
    outputs['styled_content'],
  )

  return losses


def style_loss(style, styled_content):
  config = get_config()
  loss = 0
  shared_keys = set(style).intersection(set(styled_content))
  assert shared_keys, 'No common layers'

  for layer in shared_keys:
    x1, x2 = style[layer], styled_content[layer]
    loss += F.mse_loss(gram_matrix(x1),
                       gram_matrix(x2)) * config.style_weights[layer]

  layer_weights = sum([config.style_weights[l] for l in shared_keys])
  loss_scale = config.style_loss_weight / layer_weights
  return loss * loss_scale


def content_loss(content, styled_content):
  config = get_config()
  loss = 0
  shared_keys = set(content).intersection(set(styled_content))
  assert shared_keys, 'No common layers'

  for layer in shared_keys:
    x1, x2 = content[layer], styled_content[layer]
    loss += F.mse_loss(x1, x2) * config.content_weights[layer]

  layer_weights = sum([config.content_weights[l] for l in shared_keys])
  loss_scale = config.content_loss_weight / layer_weights
  return loss * loss_scale


# def gram_matrix(x):
#   a, b, c, d = x.size()  # a=batch size(=1)
#   # b=number of feature maps
#   # (c,d)=dimensions of a f. map (N=c*d)

#   features = x.view(a * b, c * d)  # resise F_XL into \hat F_XL

#   G = torch.mm(features, features.t())  # compute the gram product

#   # we 'normalize' the values of the gram matrix
#   # by dividing by the number of element in each feature maps.
#   return G.div(a * b * c * d)


# Gram Matrix
def gram_matrix(tensor):
  B, C, H, W = tensor.shape
  x = tensor.view(B, C, H * W)
  x_t = x.transpose(1, 2)
  return torch.bmm(x, x_t) / (C * H * W)
