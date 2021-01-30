import torch
import torch.nn as nn
import torch.nn.functional as F
from anyfig import get_config


def calc_loss(outputs):
  config = get_config()
  losses = dict()
  losses['style'] = style_loss(
    outputs['style'],
    outputs['styled_content'],
  ) * config.style_loss_weight

  losses['content'] = content_loss(
    outputs['content'],
    outputs['styled_content'],
  ) * config.content_loss_weight

  return losses


def style_loss(style, styled_content):
  loss = 0
  for x1, x2 in zip(style, styled_content):
    loss += F.mse_loss(gram_matrix(x1), gram_matrix(x2))
  return loss


def content_loss(content, styled_content):
  loss = 0
  styled_content = [styled_content[-2]]  # TODO: Might select the wrong one
  for x1, x2 in zip(content, styled_content):
    loss += F.mse_loss(x1, x2)
  return loss


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
