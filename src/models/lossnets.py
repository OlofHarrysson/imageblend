import torch
import torch.nn as nn
from torchvision import models, transforms
from collections import defaultdict


class VGG19(nn.Module):
  def __init__(self):
    super().__init__()
    vgg19_features = models.vgg19(pretrained=True)
    self.features = vgg19_features.features

    # Turn-off Gradient History
    for param in self.features.parameters():
      param.requires_grad = False

    # self.output_layers = dict(
    #   style=[3, 8, 17, 26, 35],
    #   # content=[26],  # 8 or 26
    #   content=[3, 8, 17, 26, 35],  # 8 or 26
    #   styled_content=[3, 8, 17, 26, 35],
    # )

    # Before Relu
    self.output_layers = dict(
      style=[2, 7, 16, 25, 34],
      content=[7],
      styled_content=[2, 7, 16, 25, 34],
    )

  def forward(self, inputs):
    outputs = defaultdict(list)
    for input_type, x in inputs.items():
      output_layers = list(self.output_layers[input_type])
      for mod_idx, mod in enumerate(self.features):
        if output_layers:
          x = mod(x)

          if mod_idx in output_layers:
            outputs[input_type].append(x)
            output_layers.remove(mod_idx)

    return outputs


# class VGG16(VGG19):
#   def __init__(self):
#     super().__init__()
#     # Load VGG Skeleton, Pretrained Weights
#     vgg16_features = models.vgg16(pretrained=True)
#     self.features = vgg16_features.features

#     # Turn-off Gradient History
#     for param in self.features.parameters():
#       param.requires_grad = False

#     self.output_layers = dict(
#         style=[3, 8, 17, 26, 35],
#         content=[26],
#         styled_content=[3, 8, 17, 26, 35],
#       )

#   def forward(self, x):
#     layers = {'3': 'relu1_2', '8': 'relu2_2', '15': 'relu3_3', '22': 'relu4_3'}
#     features = {}
#     for name, layer in self.features._modules.items():
#       x = layer(x)
#       if name in layers:
#         features[layers[name]] = x
#         if (name == '22'):
#           break

#     return features
