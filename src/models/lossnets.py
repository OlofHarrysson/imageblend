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

    # Conv layer outputs
    # Conv layers, 0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34
    # -1 equals the raw-styled image
    self.output_layers = dict(
      style=[7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34],
      content=[-1, 25],
      styled_content=[
        -1, 0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34
      ],
    )

  def forward(self, inputs):
    outputs = defaultdict(dict)
    for input_type, x in inputs.items():
      output_layers = list(self.output_layers[input_type])

      # The input image
      if -1 in output_layers:
        outputs[input_type][-1] = x
        output_layers.remove(-1)

      for mod_idx, mod in enumerate(self.features):
        if output_layers:
          x = mod(x)

          if mod_idx in output_layers:
            outputs[input_type][mod_idx] = x
            output_layers.remove(mod_idx)

    return outputs
