import torch
import torch.nn as nn
from torchvision import models, transforms
from collections import defaultdict
from anyfig import get_config


class VGG19(nn.Module):
  def __init__(self):
    super().__init__()
    vgg19_features = models.vgg19(pretrained=True)
    self.features = vgg19_features.features

    # Turn-off Gradient History
    for param in self.features.parameters():
      param.requires_grad = False

    config = get_config()
    self.output_layers = dict(
      style=config.style_layers,
      content=config.content_layers,
      styled_content=config.styled_content_layers,
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
