import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from torchvision.models.utils import load_state_dict_from_url
from typing import Union, List, Dict, Any, cast
from collections import defaultdict
from torchvision import transforms

from . import stylenets
from . import lossnets


def get_model(config):
  model = MyModel(config)
  model.to(model.device)
  return model


class MyModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.device = 'cpu' if config.gpu < 0 else torch.device('cuda', config.gpu)

    # self.stylenet = stylenets.TransformerResNextNetwork_Pruned(alpha=0.3)
    self.stylenet = stylenets.UNet(3, 3)
    # self.stylenet = stylenets.InstanceNet()
    self.loss_net = lossnets.VGG19()

    self.normalize = transforms.Normalize((0.485, 0.456, 0.406),
                                          (0.229, 0.224, 0.225))

  def forward(self, inputs):
    # inputs = inputs.to(self.device)
    styled_content = inputs['styled_content']
    # styled_content = self.stylenet(styled_content) + styled_content
    styled_content = self.stylenet(styled_content)
    styled_img = styled_content.clone().detach()
    styled_content = styled_content / 255  # Range [0, 1]
    styled_content = self.normalize(styled_content)
    inputs['styled_content'] = styled_content

    return self.loss_net(inputs), styled_img[0]
    # return self.loss_net(inputs), styled_img[0]

  def predict(self, inputs):
    with torch.no_grad():
      return self.loss_net(inputs)

  def save(self, path):
    path = Path(path)
    err_msg = f"Expected path that ends with '.pt' or '.pth' but was '{path}'"
    assert path.suffix in ['.pt', '.pth'], err_msg
    path.parent.mkdir(exist_ok=True)
    print("Saving Weights @ " + str(path))
    torch.save(self.state_dict(), path)

  def load(self, path):
    print('Loading weights from {}'.format(path))
    weights = torch.load(path, map_location='cpu')
    self.load_state_dict(weights, strict=False)
    self.to(self.device)
