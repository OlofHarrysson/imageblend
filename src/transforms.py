import torch
import torch.nn as nn
from torchvision import transforms
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from anyfig import get_config


def get_train_transforms():
  transformer = Transformer()

  return transforms.Compose([
    transformer,
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
  ])


def get_val_transforms():
  return get_train_transforms()


class Transformer():
  def __init__(self):
    im_size = get_config().input_size
    self.seq = iaa.Resize({"height": im_size, "width": im_size})

  def __call__(self, im):
    return self.seq.augment_image(np.array(im))


def un_norm_img(img):
  img = img.clone().cpu().detach()
  img = un_normalize(img).clamp(0, 1).numpy()
  return img


def un_normalize(tensor):
  mean = (0.485, 0.456, 0.406)
  std = (0.229, 0.224, 0.225)

  for t, m, s in zip(tensor, mean, std):
    t.mul_(s).add_(m)
    # The normalize code -> t.sub_(m).div_(s)
  return tensor
