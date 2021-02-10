import torch
import torch.nn as nn
from torchvision import transforms
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from anyfig import get_config
import cv2
from scipy import ndimage


def get_train_augmenter():
  return Augmenter()


def get_val_transforms():
  return get_train_augmenter()


class Augmenter():
  def __init__(self):
    im_size = get_config().input_size

    self.augmentations = [
      iaa.Resize({
        "height": im_size,
        "width": im_size
      }),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]

  def __call__(self, img, start=0, end=None):
    """ Applies the transformations to the image
      Keyword Arguments:
        start {int} -- start index of transforms
        end {int} -- end index of transforms. None to include all. -1 to skip last transform
    """
    augmentations = self.augmentations[start:end]
    assert augmentations
    x = np.array(img)
    for augmentation in augmentations:
      if isinstance(augmentation, iaa.Augmenter):
        x = augmentation.augment_image(x)
      else:
        x = augmentation(x)
    return x


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


def smooth_distance_mask(mask):
  mask = np.array(mask.squeeze())
  mask = ndimage.grey_erosion(mask, size=(3, 3))
  inverted_mask = 255 - mask
  img_dist = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, maskSize=3)
  # img_dist = img_dist**6
  img_dist[img_dist > 255] = 255
  soft_mask = torch.from_numpy(img_dist)
  return soft_mask
