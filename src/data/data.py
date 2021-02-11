from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
from anyfig import get_config
from PIL import Image
import numpy as np

from ..transforms import get_train_augmenter
from ..utils.meta_utils import get_project_root


def setup_dataloaders():
  dataloaders = namedtuple('Dataloaders', ['train'])
  return dataloaders(train=setup_trainloader())


def setup_trainloader():
  transforms = get_train_augmenter()
  dataset_dir = get_project_root() / 'datasets' / 'mask'
  dataset = ImageTransfer(dataset_dir, transforms)
  return DataLoader(dataset,
                    batch_size=get_config().batch_size,
                    num_workers=get_config().num_workers,
                    shuffle=True)


class ImageTransfer(Dataset):
  def __init__(self, path, augmenter):
    super().__init__()
    self.augmenter = augmenter
    self.data_root = path

  def __len__(self):
    return 1

  def __getitem__(self, index):
    style_img = Image.open(self.data_root / 'style.jpg')
    src_img = np.array(style_img)
    raw_content_img = Image.open(self.data_root / 'content.jpg')
    mask = Image.open(self.data_root / 'mask.jpg')
    bbox = mask.getbbox()
    content_image = style_img.copy()
    content_image.paste(raw_content_img, mask=mask)

    style_img = style_img.crop(bbox)
    content_image = content_image.crop(bbox)
    # mask = mask.crop(bbox)

    mask_img = self.augmenter(mask, end=-2)
    return self.augmenter(style_img), self.augmenter(
      content_image), mask_img, src_img
