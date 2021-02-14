from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
from anyfig import get_config
from PIL import Image

from ..transforms import get_train_transforms, get_val_transforms
from ..utils.meta_utils import get_project_root


def setup_dataloaders():
  dataloaders = namedtuple('Dataloaders', ['train', 'val'])
  return dataloaders(train=setup_trainloader(), val=setup_valloader())


def setup_trainloader():
  transforms = get_train_transforms()
  dataset_dir = get_project_root() / 'datasets' / 'easy'
  dataset = ImageTransfer(dataset_dir, transforms)
  return DataLoader(dataset,
                    batch_size=get_config().batch_size,
                    num_workers=get_config().num_workers,
                    shuffle=True)


def setup_valloader():
  transforms = get_val_transforms()
  dataset_dir = get_project_root() / 'datasets' / 'easy'
  dataset = ImageTransfer(dataset_dir, transforms)
  return DataLoader(dataset,
                    batch_size=get_config().batch_size,
                    num_workers=get_config().num_workers)


class ImageTransfer(Dataset):
  def __init__(self, path, transforms):
    super().__init__()
    self.transforms = transforms
    self.data_root = path

  def __len__(self):
    return 1

  def __getitem__(self, index):
    style_img = Image.open(self.data_root / 'style.jpg')
    content_img = Image.open(self.data_root / 'content.jpg')
    return self.transforms(style_img, start=1), self.transforms(content_img)
