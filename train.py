import math

import anyfig
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from pathlib import Path
from progressbar import progressbar
import numpy as np

from src.data.data import setup_dataloaders
from src.models.model import get_model
from src.logger import Logger
from src.evaluation.validator import Validator
from src.evaluation.metrics import setup_metrics
from settings import configs
from src.utils.meta_utils import ProgressbarWrapper as Progressbar
from src.utils.meta_utils import speed_up_cuda
import src.utils.setup_utils as setup_utils
import losses

from src.transforms import un_norm_img


def train(config):
  speed_up_cuda()
  dataloaders = setup_dataloaders()

  logger = Logger()
  model = get_model(config)
  optimizer = torch.optim.Adam(model.stylenet.parameters(), lr=config.start_lr)
  lr_scheduler = CosineAnnealingLR(optimizer,
                                   T_max=config.optim_steps,
                                   eta_min=config.end_lr)

  mixed_precision = config.mixed_precision and model.device != 'cpu'
  scaler = GradScaler(enabled=mixed_precision)

  # Init progressbar
  # n_batches = len(dataloaders.train)
  # n_epochs = math.ceil(config.optim_steps / n_batches)
  # progressbar = Progressbar(n_epochs, n_batches)

  # Data
  style_img, content_img = next(iter(dataloaders.train))
  styled_image = content_img.clone()

  logger.log_image(un_norm_img(style_img[0]), 'Style Image')
  logger.log_image(un_norm_img(content_img[0]), 'Content Image')

  # Training loop
  style_img = style_img.to(model.device)
  content_img = content_img.to(model.device)
  styled_image = styled_image.to(model.device)
  style_fmaps = model.predict(dict(style=style_img))
  content_fmaps = model.predict(dict(content=content_img))

  for optim_steps in progressbar(range(config.optim_steps),
                                 redirect_stdout=True):
    # Forward pass
    with autocast(False):
      optimizer.zero_grad()
      inputs = dict(styled_content=styled_image)
      styled_content, styled_img = model(inputs)
      # styled_img = un_norm_img(styled_img, unnorm=False)
      styled_img = styled_img.clone().cpu().detach().numpy()
      fmaps = {**style_fmaps, **content_fmaps, **styled_content}
      loss_dict = losses.calc_loss(fmaps)
      loss = sum(loss_dict.values())

      # Backward pass
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      # Decrease learning rate
      lr_scheduler.step()

      # Log
      logger.log_image(styled_img, 'Styled Image')
      if optim_steps > 50:
        logger.log_losses(loss_dict, optim_steps)

      # pil_img = (styled_img * 255).astype(np.uint8)
      # pil_img = styled_img.astype(np.uint8)
      # pil_img = np.moveaxis(pil_img, 0, -1)
      # outdir = Path('output') / 'stylenet'
      # outdir.mkdir(parents=True, exist_ok=True)
      # Image.fromarray(pil_img).save(outdir / f'{optim_steps}.png')


if __name__ == '__main__':
  config = anyfig.init_config(default_config=configs.TrainLaptop)
  print(config)  # Remove if you dont want to see config at start
  print('\n{}\n'.format(config.misc.save_comment))
  setup_utils.setup(config.misc)
  train(config)
