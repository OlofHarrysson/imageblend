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
import torchvision.transforms.functional as TF
import time

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

from src.transforms import un_norm_img, smooth_distance_mask


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

  # Data
  style_img, content_img, mask_img, src_img, style2_img = next(
    iter(dataloaders.train))
  soft_mask = smooth_distance_mask(mask_img)
  styled_img = content_img.clone()

  mask_pil_img = tensor2img(mask_img)
  src_pil_img = tensor2img(src_img)
  bbox = mask_pil_img.getbbox()

  logger.log_image(un_norm_img(style_img[0]), 'Style Image')
  logger.log_image(un_norm_img(content_img[0]), 'Content Image')
  logger.log_image(un_norm_img(styled_img[0]), 'Styled Image')
  logger.log_image(mask_img, 'Mask')
  logger.log_image(soft_mask, 'Soft Mask')
  logger.log_text(str(config).replace('\n', '<br>'))

  # Training loop
  style_img = style_img.to(model.device)
  style2_img = style2_img.to(model.device)
  content_img = content_img.to(model.device)
  styled_img = styled_img.to(model.device)
  soft_mask = soft_mask.to(model.device)
  # style_fmaps = model.predict(dict(style=style_img))
  style_fmaps = model.predict(dict(style=style2_img))
  content_fmaps = model.predict(dict(content=content_img))
  last_log_time = time.time()

  for optim_steps in progressbar(range(config.optim_steps),
                                 redirect_stdout=True):
    # Forward pass
    with autocast(mixed_precision):
      optimizer.zero_grad()
      inputs = dict(styled_content=styled_img)
      styled_content, out_img = model(inputs)
      fmaps = {**style_fmaps, **content_fmaps, **styled_content}

      # Start with only content loss for faster convergence
      warmup = optim_steps < config.warmup_steps
      loss_dict = losses.calc_loss(
        fmaps,
        soft_mask,
        out_img,
        warmup,
      )
      loss = sum(loss_dict.values())

      # Backward pass
      scaler.scale(loss).backward()
      torch.nn.utils.clip_grad_norm_(model.stylenet.parameters(),
                                     config.gradient_clip)
      scaler.step(optimizer)
      scaler.update()

      # Decrease learning rate
      lr_scheduler.step()

      # Log
      if time.time() - last_log_time > 5:
        last_log_time = time.time()
        logger.log_image(out_img[0], 'Styled Image')
        styled_pil_img = tensor2img(out_img)
        src_pil_img.paste(styled_pil_img,
                          box=bbox,
                          mask=mask_pil_img.crop(bbox))
        logger.log_image(img2tensor(src_pil_img), 'Composite Image')

      if not warmup:
        logger.log_losses(loss_dict, optim_steps)
      # logger.log_gradients(model.stylenet, optim_steps)

      # pil_img = (styled_img * 255).astype(np.uint8)
      # pil_img = styled_img.astype(np.uint8)
      # pil_img = np.moveaxis(pil_img, 0, -1)
      # outdir = Path('output') / 'stylenet'
      # outdir.mkdir(parents=True, exist_ok=True)
      # Image.fromarray(pil_img).save(outdir / f'{optim_steps}.png')


def tensor2img(img):
  img = img.squeeze().detach().cpu()
  img = img.numpy().astype(np.uint8)
  if img.ndim == 3 and img.shape[0] == 3:
    img = np.moveaxis(img, 0, -1)

  return Image.fromarray(img)


def img2tensor(img):
  img = np.array(img)
  if img.ndim == 3 and img.shape[-1] == 3:
    img = np.moveaxis(img, -1, 0)
  return torch.from_numpy(img)
  # return TF.pil_to_tensor(img) # Gives warning


if __name__ == '__main__':
  config = anyfig.init_config(default_config=configs.TrainLaptop)
  print(config)  # Remove if you dont want to see config at start
  print('\n{}\n'.format(config.misc.save_comment))
  setup_utils.setup(config.misc)
  train(config)
