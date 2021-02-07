import visdom
import functools
from anyfig import get_config
import numpy as np

from .utils import plotly_plots as plts


def clear_old_data(vis):
  [vis.close(env=env) for env in vis.get_env_list()]  # Kills wind
  # [vis.delete_env(env) for env in vis.get_env_list()] # Kills envs


def log_if_active(func):
  ''' Decorator which only calls logging function if logger is active '''
  @functools.wraps(func)
  def wrapper(self, *args, **kwargs):
    if get_config().misc.log_data:
      return func(self, *args, **kwargs)

  return wrapper


class Logger():
  def __init__(self):
    config = get_config()
    if config.misc.log_data:
      # outpath = f'output/visdom_{config.misc.start_time}.txt'
      try:
        # self.vis = visdom.Visdom(log_to_filename=outpath)
        self.vis = visdom.Visdom()
        clear_old_data(self.vis)
      except Exception as e:
        err_msg = (
          "Couldn't connect to Visdom. Make sure to have a Visdom server running or turn of "
          "logging in the config")
        raise ConnectionError(err_msg) from e

  @log_if_active
  def log_text(self, text):
    self.vis.text(text)

  @log_if_active
  def log_image(self, image, caption):
    opts = dict(store_history=True, caption=caption)
    self.vis.image(image, opts)

  @log_if_active
  def log_accuracy(self, accuracy, step, name='train'):
    title = f'{name} Accuracy'.title()
    plot = plts.accuracy_plot(self.vis.line, title)
    plot(X=[step], Y=[accuracy])

  @log_if_active
  def log_gradients(self, model, step):
    max_gradients = []
    avg_gradients = []
    legends = []
    for name, params in model.named_parameters():
      max_ = params.abs().max()
      avg = params.abs().mean()
      max_gradients.append(max_.item())
      avg_gradients.append(avg.item())
      legends.append(name)

    title = 'Max Abs Gradient'.title()
    opts = dict(xlabel='Steps', title=title)
    self.vis.line(Y=np.array(max_gradients).reshape(1, -1),
                  update='append',
                  win=title,
                  X=[step],
                  opts=opts)

    title = 'Mean Abs Gradient'.title()
    opts = dict(xlabel='Steps', title=title)
    self.vis.line(Y=np.array(avg_gradients).reshape(1, -1),
                  update='append',
                  win=title,
                  X=[step],
                  opts=opts)

  @log_if_active
  def log_losses(self, loss_dict, step):
    legend, losses = [], []
    for name, loss in loss_dict.items():
      legend.append(name)
      losses.append(loss.item())

    tot_loss = sum(losses)
    accumulated_loss = 0
    Y = []
    for loss in losses:
      Y.append((accumulated_loss + loss) / tot_loss)
      accumulated_loss += loss

    self.vis.line(Y=np.array(Y).reshape(1, -1),
                  X=[step],
                  update='append',
                  win='losspercent',
                  opts=dict(fillarea=True,
                            xlabel='Steps',
                            ylabel='Percentage',
                            title='Loss Percentage',
                            stackgroup='one',
                            legend=legend))

    all_losses = losses + [tot_loss]
    legends = legend + ['Total']
    self.vis.line(Y=np.array(all_losses).reshape(1, -1),
                  X=[step],
                  update='append',
                  win='TotalLoss',
                  opts=dict(xlabel='Steps',
                            ylabel='Loss',
                            title='Training Loss',
                            legend=legends))
