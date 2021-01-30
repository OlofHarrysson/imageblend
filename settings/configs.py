import anyfig
import pyjokes
import random
from datetime import datetime
from collections import defaultdict


@anyfig.config_class
class MiscConfig():
  # ~~~~~~~~~~~~~~ General Parameters ~~~~~~~~~~~~~~
  def __init__(self):
    super().__init__()

    # Creates directory. Saves config & git info
    self.save_experiment: bool = False

    # An optional comment to differentiate this run from others
    self.save_comment: str = pyjokes.get_joke()

    # Start time to keep track of when the experiment was run
    self.start_time: str = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")

    # Seed for reproducability
    self.seed: int = random.randint(0, 2**31)

    # Decides if logger should be active
    self.log_data: bool = False


@anyfig.config_class
class TrainingConfig():
  # ~~~~~~~~~~~~~~ Training Parameters ~~~~~~~~~~~~~~
  def __init__(self):

    # The GPU device to use. Set the value to -1 to not use the GPU
    self.gpu: int = 0

    # Runs the computations with mixed precision. Only works with GPUs enabled
    self.mixed_precision = True

    # Number of threads to use in data loading
    self.num_workers: int = 0

    # Number of update steps to train
    self.optim_steps: int = 3000

    # Number of optimization steps between validation
    self.validation_freq: int = 500

    # Start and end learning rate for the scheduler
    self.start_lr: float = 1e-3
    self.end_lr: float = 1e-4

    # Batch size going into the network
    self.batch_size: int = 1

    # Size for image that is fed into the network
    self.input_size = 128

    # Use a pretrained network
    self.pretrained: bool = False

    # Misc configs
    self.misc = MiscConfig()

    # Weight for losses
    self.style_loss_weight = 4e4
    self.content_loss_weight = 1

    # Loss weights for layers
    self.style_weights = defaultdict(lambda: 1)
    # self.style_weights[-1] = 10

    self.content_weights = defaultdict(lambda: 1)
    # self.content_weights[-1] = 10

    # Conv layer outputs
    # Conv layers, 0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34
    # -1 equals the raw-styled image
    # 37 is avg_pooling
    self.style_layers = [
      -1, 0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34
    ]
    self.content_layers = [37]
    self.styled_content_layers = set(self.style_layers + self.content_layers)


@anyfig.config_class
class TrainLaptop(TrainingConfig):
  def __init__(self):
    super().__init__()
    ''' Change default parameters here. Like this
    self.seed = 666            ____
      ________________________/ O  \___/  <--- Python <3
     <_#_#_#_#_#_#_#_#_#_#_#_#_____/   \
    '''
    self.gpu: int = -1
    self.misc.log_data = True
    # self.misc.save_experiment: bool = True


@anyfig.config_class
class Colab(TrainingConfig):
  def __init__(self):
    super().__init__()
    self.misc.log_data = True
    self.input_size = 256
