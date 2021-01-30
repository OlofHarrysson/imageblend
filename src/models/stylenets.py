import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceNet(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = _conv_layer(3, 32, 9, 1)
    self.conv2 = _conv_layer(32, 64, 3, 2)
    self.conv3 = _conv_layer(64, 128, 3, 2)
    self.resid1 = ResidualBlock(128)
    self.resid2 = ResidualBlock(128)
    self.resid3 = ResidualBlock(128)
    self.resid4 = ResidualBlock(128)
    self.resid5 = ResidualBlock(128)
    self.conv_t1 = _conv_tranpose_layer(128, 64, 3, 2)
    self.conv_t2 = _conv_tranpose_layer(64, 32, 3, 2)
    self.conv_t3 = _conv_layer(32, 3, 9, 1, relu=False)
    self.tanh = nn.Tanh()

  def forward(self, x):
    for mod in self.children():
      x = mod(x)

    # x = self.tanh(x) * 150 + 255. / 2  # TODO: Wtf is this... from github
    x = (self.tanh(x) + 1) * 255 / 2  # [0, 255]

    return x


def _conv_layer(in_channels, out_channels, kernel_size=3, stride=1, relu=True):
  conv = nn.Conv2d(in_channels,
                   out_channels,
                   kernel_size=kernel_size,
                   padding=kernel_size // 2,
                   stride=stride)
  norm = nn.InstanceNorm2d(out_channels, affine=True)
  relu = nn.ReLU()
  return nn.Sequential(*[conv, norm, relu])


def _conv_tranpose_layer(in_channels, out_channels, kernel_size=3, stride=1):
  conv = nn.ConvTranspose2d(in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride)
  norm = nn.InstanceNorm2d(out_channels, affine=True)
  relu = nn.ReLU()
  return nn.Sequential(*[conv, norm, relu])


class ResidualBlock(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    out_channels = in_channels
    self.block1 = _conv_layer(in_channels, out_channels)
    self.block2 = _conv_layer(in_channels, out_channels, relu=False)
    self.relu = nn.ReLU()

  def forward(self, x):
    residual = x
    x = self.block1(x)
    x = self.block2(x)
    x += residual
    x = self.relu(x)
    return x


class UNet(nn.Module):
  def __init__(self, n_channels, n_classes, bilinear=False):
    super().__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes  # Number of channels RGB
    self.bilinear = bilinear

    self.inc = DoubleConv(n_channels, 64)
    self.down1 = Down(64, 128)
    self.down2 = Down(128, 256)
    self.down3 = Down(256, 512)
    factor = 2 if bilinear else 1
    self.down4 = Down(512, 1024 // factor)
    self.up1 = Up(1024, 512 // factor, bilinear)
    self.up2 = Up(512, 256 // factor, bilinear)
    self.up3 = Up(256, 128 // factor, bilinear)
    self.up4 = Up(128, 64, bilinear)
    self.outc = OutConv(64, n_classes)
    self.tanh = nn.Tanh()

  def forward(self, inputs):
    x1 = self.inc(inputs)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x5 = self.down4(x4)
    x = self.up1(x5, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)
    x = self.outc(x)
    x = (self.tanh(x) + 1) * 255 / 2  # [0, 255]

    return x


class DoubleConv(nn.Module):
  """(convolution => [BN] => ReLU) * 2"""
  def __init__(self, in_channels, out_channels, mid_channels=None):
    super().__init__()
    if not mid_channels:
      mid_channels = out_channels
    self.double_conv = nn.Sequential(
      nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
      nn.InstanceNorm2d(mid_channels, affine=True),
      nn.ReLU(inplace=True),
      nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
      nn.InstanceNorm2d(out_channels, affine=True),
      nn.ReLU(inplace=True),
    )

  def forward(self, x):
    return self.double_conv(x)


class Down(nn.Module):
  """Downscaling with maxpool then double conv"""
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2),
                                      DoubleConv(in_channels, out_channels))

  def forward(self, x):
    return self.maxpool_conv(x)


class Up(nn.Module):
  """Upscaling then double conv"""
  def __init__(self, in_channels, out_channels, bilinear=True):
    super().__init__()

    # if bilinear, use the normal convolutions to reduce the number of channels
    if bilinear:
      self.up = nn.Upsample(scale_factor=2,
                            mode='bilinear',
                            align_corners=True)
      self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
    else:
      self.up = nn.ConvTranspose2d(in_channels,
                                   in_channels // 2,
                                   kernel_size=2,
                                   stride=2)
      self.conv = DoubleConv(in_channels, out_channels)

  def forward(self, x1, x2):
    x1 = self.up(x1)
    # input is CHW
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(
      x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    # if you have padding issues, see
    # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)


class OutConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

  def forward(self, x):
    return self.conv(x)


class TransformerResNextNetwork_Pruned(nn.Module):
  """
    Feedforward Transformation Network - ResNeXt
    
        - No Tanh
        + ResNeXt Layer
        + Pruned
        Reference: https://heartbeat.fritz.ai/creating-a-17kb-style-transfer-model-with-layer-pruning-and-quantization-864d7cc53693 
    """
  def __init__(self, alpha=1.0):
    super().__init__()
    a = alpha
    self.ConvBlock = nn.Sequential(
      ConvLayer(3, int(a * 32), 9, 1),
      nn.ReLU(),
      ConvLayer(int(a * 32), int(a * 32), 3, 2),
      nn.ReLU(),
      ConvLayer(int(a * 32), int(a * 32), 3, 2),
      nn.ReLU(),
    )
    self.ResidualBlock = nn.Sequential(
      ResNextLayer(
        int(a * 32),
        [int(a * 16), int(a * 16), int(a * 32)], kernel_size=3),
      ResNextLayer(
        int(a * 32),
        [int(a * 16), int(a * 16), int(a * 32)], kernel_size=3),
      ResNextLayer(
        int(a * 32),
        [int(a * 16), int(a * 16), int(a * 32)], kernel_size=3),
    )
    self.DeconvBlock = nn.Sequential(
      DeconvLayer(int(a * 32), int(a * 32), 3, 2, 1), nn.ReLU(),
      DeconvLayer(int(a * 32), int(a * 32), 3, 2, 1), nn.ReLU(),
      ConvLayer(int(a * 32), 3, 9, 1, norm="None"))

    self.tanh = nn.Tanh()

  def forward(self, x):
    x = self.ConvBlock(x)
    x = self.ResidualBlock(x)
    x = self.DeconvBlock(x)
    x = (self.tanh(x) + 1) * 255 / 2  # [0, 255]
    return x


class ResNextLayer(nn.Module):
  """
    Aggregated Residual Transformations for Deep Neural Networks

        Equal to better performance with 10x less parameters

    https://arxiv.org/abs/1611.05431
    """
  def __init__(self, in_ch=128, channels=[64, 64, 128], kernel_size=3):
    super().__init__()
    ch1, ch2, ch3 = channels
    self.conv1 = ConvLayer(in_ch, ch1, kernel_size=1, stride=1)
    self.relu1 = nn.ReLU()
    self.conv2 = ConvLayer(ch1, ch2, kernel_size=kernel_size, stride=1)
    self.relu2 = nn.ReLU()
    self.conv3 = ConvLayer(ch2, ch3, kernel_size=1, stride=1)

  def forward(self, x):
    identity = x
    out = self.relu1(self.conv1(x))
    out = self.relu2(self.conv2(out))
    out = self.conv3(out)
    out = out + identity
    return out


class DeconvLayer(nn.Module):
  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride,
               output_padding,
               norm="instance"):
    super().__init__()

    # Transposed Convolution
    padding_size = kernel_size // 2
    self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels,
                                             kernel_size, stride, padding_size,
                                             output_padding)

    # Normalization Layers
    self.norm_type = norm
    if (norm == "instance"):
      self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
    elif (norm == "batch"):
      self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

  def forward(self, x):
    x = self.conv_transpose(x)
    if (self.norm_type == "None"):
      out = x
    else:
      out = self.norm_layer(x)
    return out


class ConvLayer(nn.Module):
  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride,
               norm="instance"):
    super().__init__()
    # Padding Layers
    padding_size = kernel_size // 2
    self.reflection_pad = nn.ReflectionPad2d(padding_size)

    # Convolution Layer
    self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    # Normalization Layers
    self.norm_type = norm
    if (norm == "instance"):
      self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
    elif (norm == "batch"):
      self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

  def forward(self, x):
    x = self.reflection_pad(x)
    x = self.conv_layer(x)
    if (self.norm_type == "None"):
      out = x
    else:
      out = self.norm_layer(x)
    return out
