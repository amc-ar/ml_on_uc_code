#
# ResNet-56 for CIFAR (teacher / FP32 baseline).
# Architecture from pytorch-cifar-models (chenyaofo).
#
# Modified from https://raw.githubusercontent.com/pytorch/vision/v0.9.1/torchvision/models/resnet.py
# BSD 3-Clause License (see upstream).
#
import logging
from pathlib import Path

import torch
import torch.nn as nn

_log = logging.getLogger(__name__)

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


CIFAR10_PRETRAINED_URLS = {
    'resnet56': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt',
}

CIFAR100_PRETRAINED_URLS = {
    'resnet56': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet56-f2eff4c8.pt',
}

# Statistics used to train chenyaofo/pytorch-cifar-models checkpoints.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# Optional local checkpoint (place file here to skip download).
_LOCAL_CIFAR10_CKPT = Path(__file__).resolve().parent.parent / 'pretrained' / 'cifar10_resnet56-187c023a.pt'
_LOCAL_CIFAR100_CKPT = Path(__file__).resolve().parent.parent / 'pretrained' / 'cifar100_resnet56-f2eff4c8.pt'
# chenyaofo CIFAR-10 ResNet-56 release is ~3.5 MiB; reject empty/partial copies.
_MIN_CKPT_BYTES = 500_000


def _announce(msg):
    """Print so load/download status is visible even when logging is quiet."""
    print(f'[resnet56_cifar] {msg}', flush=True)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):
    """ResNet for CIFAR-10/100."""

    def __init__(self, block, layers, num_classes=10, adapt_ai8x_input=False):
        super().__init__()
        self.adapt_ai8x_input = adapt_ai8x_input
        if num_classes == 10:
            mean, std = CIFAR10_MEAN, CIFAR10_STD
        else:
            mean, std = CIFAR100_MEAN, CIFAR100_STD
        self.register_buffer('cifar_mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('cifar_std', torch.tensor(std).view(1, 3, 1, 1))

        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _from_ai8x_input(self, x):
        """
        Convert ai8x-training CIFAR tensors to the normalization expected by
        chenyaofo/pytorch-cifar-models pretrained weights.
        """
        # FP32 ai8x path: y = round((pixel - 0.5) * 256).clamp(-128, 127) / 128
        if x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            pixels = x.float().div(256.0).add(0.5)
        elif x.abs().max() > 1.5:
            # QAT / 8-bit style inputs in [-128, 127] without final /128 scaling
            pixels = x.float().div(256.0).add(0.5)
        else:
            pixels = x.mul(0.5).add(0.5)
        pixels = pixels.clamp(0.0, 1.0)
        return (pixels - self.cifar_mean) / self.cifar_std

    def forward(self, x):
        if self.adapt_ai8x_input:
            x = self._from_ai8x_input(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _extract_state_dict(loaded):
    """Unwrap checkpoints saved by train.py or torch.hub."""
    if not isinstance(loaded, dict):
        return loaded
    for key in ('state_dict', 'model_state_dict', 'model'):
        if key in loaded and isinstance(loaded[key], dict):
            return loaded[key]
    return loaded


def _local_ckpt_usable(local_path):
    if not local_path.is_file():
        return False
    size = local_path.stat().st_size
    if size < _MIN_CKPT_BYTES:
        _announce(
            f'Ignoring corrupt checkpoint {local_path} ({size} bytes, '
            f'expected >= {_MIN_CKPT_BYTES})'
        )
        return False
    return True


def _load_pretrained_weights(model, num_classes, progress=True):
    """Load chenyaofo/pytorch-cifar-models ResNet-56 weights."""
    arch = 'resnet56'
    if num_classes == 10:
        local_path = _LOCAL_CIFAR10_CKPT
        url = CIFAR10_PRETRAINED_URLS[arch]
    elif num_classes == 100:
        local_path = _LOCAL_CIFAR100_CKPT
        url = CIFAR100_PRETRAINED_URLS[arch]
    else:
        raise ValueError(f'No pretrained ResNet-56 weights for num_classes={num_classes}')

    if _local_ckpt_usable(local_path):
        source = str(local_path)
        _announce(f'Loading pretrained weights from local file: {source}')
        try:
            loaded = torch.load(local_path, map_location='cpu', weights_only=True)
        except TypeError:
            loaded = torch.load(local_path, map_location='cpu')
    else:
        source = url
        _announce(f'Downloading pretrained weights from {url} ...')
        loaded = load_state_dict_from_url(url, progress=progress)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(loaded, local_path)
            _announce(
                f'Download complete ({local_path.stat().st_size} bytes). '
                f'Copy pretrained/ to offline machines.'
            )
        except OSError as exc:
            _announce(f'WARNING: could not cache weights to {local_path}: {exc}')

    _log.info('Loading ResNet-56 pretrained weights from %s', source)
    state_dict = _extract_state_dict(loaded)
    # Strip DistributedDataParallel / nested module prefixes if present.
    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    incompatible = model.load_state_dict(state_dict, strict=False)
    # Input-adapter buffers are not in the upstream checkpoint.
    ignore_missing = {'cifar_mean', 'cifar_std'}
    missing = [k for k in incompatible.missing_keys if k not in ignore_missing]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            'ResNet-56 pretrained checkpoint does not match model architecture.\n'
            f'  missing_keys: {missing}\n'
            f'  unexpected_keys: {incompatible.unexpected_keys}'
        )
    _announce(f'Pretrained weights applied ({len(state_dict)} tensors)')


def resnet56_cifar(
    pretrained=False,
    num_classes=10,
    num_channels=3,  # pylint: disable=unused-argument
    dimensions=(32, 32),  # pylint: disable=unused-argument
    adapt_ai8x_input=None,
    **kwargs  # pylint: disable=unused-argument
):
    """
    ResNet-56 for CIFAR-10/100 (FP32 teacher).

    Set pretrained=True to load chenyaofo/pytorch-cifar-models weights. When using
    those weights with ai8x-training's CIFAR loader, adapt_ai8x_input defaults to
    True (converts ai8x.normalize tensors to standard CIFAR mean/std).

    If the model was trained in this repo (no external pretrained), leave
    adapt_ai8x_input=False (default).
    """
    if adapt_ai8x_input is None:
        adapt_ai8x_input = pretrained
    model = CifarResNet(BasicBlock, [9, 9, 9], num_classes=num_classes,
                      adapt_ai8x_input=adapt_ai8x_input)
    if pretrained:
        _load_pretrained_weights(model, num_classes)
        _announce(
            f'Ready: adapt_ai8x_input={adapt_ai8x_input} '
            '(expect ~94% test Top1 on ai8x CIFAR when True)'
        )
        if not adapt_ai8x_input:
            _announce(
                'WARNING: adapt_ai8x_input=False usually gives ~10% Top1 on ai8x CIFAR data'
            )
    else:
        _announce(
            'pretrained=False: using random init (expect ~10% Top1). '
            'Use --pretrained or --kd-pretrained to load chenyaofo weights.'
        )
    return model


models = [
    {
        'name': 'resnet56_cifar',
        'min_input': 1,
        'dim': 2,
    },
]
