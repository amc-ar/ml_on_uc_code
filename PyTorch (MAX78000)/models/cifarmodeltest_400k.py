#
# CIFAR-10 student (~400k params) for MAX78000 + knowledge distillation.
# All channel counts are multiples of 32. BN-fused layers, residuals, 1x1 classifier.
#
from torch import nn

import ai8x


class CifarModelTest400k(nn.Module):
    """
    CIFAR student targeting <=400k parameters.
    Channel widths: 32, 64, 96 (all multiples of 32).
    Spatial: 32x32 -> 16x16 -> 8x8 -> 1x1 classifier.
    """

    def __init__(
        self,
        num_classes=10,
        num_channels=3,
        dimensions=(32, 32),  # pylint: disable=unused-argument
        bias=True,
        **kwargs
    ):
        super().__init__()

        self.conv1 = ai8x.FusedConv2dBNReLU(num_channels, 32, 3, stride=1, padding=1,
                                            bias=bias, **kwargs)
        self.conv2 = ai8x.FusedConv2dBNReLU(32, 64, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv3 = ai8x.FusedConv2dBNReLU(64, 64, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.resid1 = ai8x.Add()

        self.conv4 = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, 3, pool_size=2, pool_stride=2,
                                                   stride=1, padding=1, bias=bias, **kwargs)
        self.conv5 = ai8x.FusedConv2dBNReLU(64, 64, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.resid2 = ai8x.Add()

        self.conv7 = ai8x.FusedMaxPoolConv2dBNReLU(64, 96, 3, pool_size=2, pool_stride=2,
                                                   stride=1, padding=1, bias=bias, **kwargs)
        self.conv8 = ai8x.FusedConv2dBNReLU(96, 96, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv9 = ai8x.FusedConv2dBNReLU(96, 96, 1, stride=1, padding=0, bias=bias, **kwargs)
        self.resid3 = ai8x.Add()

        self.conv10 = ai8x.FusedConv2dBNReLU(96, 32, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv11 = ai8x.FusedMaxPoolConv2dBNReLU(32, 32, 1, pool_size=8, pool_stride=8,
                                                    padding=0, bias=bias, **kwargs)
        self.classifier = ai8x.Conv2d(32, num_classes, 1, stride=1, padding=0, bias=bias,
                                      wide=True, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x_res = self.conv2(x)
        x = self.conv3(x_res)
        x = self.resid1(x, x_res)

        x_res = self.conv4(x)
        x = self.conv5(x_res)
        x = self.resid2(x, x_res)

        x = self.conv7(x)
        x_res = self.conv8(x)
        x = self.conv9(x_res)
        x = self.resid3(x, x_res)

        x = self.conv10(x)
        x = self.conv11(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x


def cifarmodeltest_400k(pretrained=False, **kwargs):
    """Constructs ~400k-param CIFAR student for MAX78000."""
    assert not pretrained
    return CifarModelTest400k(**kwargs)


models = [
    {
        'name': 'cifarmodeltest_400k',
        'min_input': 1,
        'dim': 2,
    },
]
