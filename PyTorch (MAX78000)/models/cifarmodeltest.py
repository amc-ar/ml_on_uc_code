from torch import nn

import ai8x

class CifarModelTest(nn.Module):
    def __init__(
        self,
        num_classes=10,
        num_channels=3,
        dimensions=(32,32),
        bias=True,
        **kwargs
    ):
        super().__init__()
        
        self.conv1 = ai8x.FusedConv2dReLU(in_channels=3, out_channels=32, kernel_size=3, padding=1,
        bias=bias)

        self.conv2 = ai8x.FusedConv2dReLU(in_channels=32, out_channels=64, kernel_size=3, padding=1,
        bias=bias)

        self.conv3 = ai8x.FusedConv2dReLU(in_channels=64, out_channels=64, kernel_size=3, padding=1,
        bias=bias)

        self.conv4 = ai8x.FusedConv2dReLU(in_channels=64, out_channels=48, kernel_size=3, padding=1,
        bias=bias)

        self.conv5 = ai8x.FusedConv2dReLU(in_channels=48, out_channels=48, kernel_size=3, padding=1,
        bias=bias)

        self.conv6 = ai8x.FusedConv2dReLU(in_channels=48, out_channels=32, kernel_size=3, padding=1,
        bias=bias)

        self.conv7 = ai8x.FusedMaxPoolConv2dReLU(in_channels=32, out_channels=16, kernel_size=3, padding=1,
        bias=bias)
        

        self.conv8 = ai8x.FusedMaxPoolConv2dReLU(in_channels=16, out_channels=8, kernel_size=3, padding=1,
        bias=bias)

        self.fc = ai8x.Linear(8*8*8, num_classes, bias=bias, wide=True, **kwargs)




    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x 

    
def cifarmodeltest(pretrained=False, **kwargs):
    """
    Constructs a Cifar Model.
    """
    assert not pretrained
    return CifarModelTest(**kwargs)


models = [
{
    'name': 'cifarmodeltest',
    'min_input': 1,
    'dim': 2,
},
]