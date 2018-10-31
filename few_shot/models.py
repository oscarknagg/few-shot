from torch import nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class GlobalMaxPool1d(nn.Module):
    def forward(self, input):
        return nn.functional.max_pool1d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


class GlobalAvgPool2d(nn.Module):
    def forward(self, input):
        return nn.functional.avg_pool2d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


class Bottleneck(nn.Module):
    """Gets bottleneck features from an nn.Sequential classifier."""
    def __init__(self, model):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(*model[:-1])

    def forward(self, x):
        return self.bottleneck(x)


def get_few_shot_encoder(num_input_channels=1):
    return nn.Sequential(
        nn.Conv2d(num_input_channels, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        Flatten(),
    )
