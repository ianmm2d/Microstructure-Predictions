from torch import nn
from torch.nn import functional as F
import torch

class CNN(nn.Module):
    def __init__(self, input_channels, input_height, input_width):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Compute output size after convolutions and pooling
        self._to_linear = None
        self._get_conv_output_shape(input_height, input_width)

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 1)

    def _get_conv_output_shape(self, height, width):
        with torch.no_grad():
            x = torch.zeros(1, 1, height, width)
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            self._to_linear = x.view(1, -1).shape[1]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x