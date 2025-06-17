from torch import nn
from torch.nn import functional as F

class CNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for image classification.
    This model consists of two convolutional layers followed by two fully connected layers.
    Args:
        input_channels (int): Number of input channels (e.g., 3 for RGB images).
        num_classes (int): Number of output classes for classification.
    """
    def __init__(self, input_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Assuming input size is 32x32
        self.fc2 = nn.Linear(128,1)  # Regression output

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x