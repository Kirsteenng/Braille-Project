import torch
import torch.nn as nn
from torch.nn import functional as F

class BrailleModel(nn.Module):
    def __init__(self, fc_units):
        super(BrailleModel, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=3,   out_channels=64,  kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64,  out_channels=64,  kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=64,  out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(10240, fc_units[0])
        self.fc2 = nn.Linear(fc_units[0], fc_units[1])
        self.fc3 = nn.Linear(fc_units[1], fc_units[2])
        # self.fc1 = nn.Linear(512, 128)
        # self.fc2 = nn.Linear(128, fc_units[2])

    def forward(self, x):
        x = torch.relu(self.conv1_1(x))
        x = torch.relu(self.conv1_2(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv2_1(x))
        x = torch.relu(self.conv2_2(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv3_1(x))
        x = torch.relu(self.conv3_2(x))
        x = torch.relu(self.conv3_3(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv4_1(x))
        x = torch.relu(self.conv4_2(x))
        x = torch.relu(self.conv4_3(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv5_1(x))
        x = torch.relu(self.conv5_2(x))
        x = torch.relu(self.conv5_3(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = F.dropout(x, 0.5) #dropout was included to combat overfitting
        x = torch.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        output = self.fc3(x)
        output = torch.sigmoid(output)

        # x = torch.relu(self.fc1(x))
        # x = F.dropout(x, 0.5)
        # output = self.fc2(x)
        # output = torch.sigmoid(output)
        return output