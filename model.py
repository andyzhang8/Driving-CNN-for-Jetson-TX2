import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        # Input: (3, 224, 224)
        self.block1 = DepthwiseSeparableConv(3, 32, kernel_size=3, stride=1, padding=1)    # 224x224
        self.block2 = DepthwiseSeparableConv(32, 64, kernel_size=3, stride=2, padding=1)   # 224 -> 112
        self.block3 = DepthwiseSeparableConv(64, 128, kernel_size=3, stride=2, padding=1)  # 112 -> 56
        self.block4 = DepthwiseSeparableConv(128, 256, kernel_size=3, stride=2, padding=1) # 56 -> 28
        self.block5 = DepthwiseSeparableConv(256, 512, kernel_size=3, stride=2, padding=1) # 28 -> 14
        self.block6 = DepthwiseSeparableConv(512, 512, kernel_size=3, stride=2, padding=1) # 14 -> 7
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(512, 128)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 64)
        self.relu_fc2 = nn.ReLU(inplace=True)
        
        # output heads for steering and throttle
        self.steering_output = nn.Linear(64, 1)
        self.throttle_output = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        
        steering = self.steering_output(x)
        throttle = self.throttle_output(x)
        
        return steering, throttle

class TeacherCNN(nn.Module):
    def __init__(self):
        super(TeacherCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 56 * 56, 128)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 64)
        self.relu_fc2 = nn.ReLU(inplace=True)
        
        self.steering_output = nn.Linear(64, 1)
        self.throttle_output = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        steering = self.steering_output(x)
        throttle = self.throttle_output(x)
        return steering, throttle

if __name__ == "__main__":
    model = ImprovedCNN()
    x = torch.randn(1, 3, 224, 224)
    steering, throttle = model(x)
    print("Steering output shape:", steering.shape)
    print("Throttle output shape:", throttle.shape)
