import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k):
        super().__init__()
        
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv1d(k, 64, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.batch_norm3 = nn.BatchNorm1d(1024)
        
        self.ffn1 = nn.Linear(1024, 512)
        self.batch_norm4 = nn.BatchNorm1d(512)
        self.ffn2 = nn.Linear(512, 256)
        self.batch_norm5 = nn.BatchNorm1d(256)
        self.ffn3 = nn.Linear(256, k*k)

        self.register_buffer('identity', torch.from_numpy(np.eye(k).flatten().astype(np.float32)).view(1, k ** 2))
        self.k = k

    def forward(self, x):
        b = x.shape[0]

        # Pass input through layers, applying the same max operation as in PointNetEncoder
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.relu(self.batch_norm3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
        
        x = self.relu(self.batch_norm4(self.ffn1(x)))
        x = self.relu(self.batch_norm5(self.ffn2(x)))
        x = self.ffn3(x)

        # Adding the identity to constrain the feature transformation matrix to be close to orthogonal matrix
        identity = self.identity.repeat(b, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, return_point_features=False):
        super().__init__()

        # Define convolution layers, batch norm layers, and ReLU
        self.mlp1 = nn.Conv1d(3, 64, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.mlp2 = nn.Conv1d(64, 128, kernel_size=1)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.mlp3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.batch_norm3 = nn.BatchNorm1d(1024)

        self.relu = nn.ReLU()

        self.input_transform_net = TNet(k=3)
        self.feature_transform_net = TNet(k=64)

        self.return_point_features = return_point_features

    def forward(self, x):
        num_points = x.shape[2]

        input_transform = self.input_transform_net(x)
        x = torch.bmm(x.transpose(2, 1), input_transform).transpose(2, 1)

        # First layer: 3->64
        x = self.relu(self.batch_norm1(self.mlp1(x)))

        feature_transform = self.feature_transform_net(x)
        x = torch.bmm(x.transpose(2, 1), feature_transform).transpose(2, 1)
        point_features = x

        # Layers 2 and 3: 64->128, 128->1024
        x = self.relu(self.batch_norm2(self.mlp2(x)))
        x = self.relu(self.batch_norm3(self.mlp3(x)))

        # This is the symmetric max operation
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.return_point_features:
            x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
            return torch.cat([x, point_features], dim=1)
        else:
            return x


class PointNetClassification(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = PointNetEncoder(return_point_features=False)
        # Add Linear layers, batch norms, dropout with p=0.3, and ReLU
        # Batch Norms and ReLUs are used after all but the last layer
        # Dropout is used only directly after the second Linear layer
        # The last Linear layer reduces the number of feature channels to num_classes (=k in the architecture visualization)
        self.ffn1 = nn.Linear(1024, 512)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.ffn2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.3)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.ffn3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)
        x = self.relu(self.batch_norm1(self.ffn1(x)))
        x = self.relu(self.batch_norm2(self.dropout(self.ffn2(x))))
        x = self.ffn3(x)
        return x


class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = PointNetEncoder(return_point_features=True)
        # Define convolutions, batch norms, and ReLU
        self.conv1 = nn.Conv1d(1088, 512, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=1)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=1)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, num_classes, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)
        # Pass x through all layers, no batch norm or ReLU after the last conv layer
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.relu(self.batch_norm3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        return x
