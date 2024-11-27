import torch.nn as nn
import torch

class PointNet(nn.Layer):
    """
    Implement of `PointNet http://arxiv.org/abs/1612.00593`
    """
    def __init__(self, name_scope='PointNet_', num_classes=16, num_point=2048):
        super(PointNet, self).__init__()
        self.input_transform_net = nn.Sequential(
            nn.Conv1D(3, 64, 1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv1D(64, 128, 1),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.Conv1D(128, 1024, 1),
            nn.BatchNorm(1024),
            nn.ReLU(),
            nn.MaxPool1D(num_point)
        )
        self.input_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 9,
                weight_attr=torch.ParamAttr(initializer=torch.nn.initializer.Assign(torch.zeros((256, 9)))),
                bias_attr=torch.ParamAttr(initializer=torch.nn.initializer.Assign(torch.reshape(torch.eye(3), [-1])))
            )
        )
        self.mlp_1 = nn.Sequential(
            nn.Conv1D(3, 64, 1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv1D(64, 64, 1),
            nn.BatchNorm(64),
            nn.ReLU()
        )
        self.feature_transform_net = nn.Sequential(
            nn.Conv1D(64, 64, 1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv1D(64, 128, 1),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.Conv1D(128, 1024, 1),
            nn.BatchNorm(1024),
            nn.ReLU(),
            nn.MaxPool1D(num_point)
        )
        self.feature_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64*64)
        )
        self.mlp_2 = nn.Sequential(
            nn.Conv1D(64, 64, 1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv1D(64, 128, 1),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.Conv1D(128, 1024, 1),
            nn.BatchNorm(1024),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.7),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(axis=-1)
        )
    def forward(self, inputs):
        batchsize = inputs.shape[0]

        t_net = self.input_transform_net(inputs)
        t_net = torch.squeeze(t_net, axis=-1)
        t_net = self.input_fc(t_net)
        t_net = torch.reshape(t_net, [batchsize, 3, 3])

        x = torch.transpose(inputs, (0, 2, 1))
        x = torch.matmul(x, t_net)
        x = torch.transpose(x, (0, 2, 1))
        x = self.mlp_1(x)

        t_net = self.feature_transform_net(x)
        t_net = torch.squeeze(t_net, axis=-1)
        t_net = self.feature_fc(t_net)
        t_net = torch.reshape(t_net, [batchsize, 64, 64])

        x = torch.squeeze(x, axis=-1)
        x = torch.transpose(x, (0, 2, 1))
        x = torch.matmul(x, t_net)
        x = torch.transpose(x, (0, 2, 1))
        x = self.mlp_2(x)
        x = torch.max(x, axis=-1)
        x = torch.squeeze(x, axis=-1)
        x = self.fc(x)

        return x
