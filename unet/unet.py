# https://aistudio.baidu.com/aistudio/projectdetail/3340332?channelType=0&channel=0
"""
paddlepaddle-gpu==2.2.1
time:2021.07.16 9:00
author:CP
backbone：U-net
"""
import paddle
from paddle import nn


class Encoder(nn.Layer):  # 下采样：两层卷积，两层归一化，最后池化。
    def __init__(self, num_channels, num_filters):
        super(Encoder, self).__init__()  # 继承父类的初始化
        self.conv1 = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=3,  # 3x3卷积核，步长为1，填充为1，不改变图片尺寸[H W]
            stride=1,
            padding=1,
        )
        self.bn1 = nn.BatchNorm(num_filters, act="relu")  # 归一化，并使用了激活函数

        self.conv2 = nn.Conv2D(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm(num_filters, act="relu")

        self.pool = nn.MaxPool2D(
            kernel_size=2, stride=2, padding="SAME"
        )  # 池化层，图片尺寸减半[H/2 W/2]

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x_conv = x  # 两个输出，灰色 ->
        x_pool = self.pool(x)  # 两个输出，红色 |
        return x_conv, x_pool


class Decoder(nn.Layer):  # 上采样：一层反卷积，两层卷积层，两层归一化
    def __init__(self, num_channels, num_filters):
        super(Decoder, self).__init__()
        self.up = nn.Conv2DTranspose(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=2,
            stride=2,
            padding=0,
        )  # 图片尺寸变大一倍[2*H 2*W]

        self.conv1 = nn.Conv2D(
            in_channels=num_filters * 2,
            out_channels=num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn1 = nn.BatchNorm(num_filters, act="relu")

        self.conv2 = nn.Conv2D(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm(num_filters, act="relu")

    def forward(self, input_conv, input_pool):
        x = self.up(input_pool)
        h_diff = input_conv.shape[2] - x.shape[2]
        w_diff = input_conv.shape[3] - x.shape[3]
        pad = nn.Pad2D(
            padding=[
                h_diff // 2,
                h_diff - h_diff // 2,
                w_diff // 2,
                w_diff - w_diff // 2,
            ]
        )
        x = pad(x)  # 以下采样保存的feature map为基准，填充上采样的feature map尺寸
        x = paddle.concat(x=[input_conv, x], axis=1)  # 考虑上下文信息，in_channels扩大两倍
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x


class UNet(nn.Layer):
    def __init__(self, num_classes=3):
        super(UNet, self).__init__()
        self.down1 = Encoder(num_channels=3, num_filters=64)  # 下采样
        self.down2 = Encoder(num_channels=64, num_filters=128)
        self.down3 = Encoder(num_channels=128, num_filters=256)
        self.down4 = Encoder(num_channels=256, num_filters=512)

        self.mid_conv1 = nn.Conv2D(512, 1024, 1)  # 中间层
        self.mid_bn1 = nn.BatchNorm(1024, act="relu")
        self.mid_conv2 = nn.Conv2D(1024, 1024, 1)
        self.mid_bn2 = nn.BatchNorm(1024, act="relu")

        self.up4 = Decoder(1024, 512)  # 上采样
        self.up3 = Decoder(512, 256)
        self.up2 = Decoder(256, 128)
        self.up1 = Decoder(128, 64)

        self.last_conv = nn.Conv2D(64, num_classes, 1)  # 1x1卷积，softmax做分类

    def forward(self, inputs):
        x1, x = self.down1(inputs)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)

        x = self.mid_conv1(x)
        x = self.mid_bn1(x)
        x = self.mid_conv2(x)
        x = self.mid_bn2(x)

        x = self.up4(x4, x)
        x = self.up3(x3, x)
        x = self.up2(x2, x)
        x = self.up1(x1, x)

        x = self.last_conv(x)

        return x


# 查看网络各个节点的输出信息
# paddle.summary(UNet(), (1, 3, 600, 600))
