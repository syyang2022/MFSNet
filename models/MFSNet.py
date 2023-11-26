from base import BaseModel
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo
from utils.helpers import initialize_weights
from itertools import chain
from math import log
from torch.nn.parameter import Parameter

''' 
-> ResNet BackBone
'''

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1


class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet101', pretrained=True):
        super(ResNet, self).__init__()
        #print(backbone)
        model = getattr(models, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            initialize_weights(self.layer0)
        else:
            self.layer0 = nn.Sequential(*list(model.children())[:4])

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if output_stride == 16:
            s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8:
            s3, s4, d3, d4 = (1, 1, 2, 4)

        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'downsample.0' in n:
                    m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'downsample.0' in n:
                m.stride = (s4, s4)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        low_level_features = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_features





class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False,
                 BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        if dilation > kernel_size // 2:
            padding = dilation
        else:
            padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=padding,
                               dilation=dilation, groups=in_channels, bias=bias)
        self.bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, exit_flow=False, use_1st_relu=True):
        super(Block, self).__init__()

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        rep = []
        self.relu = nn.ReLU(inplace=True)

        rep.append(self.relu)
        rep.append(SeparableConv2d(in_channels, out_channels, 3, stride=1, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))

        rep.append(self.relu)
        rep.append(SeparableConv2d(out_channels, out_channels, 3, stride=1, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))

        rep.append(self.relu)
        rep.append(SeparableConv2d(out_channels, out_channels, 3, stride=stride, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))

        if exit_flow:
            rep[3:6] = rep[:3]
            rep[:3] = [
                self.relu,
                SeparableConv2d(in_channels, in_channels, 3, 1, dilation),
                nn.BatchNorm2d(in_channels)]

        if not use_1st_relu: rep = rep[1:]
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        output = self.rep(x)
        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else:
            skip = x

        x = output + skip
        return x


class EFAM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(EFAM, self).__init__()

        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.process1 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process2 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process3 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process4 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.compression = nn.Sequential(
            BatchNorm2d(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )



    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))

        x_list.append(self.process1((F.interpolate(self.scale1(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[0])))

        x_list.append((self.process2((F.interpolate(self.scale2(x),
                                                    size=[height, width],
                                                    mode='bilinear') + x_list[1]))))

        x_list.append(self.process3((F.interpolate(self.scale3(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[2])))

        x_list.append(self.process4((F.interpolate(self.scale4(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[3])))

        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)

        return out





class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=(1, 1), group=1, bn_act=False,
                 bias=False):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=group, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(out_channels)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.conv(x)

class shuffle_layer(nn.Module):


    def __init__(self, channel, groups=64):
        super(shuffle_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Hardsigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        #print(self.cweight)
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out




def branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channles),
        nn.ReLU(inplace=True))


class PSM(nn.Module):
    def __init__(self, in_channels, output_stride):
        super(PSM, self).__init__()

        assert output_stride in [8, 16], 'Only output strides of 8 or 16 are suported'
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]

        self.a1 = branch(in_channels, 256, 1, dilation=dilations[0])
        self.a2 = branch(in_channels, 256, 3, dilation=dilations[1])
        self.a3 = branch(in_channels, 256, 3, dilation=dilations[2])
        self.a4 = branch(in_channels, 256, 3, dilation=dilations[3])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.sa = shuffle_layer(256 * 5)
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        initialize_weights(self)

    def forward(self, x):
        x1 = self.a1(x)
        x2 = self.a2(x)
        x3 = self.a3(x)
        x4 = self.a4(x)
        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        x = self.sa(torch.cat((x1, x2, x3, x4, x5), dim=1))
        #print(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout(self.relu(x))

        return x


''' 
-> Decoder
'''


class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()

        #self.spp = DAPPM(low_level_channels, 128, low_level_channels // 2)
        #self.conv1 = nn.Conv2d(low_level_channels // 2, 48, 1, bias=False)
        self.spp = EFAM(low_level_channels, 128, 48)
        #self.egca1 = EGCA(low_level_channels)
        #self.eca = eca_layer(low_level_channels, 3)
        #self.pam = _PositionAttentionModule(low_level_channels)
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        self.downchannel = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=1, bias=False),#96->48
            BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        # self.upsample = Upsample(mode='learned-3x3',
        #                        channels=256)
        # Table 2, best performance with two 3x3 convs
        self.output = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1, stride=1),
        )
        initialize_weights(self)

    def forward(self, x, low_level_features):
        low_level_dappm = low_level_features
        #print(low_level_features.size())
        low_level_dappm = self.spp(low_level_dappm)
        #print(low_level_features.size())
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.relu(self.bn1(low_level_features))

        low_level_features = self.downchannel(torch.cat((low_level_features,low_level_dappm), dim=1))
        # low_level_dappm = self.spp(low_level_dappm)

        #print(low_level_dappm.size())
        #print(low_level_features.size())
        #low_level_features = self.downchannel(torch.cat(low_level_features,low_level_dappm))

        H, W = low_level_features.size(2), low_level_features.size(3)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        # x = self.upsample(x)
        # print(x.size())
        x = self.output(torch.cat((low_level_features, x), dim=1))
        return x


class MFSNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='resnet50', pretrained=True,
                 output_stride=16, freeze_bn=False, freeze_backbone=False, **_):

        super(MFSNet, self).__init__()
        assert ('xception' or 'resnet' in backbone)
        if 'resnet' in backbone:
            self.backbone = ResNet(in_channels=in_channels, output_stride=output_stride, pretrained=pretrained)
            low_level_channels = 256
        else:
            self.backbone = Xception(output_stride=output_stride, pretrained=pretrained)
            low_level_channels = 128

        self.PSM = PSM(in_channels=2048, output_stride=output_stride)
        self.decoder = Decoder(low_level_channels, num_classes)

        if freeze_bn: self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.backbone], False)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x, low_level_features = self.backbone(x)
        x = self.PSM(x)
        x = self.decoder(x, low_level_features)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x



    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PSM.parameters(), self.decoder.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


if __name__ == '__main__':
    img = torch.randn(2, 3, 473, 473)
    model = MFSNet(12)
    out = model(img)
    # print(out)