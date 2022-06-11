#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torch
import torch.nn as nn
from torch.nn import init
import math
import torch.nn.functional as F
class HE(nn.Module):
    def __init__(self, inplanes):
        super(HE, self).__init__()
        self.inplanes = inplanes
        self.conv = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=1, padding=0)
        self.convh = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=1, padding=0)
        self.convw = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Softmax(dim=2)
        self.poolh = nn.AdaptiveAvgPool2d((None, 1))
        self.poolw = nn.AdaptiveAvgPool2d((1, None))
        self.fusion=ConvBNReLU(inplanes*2,inplanes,ks=3,stride=1)
    def forward(self, x):
        resudil=x
        batch_size,value_channels, h, w = x.size(0),x.size(1), x.size(2), x.size(3)
        x_v=self.conv(x)
        #print("x_v.size()",x_v.size())
        x_v=x_v.view(batch_size, value_channels, -1)
        #print("x_v.size()", x_v.size())
        x_m=self.conv1(x)
        #print("x_m.size()", x_m.size())
        x_m=x_m.view(batch_size, 1, -1)
        x_m = self.sigmoid(x_m)
        #print("x_m.size()", x_m.size())  # [batch_size, c, 1, w]
        x_h=self.convh(x)
        x_w=self.convw(x)
        #print("x_h.size()", x_h.size())
        #print("x_w.size()", x_w.size())
        x_h = self.poolh(x_h)
        x_w = self.poolw(x_w)
        #print("x_h.size()", x_h.size())
        #print("x_w.size()", x_w.size())
        attention=x_h*x_w
        attention=attention.view(batch_size,1,-1)
        attention = self.sigmoid(attention)
        #print("attention.size()", attention.size())
        #print("x_m.size()", x_m.size())
        #print("x_v.size()", x_v.size())
        y1=x_v * x_m
        y1=y1.view(batch_size, value_channels, h,w)
        #print("y1.size()", y1.size())
        y2=x_v * attention
        y2=y2.view(batch_size, value_channels, h,w)
        #print("y2.size()", y2.size())
        out1 = resudil + y1
        out2 = resudil + y2
        out=torch.cat((out2,out1),dim=1)
        out=self.fusion(out)
        #print("out.size()", out.size())
        return out
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params
class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        return torch.cat(out_list, dim=1) + x


class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        return out
class STD1(nn.Module):

    def __init__(self, in_chan, out_chan):
        super(STD1, self).__init__()
        first=in_chan//2
        second=first//2
        third=second//2
        fourth=third
        #self.conv1 = ConvX(in_chan, in_chan, 3, stride=1)
        self.f = nn.Sequential(
            nn.Conv2d(
                first, first, kernel_size=3, stride=1,
                padding=1,  bias=False),
            nn.BatchNorm2d(first),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.s = nn.Sequential(
            nn.Conv2d(
                second, second, kernel_size=3, stride=1,
                padding=1,bias=False),
            nn.BatchNorm2d(second),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.s1 = nn.Sequential(
            nn.Conv2d(
                second, second, kernel_size=3, stride=1,
                padding=1,  bias=False),
            nn.BatchNorm2d(second),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.t = nn.Sequential(
            nn.Conv2d(
                third, third, kernel_size=3, stride=1,
                padding=1,bias=False),
            nn.BatchNorm2d(third),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.t1 = nn.Sequential(
            nn.Conv2d(
                third, third, kernel_size=3, stride=1,
                padding=1,  bias=False),
            nn.BatchNorm2d(third),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.t2 = nn.Sequential(
            nn.Conv2d(
                third, third, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(third),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.four = nn.Sequential(
            nn.Conv2d(
                fourth, fourth, kernel_size=3, stride=1,
                padding=1,bias=False),
            nn.BatchNorm2d(fourth),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.four1= nn.Sequential(
            nn.Conv2d(
                fourth, fourth, kernel_size=3, stride=1,
                padding=1,  bias=False),
            nn.BatchNorm2d(fourth),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.four2= nn.Sequential(
            nn.Conv2d(
                fourth, fourth, kernel_size=3, stride=1,
                padding=1,  bias=False),
            nn.BatchNorm2d(fourth),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.four3= nn.Sequential(
            nn.Conv2d(
                fourth, fourth, kernel_size=3, stride=1,
                padding=1,  bias=False),
            nn.BatchNorm2d(fourth),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.four4= nn.Sequential(
            nn.Conv2d(
                fourth, fourth, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(fourth),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                2*in_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        #print(x.size())
        C=x.size()[1]
        #print(C)
        C1,C2,C3,C4=C//2,C//4,C//8,C//8
        X1,X2,X3,X4=torch.split(x,[C1,C2,C3,C4],dim=1)
        #print(X1.size())
        #print(X2.size())
        #print(X3.size())
        #print(X4.size())
        f=self.f(X1)
        #print("first.size()",f.size())
        s=self.s(X2)
        s1 = self.s1(s)
        sfinal = torch.cat((s, s1), dim=1)
        #print("second.size()",s.size())
        t=self.t(X3)
        t1 = self.t1(t)
        t2 = self.t2(t1)
        tfinal=torch.cat((X3,t,t1, t2), dim=1)
        #print("third.size()",t.size())
        four=self.four(X4)
        four1 = self.four1(four)
        four2 = self.four2(four1)
        four3 = self.four3(four2)
        fourfinal = torch.cat((four,four1,four2,four3), dim=1)
        #print("fourth.size()",four.size())
        feat = torch.cat((f, sfinal, tfinal,fourfinal), dim=1)
        feat = self.conv2(feat)
        feat = self.relu(feat)
        return feat

class STD2(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(STD2, self).__init__()
        first=in_chan//2
        second=first//2
        third=second//2
        fourth=third
        #self.conv1 = ConvX(in_chan, in_chan, 3, stride=1)
        self.f = nn.Sequential(
            nn.Conv2d(
                first, first, kernel_size=3, stride=1,
                padding=1,  bias=False),
            nn.BatchNorm2d(first),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.s = nn.Sequential(
            nn.Conv2d(
                second, second, kernel_size=3, stride=2,
                padding=1,  bias=False),
            nn.BatchNorm2d(second),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.s1 = nn.Sequential(
            nn.Conv2d(
                second, second, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(second),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.t = nn.Sequential(
            nn.Conv2d(
                third, third, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(third),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.t1 = nn.Sequential(
            nn.Conv2d(
                third, third, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(third),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.t2 = nn.Sequential(
            nn.Conv2d(
                third, third, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(third),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.four = nn.Sequential(
            nn.Conv2d(
                fourth, fourth, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(fourth),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.four1= nn.Sequential(
            nn.Conv2d(
                fourth, fourth, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(fourth),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.four2= nn.Sequential(
            nn.Conv2d(
                fourth, fourth, kernel_size=3, stride=1,
                padding=1,bias=False),
            nn.BatchNorm2d(fourth),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.four3= nn.Sequential(
            nn.Conv2d(
                fourth, fourth, kernel_size=3, stride=1,
                padding=1,bias=False),
            nn.BatchNorm2d(fourth),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.four4= nn.Sequential(
            nn.Conv2d(
                fourth, fourth, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(fourth),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                2*in_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        #print(x.size())
        C=x.size()[1]
        #print(C)
        C1,C2,C3,C4=C//2,C//4,C//8,C//8
        X1,X2,X3,X4=torch.split(x,[C1,C2,C3,C4],dim=1)
        #print(X1.size())
        #print(X2.size())
        #print(X3.size())
        #print(X4.size())
        f=self.f(X1)
        #print(f.size())
        f=F.adaptive_avg_pool2d(f,(x.size()[2]//2,x.size()[3]//2))
        #print(f.size())
        #print("first.size()",f.size())
        s=self.s(X2)
        s1 = self.s1(s)
        sfinal = torch.cat((s, s1), dim=1)
        #print("second.size()",s.size())
        t=self.t(X3)
        t1 = self.t1(t)
        t2 = self.t2(t1)
        X3=F.adaptive_avg_pool2d(X3,(x.size()[2]//2,x.size()[3]//2))
        #print(X3.size())
        tfinal=torch.cat((X3,t,t1, t2), dim=1)
        #print("third.size()",t.size())
        four=self.four(X4)
        four1 = self.four1(four)
        four2 = self.four2(four1)
        four3 = self.four3(four2)
        fourfinal = torch.cat((four,four1,four2,four3), dim=1)
        #print("fourth.size()",four.size())
        feat = torch.cat((f, sfinal, tfinal,fourfinal), dim=1)
        feat = self.conv2(feat)
        feat = self.relu(feat)
        return feat



# STDC2Net
class STDCNet1446(nn.Module):
    def __init__(self, base=64, layers=[4, 5, 3], block_num=4, type="cat", num_classes=1000, dropout=0.20,
                 pretrain_model=False, use_conv_last=False):
        super(STDCNet1446, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block)
        self.conv_last = ConvX(base * 16, max(1024, base * 16), 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(max(1024, base * 16), max(1024, base * 16), bias=False)
        self.bn = nn.BatchNorm1d(max(1024, base * 16))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(max(1024, base * 16), num_classes, bias=False)

        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:6])
        self.x16 = nn.Sequential(self.features[6:11])
        self.x32 = nn.Sequential(self.features[11:])

        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()

    def init_weight(self, pretrain_model):

        state_dict = torch.load(pretrain_model)["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        # features = []
        # features += [ConvX(3, base // 2, 3, 2)]
        # features += [ConvX(base // 2, base, 3, 2)]
        #
        # for i, layer in enumerate(layers):
        #     for j in range(layer):
        #         if i == 0 and j == 0:
        #             features.append(block(base, base * 4, block_num, 2))
        #         elif j == 0:
        #             features.append(block(base * int(math.pow(2, i + 1)), base * int(math.pow(2, i + 2)), block_num, 2))
        #         else:
        #             features.append(block(base * int(math.pow(2, i + 2)), base * int(math.pow(2, i + 2)), block_num, 1))
        features = []
        features += [ConvX(3, base//2, 3, 2)]
        features += [ConvX(base//2, base, 3, 2)]
        features+=[STD2(64,256),STD1(256,256),STD1(256,256),STD1(256,256)]
        features += [STD2(256,512),STD1(512,512),STD1(512,512),STD1(512,512),STD1(512,512)]
        features += [STD2(512,1024),STD1(1024,1024),STD1(1024,1024)]
        return nn.Sequential(*features)

    def forward(self, x):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)
        if self.use_conv_last:
            feat32 = self.conv_last(feat32)

        return feat2, feat4, feat8, feat16, feat32

    def forward_impl(self, x):
        out = self.features(x)
        out = self.conv_last(out).pow(2)
        out = self.gap(out).flatten(1)
        out = self.fc(out)
        # out = self.bn(out)
        out = self.relu(out)
        # out = self.relu(self.bn(self.fc(out)))
        out = self.dropout(out)
        out = self.linear(out)
        return out


# STDC1Net
class STDCNet813(nn.Module):
    def __init__(self, base=64, layers=[2, 2, 2], block_num=4, type="cat", num_classes=1000, dropout=0.20,
                 pretrain_model=False, use_conv_last=False):
        super(STDCNet813, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block)
        self.conv_last = ConvX(base * 16, max(1024, base * 16), 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(max(1024, base * 16), max(1024, base * 16), bias=False)
        self.bn = nn.BatchNorm1d(max(1024, base * 16))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(max(1024, base * 16), num_classes, bias=False)

        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:4])
        self.x16 = nn.Sequential(self.features[4:6])
        self.x32 = nn.Sequential(self.features[6:])

        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()

    def init_weight(self, pretrain_model):

        state_dict = torch.load(pretrain_model)["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvX(3, base//2, 3, 2)]
        features += [ConvX(base//2, base, 3, 2)]
        #
        features+=[STD2(64,256),STD1(256,256)]
        features += [STD2(256,512),STD1(512,512)]
        features += [STD2(512,1024),STD1(1024,1024)]

        return nn.Sequential(*features)

    def forward(self, x):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)
        if self.use_conv_last:
            feat32 = self.conv_last(feat32)

        return feat2, feat4, feat8, feat16, feat32

    def forward_impl(self, x):
        out = self.features(x)
        out = self.conv_last(out).pow(2)
        out = self.gap(out).flatten(1)
        out = self.fc(out)
        # out = self.bn(out)
        out = self.relu(out)
        # out = self.relu(self.bn(self.fc(out)))
        out = self.dropout(out)
        out = self.linear(out)
        return out


# from modules.bn import InPlaceABNSync as BatchNorm2d
# BatchNorm2d = nn.BatchNorm2d

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        # self.bn = BatchNorm2d(out_chan)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        # self.bn_atten = BatchNorm2d(out_chan)
        self.bn_atten = nn.BatchNorm2d(out_chan)

        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, backbone='CatNetSmall', pretrain_model=False, use_conv_last=False, *args, **kwargs):
        super(ContextPath, self).__init__()

        self.backbone_name = backbone
        if backbone == 'STDCNet1446':
            self.backbone = STDCNet1446(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
            self.arm16 = AttentionRefinementModule(512, 128)
            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
            self.arm32 = AttentionRefinementModule(inplanes, 128)
            self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_avg = ConvBNReLU(inplanes, 128, ks=1, stride=1, padding=0)

        elif backbone == 'STDCNet813':
            self.backbone = STDCNet813(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
            self.arm16 = AttentionRefinementModule(512, 128)
            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
            self.arm32 = AttentionRefinementModule(inplanes, 128)
            self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_avg = ConvBNReLU(inplanes, 128, ks=1, stride=1, padding=0)
        else:
            print("backbone is not in backbone lists")
            exit(0)

        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]

        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])

        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat2, feat4, feat8, feat16, feat16_up, feat32_up  # x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                               out_chan // 4,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4,
                               out_chan,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.H = HE(256)
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        #atten = F.avg_pool2d(feat, feat.size()[2:])
        #atten = self.conv1(atten)
        #atten = self.relu(atten)
        #atten = self.conv2(atten)
        #atten = self.sigmoid(atten)
        feat_out = self.H(feat)
        #feat_atten = torch.mul(feat, atten)
        #feat_out = feat_atten + feat
        #feat_out = self.H(feat_out)
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNetV1(nn.Module):
    def __init__(self, n_classes, aux_mode='train', *args, **kwargs):
        super(BiSeNetV1, self).__init__()

        # self.heat_map = heat_map
        backbone='STDCNet1446'
        self.cp = ContextPath(backbone, False, use_conv_last=False)

        if backbone == 'STDCNet1446':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes

        elif backbone == 'STDCNet813':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes

        else:
            print("backbone is not in backbone lists")
            exit(0)

        self.ffm = FeatureFusionModule(inplane, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(conv_out_inplanes, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(conv_out_inplanes, 64, n_classes)

        self.conv_out_sp16 = BiSeNetOutput(sp16_inplanes, 64, 1)

        self.conv_out_sp8 = BiSeNetOutput(sp8_inplanes, 64, 1)
        self.conv_out_sp4 = BiSeNetOutput(sp4_inplanes, 64, 1)
        self.conv_out_sp2 = BiSeNetOutput(sp2_inplanes, 64, 1)
        self.aux_mode = aux_mode
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]

        feat_res2, feat_res4, feat_res8, feat_res16, feat_cp8, feat_cp16 = self.cp(x)

        feat_out_sp2 = self.conv_out_sp2(feat_res2)

        feat_out_sp4 = self.conv_out_sp4(feat_res4)

        feat_out_sp8 = self.conv_out_sp8(feat_res8)

        feat_out_sp16 = self.conv_out_sp16(feat_res16)

        feat_fuse = self.ffm(feat_res8, feat_cp8)
        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
        if self.aux_mode == 'train':
            return feat_out, feat_out16, feat_out32
        elif self.aux_mode == 'eval':
            return feat_out,
        elif self.aux_mode == 'pred':
            feat_out = feat_out.argmax(dim=1)
            return feat_out
        else:
            raise NotImplementedError


    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params
def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1 

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)
if __name__ == "__main__":
    net = BiSeNetV1(19)
    in_ten = torch.randn(2, 3, 768, 1536)
    out, out16, out32 = net(in_ten)
    print(model_structure(net))
    print(out.shape)
    print(out16.shape)
    print(out32.shape)



