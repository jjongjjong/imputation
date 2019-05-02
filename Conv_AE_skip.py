import torch
import torch.nn as nn

def conv1d_5(indim,outdim,stride = 1):
    return nn.Conv1d(indim,outdim,kernel_size=5,stride=stride,padding=2,bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv1d_5(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1d_5(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.Tanh = nn.Tanh()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.Tanh(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.Tanh(out)
        return out


# ResNet
class Resnet_Encoder(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(Resnet_Encoder, self).__init__()
        self.in_channels = 8
        self.conv = conv1d_5(1, 8)
        self.bn = nn.BatchNorm1d(8)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 8, layers[0])
        self.layer2 = self.make_layer(block, 16, layers[1], 2)
        self.layer3 = self.make_layer(block, 32, layers[2], 2)
        self.layer4 = self.make_layer(block, 64, layers[3], 2)
        self.layer5 = self.make_layer(block, 128, layers[4], 2)

        # self.avg_pool = nn.AvgPool2d(8)
        # self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv1d_5(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm1d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1,1,720)

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        # out = self.avg_pool(out)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        return out



def transconv1d_5(indim,outdim,stride=1,padding=2,output_padding=0):
    if stride==2 :
        padding=2
        output_padding=1
    return nn.ConvTranspose1d(indim,outdim,5,stride,padding,output_padding,bias=False)

class ResidualBlock_trans(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, upsample=None):
        super(ResidualBlock_trans, self).__init__()
        self.conv1 = transconv1d_5(in_channels, out_channels,stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = transconv1d_5(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.upsample = upsample
        self.Tanh = nn.Tanh()

    def forward(self, x):

        residual = x
        #print(residual.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.Tanh(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.upsample:
            residual = self.upsample(x)
            #print(residual.shape)
        out += residual
        out = self.Tanh(out)
        return out


# ResNet
class Resnet_Decoder(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(Resnet_Decoder, self).__init__()
        self.in_channels = 128
        #self.conv = nn.Conv1d(8, 1)
        self.bn = nn.BatchNorm1d(1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 128, layers[0],2)
        self.layer2 = self.make_layer(block, 64, layers[1],2)
        self.layer3 = self.make_layer(block, 32, layers[2],2)
        self.layer4 = self.make_layer(block, 16, layers[3],2)

        self.conv_last = nn.Conv1d(16,1,1)
        self.Tanh = nn.Tanh()
        #self.avg_pool = nn.AvgPool2d(1)
        # self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        upsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            upsample = nn.Sequential(
                transconv1d_5(self.in_channels, out_channels, stride=2,padding=2,output_padding=1),
                nn.BatchNorm1d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, upsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = self.conv(x)
        # out = self.bn(out)
        # out = self.relu(out)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.conv_last(out)
        out = self.bn(out)
        out = self.Tanh(out)
        out = torch.clamp(out,min=0)
        out = torch.squeeze(out,dim=1)

        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        return out


class Resnet_AE(nn.Module):
    def __init__(self,encode_blocks,decode_blocks):
        super(Resnet_AE,self).__init__()
        self.encoder = Resnet_Encoder(ResidualBlock,encode_blocks)
        self.decoder = Resnet_Decoder(ResidualBlock_trans,decode_blocks)

    def forward(self, x):

        out = self.encoder(x)
        out = self.decoder(out)
        return out


# encoder = Resnet_Encoder(ResidualBlock,[2,2,2,2,2])
# decoder = Resnet_Decoder(ResidualBlock_trans,[2,2,2,2])
# input = torch.ones(2,1,720)
# encoded_vector = encoder(input)
# print(encoded_vector.shape)
# decoded_vector = decoder(encoded_vector)
#
# print(decoded_vector.shape)
# print(decoded_vector)
#
# resnet_ae = Resnet_AE([2,2,2,2,2],[2,2,2,2])
# out = resnet_ae(torch.ones(2,1,720))
# print(out.shape)