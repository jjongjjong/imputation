
import torch
import torch.nn as nn

class Conv_AE(nn.Module):
    def __init__(self):
        super(Conv_AE ,self).__init__()

        self.Encoder = nn.Sequential(
            nn.Conv1d(1 ,16 ,10 ,3),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.Conv1d(16 ,32 ,10 ,2),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Conv1d(32 ,64 ,10 ,2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64 ,128 ,10 ,2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(128 ,256 ,10 ,2)
        )

        self.Decoder = nn.Sequential(
            nn.ConvTranspose1d(256 ,128 ,10 ,1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(128 ,64 ,10 ,1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(64 ,32 ,10 ,1),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.ConvTranspose1d(32 ,16 ,10 ,4 ,output_padding=1),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.ConvTranspose1d(16 ,1 ,10 ,5),
            nn.Sigmoid()
        )

    def encoder(self,x):
        x = x.view(-1,1,720)
        return self.Encoder(x)

    def decoder(self,x):
        return  self.Decoder(x).view(-1,720)

    def forward(self ,x):
        x = x.view(-1,1,720)
        encode = self.Encoder(x)
        decode = self.Decoder(encode)
        return decode.view(-1,720)
