import torch
import torch.nn as nn

class AutoEncoder(torch.nn.Module):
    def __init__(self,input_size,output_size):
        super(AutoEncoder,self).__init__()
        self.input_size= input_size
        self.output_size = output_size


        self.Encoder = nn.Sequential(
            nn.Linear(input_size,int(input_size/2)),
            nn.Tanh(),
            nn.Linear(int(input_size/2),int(input_size/4)),
            nn.Tanh(),
            nn.Linear(int(input_size/4),output_size)
        )

        self.Decoder = nn.Sequential(
            nn.Linear(output_size, int(input_size / 4)),
            nn.Tanh(),
            nn.Linear(int(input_size / 4), int(input_size / 2)),
            nn.Tanh(),
            nn.Linear(int(input_size / 2), input_size)
        )


    def encoder(self,x):
        return self.Encoder(x)

    def decoder(self,x):
        return  self.Decoder(x)

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x
