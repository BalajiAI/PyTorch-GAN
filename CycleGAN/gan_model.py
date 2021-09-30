import torch
import torch.nn as nn

# Generator
class ConvBlock(nn.Module):
  
    def __init__(self,in_channels,out_channels,down=True,act=True,**kwargs):
        super().__init__()
        self.conv = nn.Sequential(
                             nn.Conv2d(in_channels=in_channels,out_channels=out_channels,padding_mode='reflect',**kwargs)
                             if down else nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,**kwargs),
                             nn.InstanceNorm2d(out_channels),
                             nn.ReLU(inplace=True) if act else nn.Identity(),
                                )

    def forward(self,x):
        return self.conv(x)
    

class ResidualBlock(nn.Module):
  
    def __init__(self,channels,**kwargs):
        super().__init__()
        self.main = nn.Sequential(
                             ConvBlock(channels,channels,kernel_size=3, padding=1),
                             ConvBlock(channels,channels,act=False,kernel_size=3, padding=1)
                                 )

    def forward(self,x):
        return x + self.main(x)
    

class Generator(nn.Module):

    def __init__(self,nb_features,nb_residualblock=9):
        super().__init__()
        self.first = ConvBlock(3,nb_features,kernel_size=7,stride=1,padding=3)
        self.down = nn.ModuleList(
                        [ConvBlock(nb_features,nb_features*2,kernel_size=3, stride=2, padding=1),
                        ConvBlock(nb_features*2,nb_features*4,kernel_size=3, stride=2, padding=1)]
                                 )
        self.residual_block = nn.Sequential(*[ResidualBlock(nb_features*4) for _ in range(nb_residualblock)])
        self.up = nn.ModuleList(
                        [ConvBlock(nb_features*4,nb_features*2,down=False,kernel_size=3, stride=2, padding=1, output_padding=1),
                        ConvBlock(nb_features*2,nb_features,down=False,kernel_size=3, stride=2, padding=1, output_padding=1)]
                               )
        self.last = nn.Conv2d(nb_features,3,kernel_size=7,stride=1,padding=3, padding_mode="reflect")

    def forward(self,x):
        x = self.first(x)
        for layer in self.down:
            x = layer(x)
        x = self.residual_block(x)
        for layer in self.up:
            x = layer(x)
        x = nn.Tanh()(self.last(x))
        return x
        
#Patch Discriminator    
class Block(nn.Module):

    def __init__(self,in_channels,out_channels,stride):
        super().__init__()
        self.conv = nn.Sequential(
                          nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=stride,padding=1,padding_mode='reflect'),
                          nn.InstanceNorm2d(out_channels),
                          nn.LeakyReLU(0.2,inplace=True),
                             )
    def forward(self,x):
        return self.conv(x)
    

class Discriminator(nn.Module):

    def __init__(self,nb_features):
        super().__init__()
        self.first = nn.Sequential(
                         nn.Conv2d(3,nb_features,kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
                         nn.LeakyReLU(0.2,inplace=True),
                              )
        self.convblock = nn.ModuleList(
                              [Block(nb_features,nb_features*2,2),
                               Block(nb_features*2,nb_features*4,2),
                               Block(nb_features*4,nb_features*6,1)]
                                  )
        self.last = nn.Sequential(
                        nn.Conv2d(nb_features*6,1,kernel_size=4,padding=1,padding_mode='reflect'),
                        nn.Sigmoid()
                             )
    def forward(self,x):
        x = self.first(x)
        for layer in self.convblock:    
            x = layer(x)
        x = self.last(x)
        return x     
