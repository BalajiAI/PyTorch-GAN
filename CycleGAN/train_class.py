import torch
import torch.nn as nn
import itertools
from gan_model import Generator,Discriminator


class CycleGAN(nn.Module):
    
    def __init__(self,nb_features,lr,beta1,beta2,lambda_cycle,device):
        super().__init__()
        self.G_AB = Generator(nb_features).to(device)
        self.G_BA = Generator(nb_features).to(device)
        self.D_A  = Discriminator(nb_features).to(device)
        self.D_B  = Discriminator(nb_features).to(device)
        self.adversarial_loss = nn.MSELoss()
        self.cycle_loss = nn.L1Loss()
        self.opt_G = torch.optim.Adam(itertools.chain(self.G_AB.parameters(),self.G_BA.parameters()),lr=lr,betas=(beta1,beta2))
        self.opt_D_A  = torch.optim.Adam(self.D_A.parameters(),lr=lr,betas=(beta1,beta2))
        self.opt_D_B  = torch.optim.Adam(self.D_B.parameters(),lr=lr,betas=(beta1,beta2))
        self.lambda_cycle = lambda_cycle
    
    def setup_input(self,real_A,real_B):
        self.real_A = real_A.to(device)
        self.real_B = real_B.to(device)
        self.fake_A = self.G_BA(self.real_B)
        self.fake_B = self.G_AB(self.real_A)
    
    def optimize_D(self):
        self.D_A.train()
        self.D_B.train()
        real_preds = self.D_A(self.real_A)
        fake_preds = self.D_A(self.fake_A.detach())
        real_loss = self.adversarial_loss(real_preds,torch.ones_like(real_preds,device=device))
        fake_loss = self.adversarial_loss(fake_preds,torch.zeros_like(fake_preds,device=device))
        loss_D_A = (real_loss+fake_loss)/2
        
        real_preds = self.D_B(self.real_B)
        fake_preds = self.D_B(self.fake_B.detach())
        real_loss = self.adversarial_loss(real_preds,torch.ones_like(real_preds,device=device))
        fake_loss = self.adversarial_loss(fake_preds,torch.zeros_like(fake_preds,device=device))
        loss_D_B = (real_loss+fake_loss)/2
    
        self.opt_D_A.zero_grad()
        loss_D_A.backward()
        self.opt_D_A.step()
        
        self.opt_D_B.zero_grad()
        loss_D_B.backward()
        self.opt_D_B.step()
        
        return (loss_D_A+loss_D_B)/2
        
    def optimize_G(self):
        self.G_AB.train()
        self.G_BA.train()
        fake_preds_A = self.D_A(self.fake_A)
        fake_preds_B = self.D_B(self.fake_B)
        adversarial_loss_G_AB = self.adversarial_loss(fake_preds_B,torch.ones_like(fake_preds_B,device=device))
        adversarial_loss_G_BA = self.adversarial_loss(fake_preds_A,torch.ones_like(fake_preds_A,device=device))
        adversarial_loss_G = (adversarial_loss_G_AB + adversarial_loss_G_BA)/2
        
        cycle_loss_G_AB = self.cycle_loss(self.real_A,self.G_BA(self.fake_B))
        cycle_loss_G_BA = self.cycle_loss(self.real_B,self.G_AB(self.fake_A))
        cycle_loss_G = (cycle_loss_G_AB + cycle_loss_G_BA)/2
        loss_G = adversarial_loss_G + (self.lambda_cycle*cycle_loss_G)
    
        self.opt_G.zero_grad()
        loss_G.backward()
        self.opt_G.step()
        
        return loss_G
