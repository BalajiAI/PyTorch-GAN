import torch
from torchvision.utils import save_image

from dataloader import make_dataloader
from train_class import CycleGAN


path = './summer2winter_yosemite'
dataloader_A = make_dataloader(path=f'{path}/trainA')
dataloader_B = make_dataloader(path=f'{path}/trainB')

#HyperParameters
nb_features = 64
lr= 0.002
beta1= 0.5
beta2= 0.999
lambda_cycle= 10.0
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gan = CycleGAN(nb_features,lr,beta1,beta2,lambda_cycle,device)#Initializes our training class.

print('Training has started')

for iteration in range(10_000):
    real_A = next(iter(dataloader_A))
    real_B = next(iter(dataloader_B))
    gan.setup_input(real_A,real_B)
    loss_D = gan.optimize_D()
    loss_G = gan.optimize_G()
    if (iteration%200==0):#Print for every 200 iterations.
        print(f'loss_D={loss_D} loss_G={loss_G}')
    if(iteration%500==0):#Save the generated images for every 500 iterations.
        save_image(gan.fake_A.detach(),"generated_images/summer%d.png" % iteration, nrow=4, normalize=True) 
        save_image(gan.fake_B.detach(),"generated_images/winter%d.png" % iteration, nrow=4, normalize=True)  

print('Training has successfully completed')

#Save the model
torch.save(gan.G_AB,'Generator_AtoB.pth')
torch.save(gan.G_BA,'Generator_BtoA.pth')
torch.save(gan.D_A,'Discriminator_A.pth')
torch.save(gan.D_B,'Discriminator_B.pth')
