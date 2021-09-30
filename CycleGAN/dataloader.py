import os
import PIL.Image as Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
   
class ImageDataset(Dataset):
    
    def __init__(self,path):
        super().__init__()
        self.path = path
        self.train = os.listdir(path)
        self.image_transforms = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),])
   
    def __len__(self):
        return len(self.train)
    
    def __getitem__(self,idx):
        image = f'{self.path}/{self.train[idx]}'
        image = Image.open(image)
        if self.image_transforms:
            return self.image_transforms(image)
        return image
    

def make_dataloader(batch_size=8, **kwargs):
    dataset = ImageDataset(**kwargs)
    dataloader = DataLoader(dataset,batch_size=batch_size,num_workers=1,pin_memory=True,shuffle=True)
    return dataloader   
