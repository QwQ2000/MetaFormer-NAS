from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import *

class Flowers(Dataset):
    def __init__(self, 
                 path = '/home/qwq/Datasets/flowers',
                 div = 'train',
                 transforms = None):
        self.path = path

        self.labels = []
        with open(os.path.join(path,'annotations.txt'),'r') as f:
            for line in f.readlines():
                self.labels.append(int(line.split(' ')[1]))

        self.files = []
        with open(os.path.join(path, div + '.txt'),'r') as f:
            for line in f.readlines():
                self.files.append(line[:-1])
        
        self.transforms = transforms if transforms is not None else Compose([Resize((224,224)), ToTensor()])
    
    def __getitem__(self, idx):
        img = self.transforms(
                Image.open(
                    os.path.join(self.path, 'jpg', self.files[idx])
                )
            )
        label = self.labels[
                    int(self.files[idx][6:11])
                ]
        return img, label
    
    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    print(Flowers()[300])