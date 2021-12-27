from poolformer import poolformer_s12
from torch import nn, optim
import torch
from data import Flowers
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torchvision.transforms import *
import pickle as pkl

torch.backends.cudnn.benchmark = True

train_data = Flowers(div = 'train', transforms = Compose([RandomResizedCrop((224,224)),
                                                          RandomHorizontalFlip(),
                                                          ToTensor()]))
val_data, test_data = Flowers(div = 'val'), Flowers(div = 'test')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = poolformer_s12(mixer_type='attention')
model.fc = nn.Linear(512, 102)
model = model.to(device)
#with open('ckpts/derived_arch.pkl', 'rb') as f:
#    arch = pkl.load(f)
arch = None

n_epochs = 600
observed_epochs = set([i for i in range(n_epochs) if (i + 1) % 10 == 0])

def train():
    loader = DataLoader(train_data, batch_size=128, shuffle=True, pin_memory=True, num_workers=4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay = 1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.85)
    criterion = nn.CrossEntropyLoss().to(device)

    for i in range(n_epochs):
        losses = []
        for idx, (x ,y) in tqdm(enumerate(loader)):
            model.train()
            x, y = x.to(device), y.to(device)
            pred = model(x, arch)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
        scheduler.step()
        if i in observed_epochs:
            acc = eval_model()
            print('Epoch:{} Accuracy:{} Loss:{}'.format(i + 1, acc, np.mean(losses)))
            torch.save(model.state_dict(), 'ckpts/epoch{}_acc{}.pth'.format(i + 1, int(acc * 1000)))
    
def eval_model(div = 'val'):
    loader = DataLoader(val_data if div == 'val' else test_data, batch_size=1)
    gts = []
    preds = []
    
    model.eval()
    for idx, (x, y) in tqdm(enumerate(loader)):
        x = x.to(device)
        with torch.no_grad():
            pred = np.argmax(model(x, arch).cpu().detach().numpy()[0])
        gt = y[0].numpy()
        
        preds.append(pred)
        gts.append(gt)
    
    return np.sum(np.array(gts) == np.array(preds)) / len(gts)

if __name__ == '__main__':
    train()
    print(eval_model(div = 'test'))
    torch.save(model.state_dict(), 'ckpts/epoch{}.pth'.format(n_epochs))