from torch.optim import optimizer
from poolformer import poolformer_s12
from torch import nn, optim
import torch
from data import Flowers
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

train_data, val_data, test_data = Flowers(div = 'train'), Flowers(div = 'val'), Flowers(div = 'test')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = poolformer_s12().to(device)

n_epochs = 20

def train():
    loader = DataLoader(train_data, batch_size=1, shuffle=True, pin_memory=True)
    optimizer = optim.AdamW(model.parameters(),lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss().to(device)

    for i in range(n_epochs):
        losses = []
        for idx, (x ,y) in tqdm(enumerate(loader)):
            model.train()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())

        print('Epoch:{} Accuracy:{} Loss:{}'.format(i + 1, eval_model(), np.mean(losses)))
    
def eval_model(div = 'val'):
    loader = DataLoader(val_data if div == 'val' else test_data, batch_size=1)
    gts = []
    preds = []
    
    model.eval()
    for idx, (x, y) in tqdm(enumerate(loader)):
        x = x.to(device)
        pred = np.argmax(model(x).cpu().detach().numpy()[0])
        gt = y[0].numpy()
        
        preds.append(pred)
        gts.append(gt)
    
    return np.sum(np.array(gts) == np.array(preds)) / len(gts)

if __name__ == '__main__':
    train()
    eval_model(div = 'test')
    pass