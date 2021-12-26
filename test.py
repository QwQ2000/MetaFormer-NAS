from poolformer import poolformer_s12
from torch import nn
from data import Flowers
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
import pickle as pkl

torch.backends.cudnn.benchmark = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_data = Flowers(div = 'test')

model = poolformer_s12(mixer_type='pooling')
model.fc = nn.Linear(512, 102)
#model.load_state_dict(torch.load('ckpts/pool_epoch590_acc699.pth'))
with open('ckpts/derived_arch.pkl', 'rb') as f:
    arch = pkl.load(f)
model = model.to(device)

def eval_model(arch=None, div = 'val'):
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
    print(eval_model(arch=arch, div = 'test'))