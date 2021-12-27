from poolformer import poolformer_s12
from torch import nn
from data import Flowers
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
import pickle as pkl
from torchsummary import summary

torch.backends.cudnn.benchmark = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_data = Flowers(div = 'test')

model = poolformer_s12(mixer_type='attention')
model.fc = nn.Linear(512, 102)
model.load_state_dict(torch.load('ckpts/mhsa_epoch580_acc665.pth'))
with open('ckpts/derived_arch_7node.pkl', 'rb') as f:
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
    #summary(model, input_size=(3, 224, 224), batch_size=-1)
    print(eval_model(arch=arch, div = 'test'))