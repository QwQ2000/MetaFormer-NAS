from poolformer import poolformer_s12
from torch import nn, optim
import torch
from data import Flowers
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torchvision.transforms import *
from enas.controller import Controller
from enas.config import get_args
from enas.utils import get_variable
import random
import pickle as pkl
import time 

train_data = Flowers(div = 'train', transforms = Compose([RandomResizedCrop((224,224)),
                                                          RandomHorizontalFlip(),
                                                          ToTensor()]))
val_data, test_data = Flowers(div = 'val'), Flowers(div = 'test')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = poolformer_s12(mixer_type='meta')
model.fc = nn.Linear(512, 102)
model = model.to(device)

controller = Controller(get_args()[0]).to(device)

def get_batch_acc(arch):
    model.eval()
    loader = DataLoader(val_data, batch_size=128, shuffle=True)
    x, y = next(iter(loader))
    x = x.to(device)
    with torch.no_grad():
        pred = model(x, arch)

    return np.mean((torch.argmax(pred, 1).cpu() == y).numpy())

shared_train_epoch = 1

def train_shared():
    controller.eval()

    arch = controller.sample()[0]

    loader = DataLoader(train_data, batch_size=128, shuffle=True, pin_memory=True, num_workers=4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss().to(device)

    print('Training shared model...')
    for i in range(shared_train_epoch):
        losses = []
        for idx, (x ,y) in enumerate(loader):
            model.train()
            x, y = x.to(device), y.to(device)
            pred = model(x, arch)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())

        if i == shared_train_epoch - 1:
            print('Train shared model: batch_acc:{} cls_loss:{}'.format(get_batch_acc(arch), np.mean(losses)))

controller_train_step = 100

def train_controller():
    controller.train()
    optimizer = optim.Adam(controller.parameters(), lr=3.5e-4)
    baseline = None

    print('Training controller...')

    batch_accs = []

    for step in range(controller_train_step):
        arch, log_probs, entropies = controller.sample(with_details=True)
        arch = arch[0]
        
        val_acc = get_batch_acc(arch)
        batch_accs.append(val_acc)
        np_entropies = entropies.data.cpu().numpy()
        rewards = val_acc + 1e-4 * np_entropies

        if baseline is None:
            baseline = rewards
        else:
            decay = 0.95
            baseline = decay * baseline + (1 - decay) * rewards
        adv = rewards - baseline

        loss = -log_probs * get_variable(adv, device, requires_grad=False)
        loss = loss.sum()  # or loss.mean()

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), 0.1)
        optimizer.step()

        if (step + 1) % 25 == 0:
            print('Step:{} mean_batch_acc:{} policy_loss:{}'.format(step + 1, np.mean(batch_accs), loss))

search_epoch = 100

def train():
    for epoch in range(search_epoch):
        train_shared()
        train_controller()
        if (epoch + 1) % 10 == 0:
            acc = eval_model(derive_arch())
            print('ENAS Epoch {}: val_acc{}'.format(epoch + 1, acc))

def derive_arch():
    controller.eval()
    candidates = [controller.sample()[0] for _ in range(10)]
    batch_accs = [get_batch_acc(arch) for arch in candidates]
    selected_arch = candidates[np.argmax(np.array(batch_accs))]
    return selected_arch

def eval_model(arch, div = 'val'):
    loader = DataLoader(val_data if div == 'val' else test_data, batch_size=1)
    gts = []
    preds = []
    
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            pred = np.argmax(model(x, arch).cpu().detach().numpy()[0])
        gt = y[0].numpy()
        
        preds.append(pred)
        gts.append(gt)
    
    return np.sum(np.array(gts) == np.array(preds)) / len(gts)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    setup_seed(42)
    t0 = time.time()
    train()
    print('Time:{}'.format(time.time() - t0))
    arch = derive_arch()
    with open('ckpts/derived_arch.pkl', 'wb') as f:
        pkl.dump(arch, f)