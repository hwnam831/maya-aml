import os
import numpy as np
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

filelist = os.listdir('logs')
labels = {
    'blackscholes':0,
    'bodytrack':1,
    'canneal':2,
    'freqmine':3,
    'vips':4,
    'streamcluster':5,
    'splash2x.radiosity':6,
    'splash2x.volrend':7,
    'splash2x.water_nsquared':8,
    'splash2x.water_spatial':9
}

class MayaDataset(Dataset):
    def __init__(self, logdir, window=500, std=None):
        self.window=window
        raw_filelist = os.listdir(logdir)
        self.matcher = re.compile(r"(.+)_(\d+)_log\.txt")
        self.dir = logdir
        self.filelist=[]
        for fname in raw_filelist:
            if self.matcher.match(fname):
                self.filelist.append(fname)
        
    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, idx):
        filename = self.filelist[idx]
        m = self.matcher.match(filename)
        label = labels[m.group(1)]
        trace = []
        with open(self.dir + '/'+filename,'r') as f:
            cnt = 0
            for line in f:
                if cnt == 0:
                    cnt = 1
                    continue
                tokens = line.split(" ")
                trace.append(float(tokens[1]))
                cnt += 1
        #offset = random.randint(0,tracelen-500)
        offset=0
        return np.array(trace[offset:offset+self.window],dtype=np.float32), label

if __name__ == '__main__':
    dataset = MayaDataset('logs', window=450)
    dsets = random_split(dataset, [800,200])
    trainset = dsets[0]
    trainloader = DataLoader(trainset, batch_size=16, num_workers=4)
    valloader = DataLoader(trainset, batch_size=16, num_workers=4)
    valset = dsets[1]
    clf = nn.Sequential(
        nn.Linear(dataset.window,512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512,10)
    ).cuda()
    optim_c = torch.optim.Adam(clf.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    for e in range(100):
        clf.train()
        for x,y in trainloader:
            optim_c.zero_grad()
            xdata, ydata = x.cuda(), y.cuda()
            output = clf(xdata)
            loss = criterion(output, ydata)
            loss.backward()
            optim_c.step()
        totcorrect = 0
        totcount = 0
        clf.eval()
        for x,y in valloader:
            xdata, ydata = x.cuda(), y.cuda()
            output = clf(xdata)
            pred = output.argmax(axis=-1)
            totcorrect += (pred==ydata).sum().item()
            totcount += y.size(0)
        macc = float(totcorrect)/totcount
        print("Epoch {} \t acc {:.4f}".format(e+1, macc))

        
    