import os
import numpy as np
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import random

labels = {
    'blackscholes':0,
    'bodytrack':1,
    'swaptions':2,
    'freqmine':3,
    'vips':4,
    'streamcluster':5,
    'splash2x.radiosity':6,
    'splash2x.volrend':7,
    'splash2x.water_nsquared':8,
    'splash2x.water_spatial':9
}

class MayaDataset(Dataset):
    def __init__(self, logdir, minpower, maxpower, window=1000):
        self.window=window
        self.minpower = minpower
        self.maxpower = maxpower
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
                val = float(tokens[1])
                if( val < 0):
                    continue
                trace.append(val)
                cnt += 1
        offset = random.randint(0,len(trace)-self.window)
        offset = min(offset, random.randint(0,50))
        offset=0
        arr = np.array(trace[offset:offset+self.window],dtype=np.float32)

        return (arr-self.minpower)/(self.maxpower-self.minpower), label

class CNNCLF(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Conv1d(1,32,16,8,4),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            nn.Conv1d(32,64,3,padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        self.resblock = nn.Sequential(
            nn.Conv1d(64,32,1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32,32,1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32,64,3,padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        self.resblock2 = nn.Sequential(
            nn.Conv1d(64,32,1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32,32,1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32,64,3,padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        testinput  = torch.zeros(1,1,window)
        testoutput = self.resblock2(self.resblock(self.clf(testinput)))

        self.fc = nn.Sequential(
            nn.Linear(testoutput.shape[1]*testoutput.shape[2],512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512,10)
        )
    def forward(self, x):
        out = self.clf(x.view(x.shape[0],1,x.shape[1]))
        out = out + self.resblock(out)
        out = out + self.resblock2(out)
        out = self.fc(out.view(x.shape[0],-1))
        return out

if __name__ == '__main__':
    dataset = MayaDataset('logs', minpower=25, maxpower=225, window=450)
    #dataset = MayaDataset('maya_logs', minpower=25, maxpower=225, window=1100)
    trainlen = (dataset.__len__()*3)//4
    vallen = dataset.__len__() - trainlen
    dsets = random_split(dataset, [trainlen,vallen])
    trainset = dsets[0]
    trainloader = DataLoader(trainset, batch_size=200, num_workers=8)
    
    valset = dsets[1]
    valloader = DataLoader(valset, batch_size=200, num_workers=8)
    
    clf = nn.Sequential(
        nn.Linear(dataset.window,1024),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(1024,1024),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(1024,10)
    ).cuda()
    
    
    clf = CNNCLF(dataset.window).cuda()
    
    optim_c = torch.optim.Adam(clf.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    moving_avg = 0.0
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
        moving_avg = moving_avg*0.9 + macc*0.1
        if e == 0:
            moving_avg = macc
        print("Epoch {} \t acc {:.4f}\tmoving avg: {:.4f}".format(e+1, macc, moving_avg))

        
    
