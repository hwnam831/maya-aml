import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from copy import deepcopy

class RNNGenerator(nn.Module):
    def __init__(self, dim, window=32, minpower=25.0, maxpower=225.0, noise=0.05):
        super().__init__()
        self.window = window
        self.minpower = minpower
        self.maxpower = maxpower
        self.noise = noise
        self.encoder = nn.Sequential(
            nn.Conv1d(1,dim,window,1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.resblock = nn.GRU(dim,dim, num_layers=1, batch_first=True)

        self.decoder = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Hardtanh()
        )

    def forward(self, x, distill=False):
        padded = F.pad(x,(self.window,0))
        encoded = self.encoder(padded.view(padded.shape[0],1,padded.shape[1])).permute(0,2,1)
        res = encoded + self.resblock(encoded)[0]
        out = self.decoder(res).view(x.shape[0],x.shape[1]+1)[:,:-1]
        if self.training == False:
            out = out + self.noise*torch.rand_like(out) #[-1~7]
        #out = F.relu(out+x)-x #[0,2]
        #out = torch.minimum(out, torch.ones_like(out))
        
        if distill:
            return (encoded, res, out)
        else: 
            return out

class NoiseInjector(nn.Module):
    def __init__(self, noiselevel=0.1):
        super().__init__()
        self.noiselevel = nn.Parameter(torch.ones(1)*noiselevel)

    def forward(self, x):
        return x + torch.randn_like(x)*self.noiselevel
        

class RNNGenerator2(nn.Module):
    def __init__(self, dim, window=32, minpower=25.0, maxpower=225.0, noise=0.05):
        super().__init__()
        self.window = window
        self.minpower = minpower
        self.maxpower = maxpower
        self.noise = noise
        self.encoder = nn.Sequential(
            NoiseInjector(noise),
            nn.Conv1d(1,dim,window,1),
            nn.ReLU(),
            NoiseInjector(noise),
        )

        self.resblock = nn.GRU(dim,dim, num_layers=1, batch_first=True)

        self.decoder = nn.Sequential(
            NoiseInjector(noise),
            nn.Linear(dim, 1),
            nn.Hardtanh()
        )

    def forward(self, x, distill=False):
        padded = F.pad(x,(self.window,0))
        encoded = self.encoder(padded.view(padded.shape[0],1,padded.shape[1])).permute(0,2,1)
        res = encoded + self.resblock(encoded)[0]
        out = self.decoder(res).view(x.shape[0],x.shape[1]+1)[:,:-1]
        
        if distill:
            return (encoded, res, out)
        else: 
            return out

#3:1 generator
class RNNGenerator3(nn.Module):
    def __init__(self, dim, window=32, minpower=25.0, maxpower=225.0, noise=0.1):
        super().__init__()
        self.window = window
        self.minpower = minpower
        self.maxpower = maxpower
        self.noise = noise
        self.encoder = nn.Sequential(
            NoiseInjector(noise),
            nn.Conv1d(1,dim,window,stride=3),
            nn.ReLU(),
            NoiseInjector(noise),
        )

        self.resblock = nn.GRU(dim,dim, num_layers=1, batch_first=True)

        self.decoder = nn.Sequential(
            NoiseInjector(noise),
            nn.Linear(dim, 1),
            nn.Hardsigmoid()
        )
    #S,N,C
    def forward(self, x, distill=False):
        #Padding so that each output only observes the history
        padded = F.pad(x,(self.window,0))
        encoded = self.encoder(padded.view(padded.shape[0],1,padded.shape[1])).permute(0,2,1)
        res = encoded + self.resblock(encoded)[0]
        out = self.decoder(res).view(x.shape[0],-1)[:,:-1]
        
        if distill:
            return (encoded, res, out)
        else: 
            return out

class RNNInference(nn.Module):
    def __init__(self, dim, window=32, noise=0.05):
        super().__init__()
        self.window = window
        self.noise = noise
        self.dim =dim
        self.encoder = nn.Sequential(
            NoiseInjector(self.noise),
            nn.Linear(self.window,self.dim),
            nn.ReLU(),
            NoiseInjector(self.noise),
        )
        

        self.rnn = nn.GRUCell(self.dim,self.dim)
        

        self.decoder = nn.Sequential(
            NoiseInjector(self.noise),
            nn.Linear(self.dim, 1),
            nn.Hardtanh()
        )
        
        
    def copy_params(self,gen):
        self.encoder[1].weight.data = gen.encoder[1].weight.data.reshape(self.dim, self.window)
        self.encoder[1].bias.data = gen.encoder[1].bias.data
        self.rnn.weight_ih.data = gen.resblock.weight_ih_l0.data
        self.rnn.weight_hh.data = gen.resblock.weight_hh_l0.data
        self.rnn.bias_ih.data = gen.resblock.bias_ih_l0.data
        self.rnn.bias_hh.data = gen.resblock.bias_hh_l0.data
        self.decoder[1].weight.data = gen.decoder[1].weight.data
        self.decoder[1].bias.data = gen.decoder[1].bias.data


    def forward(self, input, hidden):
        encoded = self.encoder(input)
        h_out = self.rnn(encoded, hidden)
        out = encoded + h_out
        out = self.decoder(out)
        
        return out, h_out

class RNNInference3(nn.Module):
    def __init__(self, dim, window=32, noise=0.05):
        super().__init__()
        self.window = window
        self.noise = noise
        self.dim =dim
        self.encoder = nn.Sequential(
            NoiseInjector(self.noise),
            nn.Linear(self.window,self.dim),
            nn.ReLU(),
            NoiseInjector(self.noise),
        )
        

        self.rnn = nn.GRUCell(self.dim,self.dim)
        

        self.decoder = nn.Sequential(
            NoiseInjector(self.noise),
            nn.Linear(self.dim, 1),
            nn.Hardsigmoid()
        )
        
        
    def copy_params(self,gen):
        self.encoder[1].weight.data = gen.encoder[1].weight.data.reshape(self.dim, self.window)
        self.encoder[1].bias.data = gen.encoder[1].bias.data
        self.rnn.weight_ih.data = gen.resblock.weight_ih_l0.data
        self.rnn.weight_hh.data = gen.resblock.weight_hh_l0.data
        self.rnn.bias_ih.data = gen.resblock.bias_ih_l0.data
        self.rnn.bias_hh.data = gen.resblock.bias_hh_l0.data
        self.decoder[1].weight.data = gen.decoder[1].weight.data
        self.decoder[1].bias.data = gen.decoder[1].bias.data


    def forward(self, input, hidden):
        encoded = self.encoder(input)
        h_out = self.rnn(encoded, hidden)
        out = encoded + h_out
        out = self.decoder(out)
        
        return out, h_out

class Discriminator(nn.Module):
    def __init__(self, dim, n_cls, window):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(window, dim),
            nn.ReLU(),
            nn.Linear(dim,dim),
            nn.ReLU(),
            nn.Linear(dim,dim),
            nn.ReLU(),
            nn.Linear(dim,n_cls)
        )
    def forward(self, x):
        return self.layers(x)
    def clip(self, low=-0.01, high=0.01):
        for l in self.layers:
            if isinstance(l, nn.Linear):
                l.weight.data.clamp_(low, high)

class Distiller(nn.Module):
    def __init__(self, tdim=256, sdim=32, lamb_d = 0.1, lamb_r = 0.1):
        super().__init__()
        self.map1 = nn.Linear(sdim, tdim)
        self.map2 = nn.Linear(sdim, tdim)
        self.criterion = nn.MSELoss()
        self.lamb_d = lamb_d
        self.lamb_r = lamb_r

    def forward(self, s_out, t_out):
        enc_s, res_s, out_s = s_out
        enc_s2 = self.map1(enc_s)
        res_s2 = self.map2(res_s)

        enc_t, res_t, out_t = t_out
        
        l_distill = self.criterion(enc_s2, enc_t.detach()) + self.criterion(res_s2, res_t.detach())
        l_recon = self.criterion(out_s, out_t.detach())
        return self.lamb_d*l_distill + self.lamb_r*l_recon


class AttnShaper(nn.Module):
    def __init__(self, dim=64, history=32, window=3, minpower=25.0, maxpower=160.0, amp=2.0, n_patterns=32):
        super().__init__()
        self.history=history
        self.window=window
        self.n_patterns=n_patterns
        self.dim = dim
        self.amp = amp
        self.conv1 = nn.Sequential(
            nn.Conv1d(1,dim,history, stride=window),
            nn.ReLU(),
        )
        self.keys = nn.GRU(dim,self.dim, num_layers=1)
        #self.keys = nn.Sequential(
        #    nn.Linear(dim, self.dim),
        #    nn.ReLU(),
        #    nn.Linear(dim,dim))
        self.offsets = nn.Linear(self.dim, 1)
        self.stds = nn.Linear(self.dim, 1)
        #self.keys=nn.Linear(dim, self.n_patterns*2, bias=False)
        #offsets = (torch.arange(n_patterns,dtype=torch.float32).view(n_patterns,1))/n_patterns
        #self.register_buffer('offsets',offsets,persistent=True)
        #stds = (torch.arange(n_patterns,dtype=torch.float32).view(n_patterns,1))/n_patterns
        #self.register_buffer('stds',stds,persistent=True)
        #self.relu6 = nn.ReLU6()

        #noiselevel = torch.arange(n_patterns,dtype=torch.float32).view(n_patterns,1)/n_patterns
        #self.register_buffer('noiselevel',noiselevel, persistent=False)
    def forward(self, x):
        
        padded = F.pad(x,(self.history-1, 0))
        
        out = self.conv1(padded[:,None,:]).permute(2,0,1) #N,C,S -> S,N,C
        #out = F.dropout(out,0.25)
        attn_scores, _ = self.keys(out)
        #attn_scores = self.keys(out)
        attn_scores = F.relu6(attn_scores).permute(1,0,2)
        #attn_scores = F.dropout(attn_scores, 0.25)
        '''
        probs = []
        for score in attn_scores:
            #prob = torch.softmax(score-avg_scores, dim=-1)
            prob = torch.softmax(score,dim=-1)
            probs.append(prob)
        attn_probs = torch.stack(probs,dim=1).to(dtype=x.dtype) #N,S,C
        '''
        #attn_probs = torch.softmax(attn_scores,dim=-1).permute(1,0,2) #S,N,C->N,S,C
        #offset_probs = attn_probs[:,:,:self.n_patterns]
        #std_probs = attn_probs[:,:,self.n_patterns:]
        #offset = torch.matmul(attn_probs, self.offsets).expand(x.shape[0],attn_probs.shape[1],self.window).reshape(x.shape[0],-1)[:,:x.shape[1]]
        #noise = torch.randn_like(offset)
        #offset = torch.matmul(offset_probs, self.offsets)
        offset = F.hardsigmoid(self.offsets(attn_scores))
        #std = torch.matmul(std_probs, self.stds)
        #std = torch.sigmoid(self.fc(attn_scores[:,:,self.n_patterns:])).permute(1,0,2)
        std = F.hardsigmoid(self.stds(attn_scores))
        noise = torch.randn_like(offset) * offset/2
        signal = (offset+noise).expand(x.shape[0],attn_scores.shape[1],self.window)
        signal = torch.clamp(signal, 0,1)
        signal = signal.reshape(x.shape[0],-1)[:,:x.shape[1]]
        padded_signal = F.pad(signal,(self.window-1, 0))[:,None,:]
        pooled_signal = F.avg_pool1d(padded_signal, self.window,stride=1)
        return pooled_signal.reshape(x.shape[0],-1)-x

class ShaperInference(nn.Module):
    def __init__(self, shaper):
        super().__init__()
        self.history=shaper.history
        self.dim=shaper.dim
        self.window=shaper.window
        self.n_patterns=shaper.n_patterns

        self.fc = nn.Linear(self.history, self.dim)
        self.fc.weight.data = shaper.conv1[0].weight.data.view(self.dim,self.history)

        self.relu6 = nn.ReLU6()
        self.keys = nn.GRUCell(self.dim,self.dim)
        self.keys.weight_ih.data = shaper.keys.weight_ih_l0.data
        self.keys.weight_hh.data = shaper.keys.weight_hh_l0.data
        self.keys.bias_ih.data = shaper.keys.bias_ih_l0.data
        self.keys.bias_hh.data = shaper.keys.bias_hh_l0.data

        self.offsets = deepcopy(shaper.offsets)
        self.stds = deepcopy(shaper.stds)
        
        
    def copy_params(self,gen):
        self.encoder[1].weight.data = gen.encoder[1].weight.data.reshape(self.dim, self.window)
        self.encoder[1].bias.data = gen.encoder[1].bias.data
        
        self.decoder[1].weight.data = gen.decoder[1].weight.data
        self.decoder[1].bias.data = gen.decoder[1].bias.data
    #1,H input, 1,W output
    def forward(self, x, hidden):
        
        out = torch.relu(self.fc(x))
        h_out = self.keys(out, hidden)
        attn_score = self.relu6(h_out)
        offset = F.hardsigmoid(self.offsets(attn_score))
        #std = F.hardsigmoid(self.stds(attn_score))
        noise = torch.randn_like(offset) * offset/2
        signal = torch.clamp(offset+noise,0,1)
        

        return signal, h_out