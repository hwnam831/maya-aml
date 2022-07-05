
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from MayaDataset import CNNCLF
import MayaDataset
import argparse
import time
import os
from Models import RNNGenerator, RNNGenerator2, Discriminator



def get_parser():
    """Get all the args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--net",
            type=str,
            choices=['cnn', 'rnn', 'ff'],
            default='cnn',
            help='Classifier choices')
    parser.add_argument(
            "--gen",
            type=str,
            choices=['rnn', 'rnn2'],
            default='rnn2',
            help='Generator choices')
    parser.add_argument(
            "--window",
            type=int,
            default='500',
            help='number of samples window')
    parser.add_argument(
            "--epochs",
            type=int,
            default='200',
            help='number of epochs')
    parser.add_argument(
            "--warmup",
            type=int,
            default='10',
            help='number of warmup epochs')
    parser.add_argument(
            "--cooldown",
            type=int,
            default='100',
            help='number of cooldown epochs')
    parser.add_argument(
            "--batch_size",
            type=int,
            default='32',
            help='batch size')
    parser.add_argument(
            "--dim",
            type=int,
            default='64',
            help='internal channel dimension')
    parser.add_argument(
            "--lr",
            type=float,
            default=1e-4,
            help='Default learning rate')
    parser.add_argument(
            "--student",
            type=int,
            default=16,
            help='Student dim')
    parser.add_argument(
            "--hinge",
            type=float,
            default='0.2',
            help='noise amp scale')
    parser.add_argument(
            "--gamma",
            type=float,
            default='0.98',
            help='decay scale for optimizer')
    parser.add_argument(
            "--lambda_h",
            type=float,
            default='5.0',
            help='lambda coef for hinge loss')
    parser.add_argument(
            "--lambda_d",
            type=float,
            default='0.01',
            help='lambda coef for discriminator loss')   
    parser.add_argument(
            "--lambda_r",
            type=float,
            default='0.0005',
            help='lambda coef for reconstruction loss')     
    parser.add_argument(
            "--fresh",
            action='store_true',
            help='Fresh start without loading')

    return parser

def Warmup(clf, clf_v, disc, gen, wepoch, lr, trainloader, valloader):
    optim_c = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-5)
    optim_c_v = torch.optim.Adam(clf_v.parameters(), lr=lr, weight_decay=1e-5)
    optim_d = torch.optim.Adam(disc.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    for e in range(wepoch):
        clf.train()
        clf_v.train()
        for x,y in trainloader:
            optim_c.zero_grad()
            optim_d.zero_grad()
            optim_c_v.zero_grad()
            xdata, ydata = x.cuda(), y.cuda()
            perturb = gen(xdata)
            p_input = torch.relu(xdata + perturb.detach())
            output = clf(p_input)
            disc_outputs = disc(p_input)
            disc_labels = torch.ones_like(disc_outputs)*0.1
            for dl,yi in zip(disc_labels, ydata):
                dl[yi] = dl[yi] - 1 # (0.9, -0.1, -0.1, ...)
            loss = criterion(output, ydata)
            loss.backward()
            loss_d = torch.mean(disc_outputs*disc_labels)
            loss_d.backward()
            optim_d.step()
            disc.clip()
            
            output_v = clf_v(p_input)
            loss_v = criterion(output_v, ydata)
            loss_v.backward()
            optim_c.step()
            optim_c_v.step()
            
        totcorrect = 0
        totcount = 0
        clf.eval()
        totdistance = 0.0
        for x,y in valloader:
            xdata, ydata = x.cuda(), y.cuda()
            p_input = torch.relu(xdata + gen(xdata).detach())
            output = clf(p_input)
            disc_outputs = disc(p_input)
            disc_labels = torch.ones_like(disc_outputs)*0.1
            for dl,yi in zip(disc_labels, ydata):
                dl[yi] = dl[yi] - 1 # (-0.9, 0.1, 0.1, ...)
            totdistance += -torch.mean(disc_outputs*disc_labels)
            pred = output.argmax(axis=-1)
            totcorrect += (pred==ydata).sum().item()
            totcount += y.size(0)
        macc = float(totcorrect)/totcount
        mdist = totdistance/len(valloader)
        print("Warmup {}\t acc {:.4f}\t disc {:.4f}".format(e+1, macc, mdist))
    return clf

if __name__ == '__main__':
    args = get_parser().parse_args()
    dataset = MayaDataset.MayaDataset('logs', minpower=25, maxpower=225, window=args.window)
    
    dsets = random_split(dataset, [7000,1500, 1500])
    trainset = dsets[0]
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=4)
    
    valset = dsets[1]
    valloader = DataLoader(valset, batch_size=args.batch_size, num_workers=4)

    testset = dsets[2]
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=4)

    clf = CNNCLF(dataset.window).cuda()
    clf_v = CNNCLF(dataset.window).cuda()
    disc = Discriminator(512, 10, dataset.window).cuda()
    if args.gen == 'rnn':
        gen = RNNGenerator(args.dim, minpower=dataset.minpower, maxpower=dataset.maxpower).cuda()
    elif args.gen == 'rnn2':
        gen = RNNGenerator2(args.dim, minpower=dataset.minpower, maxpower=dataset.maxpower).cuda()

    if os.path.isfile('./best_{}_{}.pth'.format(args.gen, args.dim)) and not args.fresh:
        print('Previous best found: loading the model...')
        gen.load_state_dict(torch.load('./best_{}_{}.pth'.format(args.gen, args.dim)))

    clf = Warmup(clf, clf_v, disc, gen, args.warmup, args.lr, valloader, testloader)

    optim_c = torch.optim.Adam(clf.parameters(), lr=2*args.lr, weight_decay=args.lr)
    optim_c_v = torch.optim.Adam(clf_v.parameters(), lr=2*args.lr, weight_decay=args.lr)
    optim_d = torch.optim.Adam(disc.parameters(), lr=2*args.lr, weight_decay=args.lr)
    optim_g = torch.optim.Adam(gen.parameters(), lr=args.lr, weight_decay=args.lr)
    sched_c   = torch.optim.lr_scheduler.StepLR(optim_c, 1, gamma=args.gamma)
    sched_c_v   = torch.optim.lr_scheduler.StepLR(optim_c_v, 1, gamma=args.gamma)
    sched_d = torch.optim.lr_scheduler.StepLR(optim_d, 1, gamma=args.gamma)
    sched_g   = torch.optim.lr_scheduler.StepLR(optim_g, 1, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()
    kldiv = nn.KLDivLoss(reduce=True,reduction='batchmean')
    bestdict = gen.state_dict()
    bestacc = 1.0
    for e in range(args.epochs):
        gen.train()
        clf.train()
        clf_v.train()

        trainstart = time.time()
        for x,y in trainloader:
            xdata, ydata = x.cuda(), y.cuda()

            #train classifier
            optim_c.zero_grad()
            optim_d.zero_grad()
            perturb = gen(xdata)
            p_input = torch.relu(xdata + perturb.detach())

            #interleaving?
            output = clf(p_input)
            disc_outputs = disc(p_input)
            disc_labels = torch.ones_like(disc_outputs)*0.1
            for dl,yi in zip(disc_labels, ydata):
                dl[yi] = dl[yi] - 1 # (0.9, -0.1, -0.1, ...)
            loss_c = criterion(output, ydata)
            loss_c.backward()
            loss_d = torch.mean(disc_outputs*disc_labels)
            loss_d.backward()
            optim_c.step()
            optim_d.step()
            disc.clip()

            #train generator
            optim_g.zero_grad()
            p_input = torch.relu(xdata + perturb)
            output = clf(p_input)
            
            #hinge = perturb.mean(dim=-1) - args.amp
            #hinge[hinge<0] = 0.0
            norm = torch.linalg.norm(perturb, dim=-1)/np.sqrt(p_input.size(-1))
            loss_p = torch.mean(torch.relu(norm-args.hinge))
            #loss_p = torch.mean(norm)
            fake_target = torch.ones_like(output)/output.shape[-1]
            loss_adv1 = kldiv(F.log_softmax(output,dim=-1), fake_target)
            disc_outputs = disc(p_input)
            disc_labels = torch.ones_like(disc_outputs)*0.1
            for dl,yi in zip(disc_labels, ydata):
                dl[yi] = dl[yi] - 1 # (-0.9, 0.1, 0.1, ...)
            loss_d2 = torch.mean(-disc_outputs*disc_labels)
            #fake_target = torch.zeros_like(ydata)
            #loss_adv1 = criterion(output, fake_target)
            loss = loss_adv1 + args.lambda_h*loss_p + args.lambda_d*loss_d2


            loss.backward()
            optim_g.step()
        #gen.eval()
        for x,y in valloader:
            xdata, ydata = x.cuda(), y.cuda()

            #train classifier_v
            #train classifier
            optim_c.zero_grad()
            optim_d.zero_grad()
            optim_c_v.zero_grad()
            perturb = gen(xdata)
            p_input = torch.relu(xdata + perturb.detach())

            #interleaving?
            output = clf(p_input)
            output_v = clf_v(p_input)
            disc_outputs = disc(p_input)
            disc_labels = torch.ones_like(disc_outputs)*0.1
            for dl,yi in zip(disc_labels, ydata):
                dl[yi] = dl[yi] - 1 # (-0.9, 0.1, 0.1, ...)
            loss_d = torch.mean(disc_outputs*disc_labels)
            loss_d.backward()
            loss_c = criterion(output, ydata)
            loss_c.backward()
            loss_c_v = criterion(output_v, ydata)
            loss_c_v.backward()
            optim_c.step()
            optim_c_v.step()
            optim_d.step()
            disc.clip()

        mloss = 0.0
        mloss2 = 0.0
        totcorrect = 0
        totcount = 0
        mnorm = 0.0
        orgpower = 0.0
        newpower = 0.0
        totdist = 0.0
        #evaluate classifier
        with torch.no_grad():
            clf_v.eval()
            gen.eval()
            for x,y in testloader:
                xdata, ydata = x.cuda(), y.cuda()
                perturb = gen(xdata)
                norm = torch.linalg.norm((perturb), dim=-1)/np.sqrt(xdata.size(-1))
                p_input = torch.relu(xdata+perturb.detach())
                output = clf_v(p_input)
                
                loss_c = criterion(output, ydata)
                pred = output.argmax(axis=-1)
                newpower += p_input.mean()
                mnorm += norm.mean().item()/len(testloader)
                mloss += loss_c.item()/len(testloader)

                totcorrect += (pred==ydata).sum().item()
                totcount += y.size(0)
                disc_outputs = disc(p_input)
                disc_labels = torch.ones_like(disc_outputs)*0.1
                for dl,yi in zip(disc_labels, ydata):
                    dl[yi] = dl[yi] - 1 # (0.9, -0.1, -0.1, ...)
                totdist += torch.mean(disc_outputs*disc_labels).item()
                
            mdist = totdist/len(testloader)
            macc = float(totcorrect)/totcount
            #mnorm = mnorm*(dataset.maxpower-dataset.minpower)
            newpower = newpower*(dataset.maxpower-dataset.minpower)/len(testloader) + dataset.minpower
            if np.abs(macc-0.1) <= np.abs(bestacc-0.1) and e > args.epochs//2:
                bestacc = macc
                bestdict = gen.state_dict()
                bestnorm = mnorm
            print("E {}\tacc {:.3f}\tAvg perturb {:.3f}\tAvg pow {:.2f}\t Avg dist {:3f}".format(e+1, macc, mnorm, newpower,mdist))
        sched_c.step()
        sched_c_v.step()
        sched_d.step()
        sched_g.step()
    
    torch.save(bestdict, "{}_{}_{:.3f}_{:.3f}.pth".format(args.gen, args.dim,bestacc,bestnorm))