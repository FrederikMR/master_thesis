# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 22:06:08 2021

@author: Frederik
"""

#%% Sources

"""
Sources:
https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
http://adamlineberry.ai/vae-series/vae-code-experiments
https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa
https://discuss.pytorch.org/t/cpu-ram-usage-increasing-for-every-epoch/24475/10
https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
"""

#%% Modules

#Loading own module from parent folder
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.realpath(currentdir))
parentdir = os.path.dirname(os.path.realpath(parentdir))
sys.path.append(parentdir)

#Modules
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import pandas as pd

#Own files
from AE_surface3d import AE_3d

#%% Parser for command line arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--data_name', default='paraboloid', 
                        type=str)
    parser.add_argument('--save_step', default=100000,
                        type=int)

    #Hyper-parameters
    parser.add_argument('--device', default='cpu', #'cuda:0'
                        type=str)
    parser.add_argument('--epochs', default=100000, #100000
                        type=int)
    parser.add_argument('--batch_size', default=100,
                        type=int)
    parser.add_argument('--lr', default=0.002,
                        type=float)
    parser.add_argument('--workers', default=0,
                        type=int)

    #Continue training or not
    parser.add_argument('--con_training', default=0,
                        type=int)
    parser.add_argument('--load_epoch', default='100000.pt',
                        type=str)

    args = parser.parse_args()
    return args

#%% Main loop

def main():

    args = parse_args()
    train_loss_rec = [] #Reconstruction loss
    epochs = args.epochs
    
    #paths
    data_path = 'sim_data/'+args.data_name+'.csv'
    load_path = 'trained_models/AE_'+args.data_name+'_epoch_'+args.load_epoch
    save_path = 'trained_models/AE_'+args.data_name+'_epoch_'

    df = pd.read_csv(data_path, index_col=0)
    DATA = torch.Tensor(df.values).to(args.device) #DATA = torch.Tensor(df.values)
    DATA = torch.transpose(DATA, 0, 1)

    if args.device == 'cpu':
        trainloader = DataLoader(dataset = DATA, batch_size= args.batch_size,
                                 shuffle = True, pin_memory=True, num_workers = args.workers)
    else:
        trainloader = DataLoader(dataset = DATA, batch_size= args.batch_size,
                                 shuffle = True)
        
    N = len(trainloader.dataset)

    model = AE_3d().to(args.device) #Model used

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.con_training:
        checkpoint = torch.load(load_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        rec_loss = checkpoint['rec_loss']
        
        train_loss_rec = rec_loss
    else:
        last_epoch = 0

    model.train()
    for epoch in range(last_epoch, epochs):
        running_loss_rec = 0.0
        for x in trainloader:
            #x = x.to(args.device) #If DATA is not saved to device
            _, x_hat, rec_loss = model(x)
            optimizer.zero_grad() #Based on performance tuning
            rec_loss.backward()
            optimizer.step()

            running_loss_rec += rec_loss.item()

            #del x, x_hat, mu, var, kld, rec_loss, elbo #In case you run out of memory

        train_epoch_loss = running_loss_rec/N
        train_loss_rec.append(train_epoch_loss)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch+1}/{epochs} - loss: {train_epoch_loss:.4f}")


        if (epoch+1) % args.save_step == 0:
            checkpoint = save_path+str(epoch+1)+'.pt'
            torch.save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rec_loss': train_loss_rec,
                }, checkpoint)


    checkpoint = checkpoint = save_path+str(epoch+1)+'.pt'
    torch.save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rec_loss': train_loss_rec,
                }, checkpoint)

    return

#%% Calling main

if __name__ == '__main__':
    main()
