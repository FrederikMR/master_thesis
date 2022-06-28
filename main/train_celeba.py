# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 23:04:19 2021

@author: Frederik
"""

#%% Sources

"""
Sources:
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

#%% Modules

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import argparse
import datetime

from scipy import ndimage

#Own files
from VAE_celeba import VAE_CELEBA

#%% Parser for command line arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--celeba_path', default="../Data",
                        type=str)
    parser.add_argument('--save_model_path', default='trained_models/celeba/celeba', #'trained_models/surface_R2'
                        type=str)
    
    #Cropping image
    parser.add_argument('--img_size', default=64,
                        type=int)
    parser.add_argument('--num_img', default=0.8, #0.8
                        type=float)

    #Hyper-parameters
    parser.add_argument('--device', default='cpu', #'cuda:0'
                        type=str)
    parser.add_argument('--workers', default=0, #2
                        type=int)
    parser.add_argument('--epochs', default=50000, #50000
                        type=int)
    parser.add_argument('--batch_size', default=100,
                        type=int)
    parser.add_argument('--lr', default=0.0002,
                        type=float)
    parser.add_argument('--save_hours', default=1,
                        type=float)

    #Continue training or not
    parser.add_argument('--con_training', default=0,
                        type=int)
    parser.add_argument('--load_model_path', default='trained_models/main/celeba_epoch_5000.pt',
                        type=str)


    args = parser.parse_args()
    return args

#%% Main loop

def main():
    
    args = parse_args()
    
    img_size = 32
    transform=transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_data = dset.ImageFolder('celeba/',
                                      transform = transform)
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=False)

    images, _ = next(iter(dataloader))
    
    img_base = np.transpose(np.asarray(images[0]), axes=(1,2,0))

    num_rotate = 100
    theta = np.linspace(0,2*np.pi,num_rotate)

    theta_degrees = theta*180/np.pi

    rot = []
    for v in theta_degrees:
        rot.append(ndimage.rotate(img_base, v, reshape=False))
    rot = np.stack(rot)
    
    rot = np.einsum('ijkm->imjk', rot)
    
    DATA = torch.Tensor(rot).to(args.device)
    
    if args.device == 'cpu':
        trainloader = DataLoader(dataset = DATA, batch_size= args.batch_size,
                                 shuffle = True, pin_memory=True, num_workers = args.workers)
    else:
        trainloader = DataLoader(dataset = DATA, batch_size= args.batch_size,
                                 shuffle = True)

    train_loss_elbo = [] #Elbo loss
    train_loss_rec = [] #Reconstruction loss
    train_loss_kld = [] #KLD loss
    epochs = args.epochs
    time_diff = datetime.timedelta(hours=args.save_hours)
    start_time = datetime.datetime.now()
    current_time = start_time

    N = len(trainloader.dataset)

    model = VAE_CELEBA().to(args.device) #Model used

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.con_training:
        checkpoint = torch.load(args.load_model_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        elbo = checkpoint['ELBO']
        rec_loss = checkpoint['rec_loss']
        kld_loss = checkpoint['KLD']

        train_loss_elbo = elbo
        train_loss_rec = rec_loss
        train_loss_kld = kld_loss
    else:
        last_epoch = 0

    model.train()
    for epoch in range(last_epoch, epochs):
        running_loss_elbo = 0.0
        running_loss_rec = 0.0
        running_loss_kld = 0.0
        for x in trainloader:
            #x = x.to(args.device) #If DATA is not saved to device
            #dat = x[0].to(args.device)
            _, x_hat, mu, var, kld, rec_loss, elbo = model(x)
            #optimizer.zero_grad(set_to_none=True) #Based on performance tuning
            optimizer.zero_grad()
            elbo.backward()
            optimizer.step()

            running_loss_elbo += elbo.item()
            running_loss_rec += rec_loss.item()
            running_loss_kld += kld.item()

            #del x, x_hat, mu, var, kld, rec_loss, elbo #In case you run out of memory

        train_epoch_loss = running_loss_elbo/N
        train_loss_elbo.append(train_epoch_loss)
        train_loss_rec.append(running_loss_rec/N)
        train_loss_kld.append(running_loss_kld/N)
        
        if epoch % 1000 == 0:
            print(epoch)
            print(elbo)
        
        current_time = datetime.datetime.now()
        if current_time - start_time >= time_diff:
            print(f"Epoch {epoch+1}/{epochs} - loss: {train_epoch_loss:.4f}")
            checkpoint = args.save_model_path+'_epoch_'+str(epoch+1)+'.pt'
            torch.save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ELBO': train_loss_elbo,
                'rec_loss': train_loss_rec,
                'KLD': train_loss_kld
                }, checkpoint)
            start_time = current_time


    checkpoint = args.save_model_path+'_epoch_'+str(epoch+1)+'.pt'
    torch.save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ELBO': train_loss_elbo,
                'rec_loss': train_loss_rec,
                'KLD': train_loss_kld
                }, checkpoint)

    return

#%% Calling main

if __name__ == '__main__':
    main()
