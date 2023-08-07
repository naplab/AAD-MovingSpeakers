import argparse
import yaml
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data_utils import SeparationDataset
from src.loss_utils import permutation_loss, snr_loss
from models import SeparationNet


def _parse_args():
    
    parser = argparse.ArgumentParser(description='sep')

    parser.add_argument('--batch-size', type=int, default=15,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--config', default='./configs/config_separation.yaml', type=str,
                        help='model config')
    parser.add_argument('--seed', type=int, default=20230222,
                        help='random seed')
    parser.add_argument('--training-file-path', default='./', type=str,
                        help='training file path')
    parser.add_argument('--validation-file-path', default='./', type=str,
                        help='validation file path')
    parser.add_argument('--checkpoint-path', type=str,  default='./',
                        help='path to save the model')
    
    args, _ = parser.parse_known_args()
    
    return args


def _load_config(path):
    with open(path, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    return config


def _save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")
    
    
def train(train_loader, validation_loader, model, optimizer, scheduler, summary_writer, args):
   
    for epoch in range(1, args.epochs + 1):

        model.train()
        train_loss = 0.

        for batch_idx, data in enumerate(train_loader):
            batch_s1 = Variable(data[0]).contiguous()
            batch_s2 = Variable(data[1]).contiguous()
            batch_noise = Variable(data[2]).contiguous()

            if args.cuda:
                batch_s1 = batch_s1.cuda()
                batch_s2 = batch_s2.cuda()
                batch_noise = batch_noise.cuda()

            batch_mix = batch_s1+batch_s2+batch_noise

            batch_left_clean = torch.stack([batch_s1[:,0,:], batch_s2[:,0,:]], dim=1)
            batch_right_clean = torch.stack([batch_s1[:,1,:], batch_s2[:,1,:]], dim=1)

            optimizer.zero_grad()
            left_output, right_output = model(batch_mix)

            loss = torch.mean(permutation_loss(batch_left_clean.unbind(dim=1), batch_right_clean.unbind(dim=1),
                                               left_output.unbind(dim=1), right_output.unbind(dim=1),
                                               snr_loss))    
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            train_loss += loss.data.item() 
            optimizer.step()
            
        train_loss /= (batch_idx+1)
        
        print(epoch, train_loss)
        
        summary_writer.add_scalar("training/train_loss", train_loss, epoch)
        
        checkpoint_path = "{}/epoch_{:03d}".format(args.checkpoint_path, epoch)
        
        _save_checkpoint(
            checkpoint_path,
            {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        })
        
        
        model.eval()
        validation_loss = 0.
        
        for batch_idx, data in enumerate(validation_loader):
            batch_s1 = Variable(data[0]).contiguous()
            batch_s2 = Variable(data[1]).contiguous()
            batch_noise = Variable(data[2]).contiguous()

            if args.cuda:
                batch_s1 = batch_s1.cuda()
                batch_s2 = batch_s2.cuda()
                batch_noise = batch_noise.cuda()
        
            batch_mix = batch_s1+batch_s2+batch_noise
            batch_left_clean = torch.stack([batch_s1[:,0,:], batch_s2[:,0,:]], dim=1)
            batch_right_clean = torch.stack([batch_s1[:,1,:], batch_s2[:,1,:]], dim=1)

            with torch.no_grad():
                left_output, right_output = model(batch_mix)
                loss = torch.mean(permutation_loss(batch_left_clean.unbind(dim=1), batch_right_clean.unbind(dim=1),
                                                   left_output.unbind(dim=1), right_output.unbind(dim=1),
                                                   snr_loss)) 

                validation_loss += loss.data.item() 
                
        validation_loss /= (batch_idx+1)
        summary_writer.add_scalar("training/val_loss", validation_loss, epoch)
        
        if epoch % 2 == 0:
            scheduler.step()

            
def main(args):
    
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        kwargs = {'num_workers': 4, 'pin_memory': True} 
    else:
        kwargs = {}

    random.seed(args.seed)
    torch.manual_seed(args.seed)
        
    train_loader = DataLoader(SeparationDataset(args.training_file_path), 
                          batch_size=args.batch_size, 
                          shuffle=True, 
                          **kwargs)
    validation_loader = DataLoader(SeparationDataset(args.validation_file_path), 
                                   batch_size=args.batch_size, 
                                   shuffle=False, 
                                   **kwargs)
    
    config = _load_config(args.config) 
    
    model = SeparationNet(
        enc_dim = config['enc_dim'],
        feature_dim = config['feature_dim'],
        hidden_dim = config['hidden_dim'],
        enc_win = config['enc_win'],
        enc_stride = config['enc_stride'],
        num_block = config['num_block'],
        num_layer = config['num_layer'],
        kernel_size = config['kernel_size'],
        stft_win = config['stft_win'],
        stft_hop = config['stft_hop'],
        num_spk = config['num_spk'],
    )
    
    if args.cuda:
        model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler  = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    sw = SummaryWriter(os.path.join(args.checkpoint_path, 'logs'))
    
    train(train_loader, validation_loader, model, optimizer, scheduler, sw, args)

            
if __name__ == '__main__':
    main(_parse_args())