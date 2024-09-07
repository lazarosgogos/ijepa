# Author: Lazaros Gogos
# 2024 - 07 - 15
#
# Linear probing on pretrained models based on the I-JEPA architecture

from src import helper
from src.utils.logging import CSVLogger

import os
import argparse
import pprint
import yaml
import logging

from datetime import timedelta
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import torchvision



parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='cls_configs/clsin100.yaml'
)
VIT_EMBED_DIMS = {
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_huge': 1280,
    'vit_giant': 1408,
}



class FeatureExtractor(nn.Module):
    def __init__(self, encoder):
        super(Both, self).__init__()
        self.encoder = encoder
        # Freeze encoder so that it is not trained
        for param in self.encoder.parameters():
            param.requires_grad = False # do ONLY linear probing

        # self.head = LinearClassifier(EMBED_DIMS, num_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        return x

class LinearClassifier(nn.Module):
    """ Create a single fully connected layer for classification"""
    def __init__(self, input_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.linear = nn.Linear(input_size, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.1)
        self.linear.bias.data.zero_()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # flatten 
        x = torch.mean(x, dim=1, dtype=x.dtype)

        # linear layer
        return self.softmax(self.linear(x))

class Both(nn.Module):
    def __init__(self, encoder, EMBED_DIMS, num_classes):
        super(Both, self).__init__()
        self.encoder = encoder
        # Freeze encoder so that it is not trained
        for param in self.encoder.parameters():
            param.requires_grad = False # do ONLY linear probing

        self.head = LinearClassifier(EMBED_DIMS, num_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x

class LinearProbe():
    def __init__(self, args, logger):
        # init LinClassifier
        # init complete model (pretrained+head)
        # init criterion for loss
        # ----------------------------------------------------------------------- #
        #  PASSED IN PARAMS FROM CONFIG FILE
        # ----------------------------------------------------------------------- #

        # -- DATA
        self.crop_size = args['data']['crop_size']
        self.num_classes = args['data']['num_classes']
        self.train_dataset_path = args['data']['train_dataset_path']
        self.val_dataset_path = args['data']['val_dataset_path']
        self.model_name = args['data']['model_name']
        self.batch_size = args['data']['batch_size']
        self.patch_size = args['data']['patch_size']
        

        # -- LOGGING
        self.log_dir = args['logging']['log_dir']
        self.pretrained_model_path = args['logging']['pretrained_model_path']
        self.save_path = args['logging']['save_path']
        self.checkpoint_freq = args['logging']['checkpoint_freq']
        self.log_file = args['logging']['log_file']

        self.pretrained_model_path = os.path.join(self.log_dir, self.pretrained_model_path)
        self.save_path = os.path.join(self.log_dir, self.save_path)
        self.log_file = os.path.join(self.log_dir, f'{self.log_file}.csv')


        # -- OPTIMIZATION
        self.lr = args['optimization']['lr']
        self.epochs = args['optimization']['epochs']
        self.embed_dims = VIT_EMBED_DIMS[self.model_name] # get dims based on model

        # -- META
        self.device_name = args['meta']['device']

        self.device = torch.device(self.device_name if torch.cuda.is_available() else 'cpu')

        self.encoder = helper.init_encoder(device=self.device, 
                                        patch_size=self.patch_size,
                                        model_name=self.model_name,
                                        crop_size=self.crop_size,)


        ckpt = torch.load(self.pretrained_model_path, map_location=torch.device('cpu'))
        pretrained_dict = ckpt['encoder']

        # -- loading encoder
        for k, v in pretrained_dict.items():
            self.encoder.state_dict()[k[len('module.'):]].copy_(v) 

        self.model = Both(self.encoder, self.embed_dims, self.num_classes)
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optim = optim.AdamW(self.model.parameters(), lr=self.lr)

        self.transform = transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

        self.train_dataset = ImageFolder(root=self.train_dataset_path, transform=self.transform)
        self.val_dataset = ImageFolder(root=self.val_dataset_path, transform=self.transform)
        
        # run feature extractor here
        feature_extractor = FeatureExtractor(self.encoder)
        logger.info('Extracting features and saving them locally..')
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)
        for inputs, labels in self.train_loader:
            pass

        self.train_features_path = ''
        self.val_features_path = ''
        
        self.train_dataset = torchvision.datasets.DatasetFolder(root=self.train_features_path)
        self.val_dataset = torchvision.datasets.DatasetFolder(root=self.val_features_path)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)
        self.logger = logger
        self.csvlogger = CSVLogger(self.log_file, 
                                   ('%d', 'epoch'),
                                   ('%.5e', 'train_accuracy'),
                                   ('%.5e', 'val_accuracy'),
                                   ('%.5e', 'loss'),
                                   ('%s', 'time'))

    def save_checkpoint(self, epoch):
        '''Save a checkpoint of a given model & an optimizer. 
        Every `checkpoint_freq` epochs save the model in a different file as well for post-use'''
        save_dict = {
            'model': self.model.state_dict(),
            'opt': self.optim.state_dict(),
            'epoch': epoch,
        }
        save_path = self.save_path
        ep = epoch + 1 # temp epoch to avoid alchemy with string formats :)
        torch.save(save_dict, save_path+'-latest.pth.tar')
        save_path = save_path + f'-ep{ep}.pth.tar'
        if (ep) % self.checkpoint_freq == 0:
            torch.save(save_dict, save_path)

        
    # create function for saving model
    def eval_linear(self):
        """ The main function in which linear probing is implemented"""
        start_time = time.perf_counter()
        for epoch in range(self.epochs):
            epoch_start_time = time.perf_counter()
            self.model.train() # set model to training mode
            running_loss = 0.0
            train_correct = 0
            total_train = 0
            for inputs, labels in self.train_loader:
                # send data to appropriate device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optim.zero_grad() 
                outputs = self.model(inputs)

                _, predicted = outputs.max(dim=1)
                train_correct += (predicted == labels).sum().item()
                total_train += labels.size(0)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optim.step()
                running_loss += loss.item()
            
            train_accuracy = train_correct / total_train

            epoch_loss = running_loss / total_train

            self.model.eval() # set to evaluation mode
            val_correct = 0
            total_val = 0
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device),\
                                        labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = outputs.max(dim=1)
                    total_val += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            time_taken = time.perf_counter() - epoch_start_time
            # duration = timedelta(seconds=time_taken)
            duration = time_taken
            val_accuracy = val_correct / total_val

            """self.logger.info('Epoch: %d/%d' % (epoch+1, self.epochs) 
                            'Train accuracy: %.5e' % train_accuracy 
                            # 'Train correct: %d' % train_correct,
                            # 'Train total: %d' % len(train_dataset),
                            'Validation accuracy: %.5e' % val_accuracy 
                            # 'Val correct: %d' % val_correct,
                            # 'Val total: %d' % len(val_dataset),
                            'Loss %.5e' % epoch_loss 
                            'Time taken:', duration)"""
            self.logger.info('Epoch: %d/%d '
                            'Train accuracy: %.5e ' 
                            'Validation accuracy: %.5e '
                            'Loss %.5e '
                            'Time taken: %d seconds'
                             % (epoch+1, self.epochs,
                                train_accuracy,
                                val_accuracy,
                                epoch_loss,
                                int(duration)) )
            self.csvlogger.log(epoch+1, 
                               train_accuracy, 
                               val_accuracy, 
                               epoch_loss, 
                               duration)
            # save checkpoint after epoch
            self.save_checkpoint(epoch+1)
        
        # report on time after all epochs are complete
        end_time = time.perf_counter()
        total_duration = timedelta(seconds=end_time-start_time)
        self.logger.info('Total time taken %s' % str(total_duration))
        self.logger.info('Done')

    

def process_main(fname, devices=['cuda:0']):
    """ This function was inspired by main.py from IJEPA"""
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info(f'called-params {fname}')

    # load script params
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params....')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
    
    
    linear_prober = LinearProbe(params, logger)

    linear_prober.eval_linear()


if __name__ == '__main__':
    """ No support for distributed training as of yet.
    Start linear probing based on config"""
    args = parser.parse_args() # get arguments from cmdline
    process_main(args.fname) 