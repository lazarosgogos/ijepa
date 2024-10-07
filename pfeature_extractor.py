# Author: Lazaros Gogos
# 2024 - 07 - 15
#
# Linear probing on pretrained models based on the I-JEPA architecture

import torch.utils
from src import helper
from src.utils.logging import CSVLoggerAppender

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

import glob
import re

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='cls_configs/clsiic.yaml'
)
VIT_EMBED_DIMS = {
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_huge': 1280,
    'vit_giant': 1408,
}


"""
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
"""
class LinearClassifier(nn.Module):
    """ Create a single fully connected layer for classification"""
    def __init__(self, input_size, num_classes, use_normalization):
        super(LinearClassifier, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.linear = nn.Linear(input_size, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.1)
        self.linear.bias.data.zero_()
        self.softmax = nn.Softmax(dim=1)
        self.use_normalization = use_normalization

        self.head_dropout = nn.Dropout(.2) # try 20% dropout 

    def forward(self, x):
        # flatten 
        x = torch.mean(x, dim=1, dtype=x.dtype)
        if self.use_normalization:
            # add dropout
            x = self.head_dropout(x)

            # add layer norm
            x = F.layer_norm(x, (x.size(-1),)) # do not touch the BATCH SIZE dimension
                                       # but normalize over feature dim
        # linear layer
        return self.softmax(self.linear(x))

class Both(nn.Module):
    def __init__(self, encoder, EMBED_DIMS, num_classes, use_normalization):
        super(Both, self).__init__()
        self.encoder = encoder
        # Freeze encoder so that it is not trained
        for param in self.encoder.parameters():
            param.requires_grad = False # do ONLY linear probing

        self.head = LinearClassifier(EMBED_DIMS, num_classes, use_normalization)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x

class FeaturesDataset(torch.utils.data.Dataset):
    def __init__(self, feature_file_path):
        data = torch.load(feature_file_path)
        self.features = data['features']
        self.labels = data['labels']

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


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
        self.probe_checkpoints = args['data'].get('probe_checkpoints', False) # default to False
        self.probe_prefix = args['data'].get('probe_prefix', None) # default to None


        # -- LOGGING
        self.log_dir = args['logging']['log_dir']
        self.pretrained_model_path = args['logging']['pretrained_model_path']
        self.save_path = args['logging']['save_path']
        self.checkpoint_freq = args['logging']['checkpoint_freq']
        self.log_file = args['logging']['log_file']

        self.pretrained_model_path = os.path.join(self.log_dir, self.pretrained_model_path)

        # self.save_path = os.path.join(self.log_dir, self.save_path)
        _classifiers_dir = os.path.join(self.log_dir, 'classifiers')
        os.makedirs(_classifiers_dir, exist_ok=True)
        
        logger.info(f'Directory {self.save_path} for saving the classifiers is now present')
        self.log_file = os.path.join(self.log_dir, log_file)
        self.train_features_file_path = os.path.join(self.log_dir, 'train_features_and_labels.pt')
        self.val_features_file_path = os.path.join(self.log_dir, 'val_features_and_labels.pt')

        # -- OPTIMIZATION
        self.lr = args['optimization']['lr']
        self.epochs = args['optimization']['epochs']
        self.embed_dims = VIT_EMBED_DIMS[self.model_name] # get dims based on model
        self.use_normalization = args['optimization'].get('use_normalization', False)
        # -- META
        self.device_name = args['meta']['device']

        self.device = torch.device(self.device_name if torch.cuda.is_available() else 'cpu')

        self.encoder = helper.init_encoder(device=self.device, 
                                        patch_size=self.patch_size,
                                        model_name=self.model_name,
                                        crop_size=self.crop_size,)

        # 404 error epoch not found
        self.pretrain_checkpoint_epoch = args.get('pretrain_checkpoint_epoch', 404) 

        ckpt = torch.load(self.pretrained_model_path, map_location=torch.device('cpu'))
        pretrained_dict = ckpt['encoder']

        # -- loading encoder
        for k, v in pretrained_dict.items():
            self.encoder.state_dict()[k[len('module.'):]].copy_(v) 

        if self.probe_checkpoints:
            self.model = LinearClassifier(self.embed_dims, self.num_classes, self.use_normalization)
        else: 
            self.model = Both(self.encoder, self.embed_dims, self.num_classes, self.use_normalization)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optim = optim.AdamW(self.model.parameters(), lr=self.lr)

        self.transform = transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

        self.train_dataset_images = ImageFolder(root=self.train_dataset_path, transform=self.transform)
        self.val_dataset_images = ImageFolder(root=self.val_dataset_path, transform=self.transform)

        # run feature extractor here
        # feature_extractor = FeatureExtractor(self.encoder)
        logger.info('Extracting features and saving them locally..')
        self.train_loader_images = DataLoader(self.train_dataset_images, batch_size=self.batch_size)
        self.val_loader_images = DataLoader(self.val_dataset_images, batch_size=self.batch_size)


        self.save_features(self.encoder, self.train_loader_images, self.train_features_file_path, self.device)
        self.save_features(self.encoder, self.val_loader_images, self.val_features_file_path, self.device)

        self.train_dataset_features = FeaturesDataset(self.train_features_file_path)
        self.val_dataset_features = FeaturesDataset(self.val_features_file_path)

        self.train_loader_features = DataLoader(self.train_dataset_features, batch_size=self.batch_size, shuffle=True)
        self.val_loader_features = DataLoader(self.val_dataset_features, batch_size=self.batch_size)
        self.logger = logger
        
        self.csvlogger = CSVLoggerAppender(self.log_file, 
                                            ('%d', 'pretrain_checkpoint_epoch'),
                                            ('%d', 'epoch'),
                                            ('%.5e', 'train_accuracy'),
                                            ('%.5e', 'val_accuracy'),
                                            ('%.5e', 'loss'),
                                            ('%.2f', 'time'))

    

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

    def save_features(self, encoder, _loader, features_file_path, device='cuda'):
        all_features = []
        all_labels = []
        with torch.no_grad():
            encoder.eval()
            for inputs, labels in _loader:
                inputs, labels = inputs.to(device), labels.to(device)
                output = encoder(inputs)

                # append to prior list
                all_features.append(output.cpu()) # shape: [batch_size, embedding shape]
                all_labels.append(labels.cpu()) # shape: [batch_size, 1]
        
        all_features = torch.cat(all_features, dim=0) # from a list of [batch_size, embedding_shape] to [total_images, embedding shape]
        all_labels = torch.cat(all_labels, dim=0) # [total_images, ]


        # save these features to disk in a compact file
        torch.save({
            'features': all_features,
            'labels' : all_labels
        }, features_file_path)

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
            for inputs, labels in self.train_loader_features:
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
                for inputs, labels in self.val_loader_features:
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

            self.logger.info('Epoch: %d/%d '
                            'Train accuracy: %.5e ' 
                            'Validation accuracy: %.5e '
                            'Loss %.5e '
                            'Time taken: %.2f seconds '
                            # 'ETA: %.2f '
                             % (epoch+1, self.epochs,
                                train_accuracy,
                                val_accuracy,
                                epoch_loss,
                                duration) )
            self.csvlogger.log(self.pretrain_checkpoint_epoch,
                               epoch+1, 
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
        self.logger.info('Cleaning up intermediate feature (.pt) files')
        os.remove(self.train_features_file_path)
        os.remove(self.val_features_file_path)
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
    
    # obtain all .tar files from the corresponding log dir in list format
    # iterate over them all
    # index them by overriding the `pretrained_model_path` in the params dictionary
    # let it do its job

    probe_checkpoints = params['data'].get('probe_checkpoints', None)
    if probe_checkpoints is None: # if it was not found or it is False
        logger.info('Doing Linear Probing on latest checkpoint')
        linear_prober = LinearProbe(params, logger)
        linear_prober.eval_linear()
        return
    
    # else:
    # ----------------------- AUTOMATIC LINEAR PROBING -----------------------
    # These are hacks and need to be cleaned up. We cannot be playing around with the config file!
    log_dir = params['logging'].get('log_dir', None)
    probe_prefix = params['data'].get('probe_prefix', None)

    prefixed_path = os.path.join(log_dir, probe_prefix)
    tarfiles = glob.glob(prefixed_path + '-ep*.pth.tar') # grab all requested pth tar files
    epoch = 0
    import copy
    for tarfile in sorted(tarfiles):
        temp_params = copy.deepcopy(params)
        eval_output = temp_params['logging'].get('eval_output', 'pfeature_extractor.out')
        logger.info('working on file %s ...' % str(tarfile))
        temp_params['logging']['pretrained_model_path'] = os.path.basename(tarfile) # use this tarfile name
        
        # First, remove all handlers! 
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # extract the epoch
        match_ = re.search(r'ep(\d+)\.', tarfile)

        if match_:
            epoch = int(match_.group(1))
        else:
            epoch += 1 # signify that no epoch could be read in the title file
        
        # keep info about what epoch this current run corresponds to
        temp_params['pretrain_checkpoint_epoch'] = epoch 

        # eval_output = os.path.join(log_dir, eval_output + f'-ep{epoch}.out') 
        # # do not alter evalout name
        logger.addHandler(logging.StreamHandler())
        logger.addHandler(logging.FileHandler(eval_output))

        temp_params['logging']['save_path'] += f'-ep{epoch}' 
        # temp_params['logging']['log_file'] += f'-ep{epoch}'  # do not create another log file, print them all in
        # pprint.pprint(temp_params)
        linear_prober = LinearProbe(temp_params, logger)
        linear_prober.eval_linear()
        logger.info('\n')


    # Για καθε αρχειο, εκτυπωσε σε αλλο output file τα αποτελεσματα
    # ωστε μετα να μπορεις να κανεις την αντιπαραβολη.
    # ή! μαζεψε τις καλυτερες επιδοσεις, αποθηκευσε τες σε ενα dictionary/output file
    # και επιστρεψε τες για plotting. Η δευτερη εναλλακτικη δεν αφήνει χώρο για μελλοντικές χρήσεις
    # και ίσως χρειαστεί να τρέξεις τα tests => πολυς χρονος χαμένος !

if __name__ == '__main__':
    """ No support for distributed training as of yet.
    Start linear probing based on config"""
    args = parser.parse_args() # get arguments from cmdline
    process_main(args.fname) 