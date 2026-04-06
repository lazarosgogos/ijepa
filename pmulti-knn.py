# Author: Lazaros Gogos
# 2024 - 07 - 15
#
# KNeighbors classification on pretrained models based on the I-JEPA architecture

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
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import glob
import re
import copy
import gc

from sklearn.neighbors import KNeighborsClassifier

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


class LinearProbe():
    def __init__(self, args, logger):
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
        self.probe_checkpoints = args['data'].get('probe_checkpoints', False)
        self.probe_prefix = args['data'].get('probe_prefix', None)
        self.num_workers = args['data'].get('num_workers', 1)
        self.pin_mem = args['data'].get('pin_mem', False)

        # -- LOGGING
        self.log_dir = args['logging']['log_dir']
        self.pretrained_model_path = args['logging']['pretrained_model_path']
        self.save_path = args['logging']['save_path']
        self.checkpoint_freq = args['logging']['checkpoint_freq']
        self.log_file = args['logging']['log_file']

        self.pretrained_model_path = os.path.join(self.log_dir, self.pretrained_model_path)

        # _classifiers_dir = os.path.join(self.log_dir, 'classifiers')
        # os.makedirs(_classifiers_dir, exist_ok=True)
        # logger.info(f'Directory {_classifiers_dir} for saving the classifiers is now present')

        self.log_file = os.path.join(self.log_dir, self.log_file)

        # -- META
        self.device_name = args['meta']['device']
        self.device = torch.device(self.device_name if torch.cuda.is_available() else 'cpu')

        self.embed_dims = VIT_EMBED_DIMS[self.model_name]

        self.encoder = helper.init_encoder(
            device=self.device,
            patch_size=self.patch_size,
            model_name=self.model_name,
            crop_size=self.crop_size,
        )

        self.pretrain_checkpoint_epoch = args.get('pretrain_checkpoint_epoch', 404)

        ckpt = torch.load(self.pretrained_model_path, map_location=torch.device('cpu'))
        pretrained_dict = ckpt['encoder']
        for k, v in pretrained_dict.items():
            self.encoder.state_dict()[k[len('module.'):]].copy_(v)

        self.transform = transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

        self.train_dataset_images = ImageFolder(root=self.train_dataset_path, transform=self.transform)
        self.val_dataset_images = ImageFolder(root=self.val_dataset_path, transform=self.transform)

        self.train_loader_images = DataLoader(
            self.train_dataset_images, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.pin_mem,
            prefetch_factor=4, persistent_workers=True
        )
        self.val_loader_images = DataLoader(
            self.val_dataset_images, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.pin_mem,
            prefetch_factor=4, persistent_workers=True
        )

        self.logger = logger
        extraction_time_start = time.perf_counter()
        self.logger.info('Extracting features...')
        self.train_features, self.train_labels = self.extract_features(self.encoder, self.train_loader_images, self.device)
        self.val_features, self.val_labels = self.extract_features(self.encoder, self.val_loader_images, self.device)
        self.logger.info(f'Time taken to extract features: {time.perf_counter() - extraction_time_start:.2f}s')

        self.csvlogger = CSVLoggerAppender(
            self.log_file,
            ('%d', 'pretrain_checkpoint_epoch'),
            ('%.5e', 'knn_accuracy_train'),
            ('%.5e', 'knn_accuracy_test'),
        )

    def extract_features(self, encoder, loader, device='cuda'):
        total_samples = len(loader.dataset)
        feature_dim = VIT_EMBED_DIMS[self.model_name]

        all_features = torch.empty(total_samples, feature_dim, device=device, dtype=torch.float32)
        all_labels = torch.empty(total_samples, device=device, dtype=torch.long)

        encoder.eval()
        start_idx = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                output = encoder(inputs)
                output = torch.mean(output, dim=1, dtype=output.dtype)

                batch_size = inputs.size(0)
                all_features[start_idx:start_idx + batch_size] = output
                all_labels[start_idx:start_idx + batch_size] = labels
                start_idx += batch_size

        return all_features, all_labels

    def eval_knn(self, n_neighbors=5):
        """Perform KNN evaluation on the already-extracted features."""
        train_features = self.train_features.cpu().numpy()
        train_labels = self.train_labels.cpu().numpy()
        val_features = self.val_features.cpu().numpy()
        val_labels = self.val_labels.cpu().numpy()

        classifier = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        classifier.fit(train_features, train_labels)

        knn_acc_train = classifier.score(train_features, train_labels)
        knn_acc_test = classifier.score(val_features, val_labels)

        self.logger.info(
            f'\tEpoch {self.pretrain_checkpoint_epoch}, '
            f'KNN accuracy train: {knn_acc_train:.5e}, '
            f'KNN accuracy test: {knn_acc_test:.5e}'
        )
        self.csvlogger.log(self.pretrain_checkpoint_epoch, knn_acc_train, knn_acc_test)

        gc.collect()
        torch.cuda.empty_cache()


def process_main(fname, devices=['cuda:0']):
    """This function was inspired by main.py from IJEPA"""
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    global_time_start = time.perf_counter()
    logger.info(f'called-params {fname}')

    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params....')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    multi_probe = params.get('multi_probing', None)
    assert multi_probe is not None, 'multi probing is not enabled'

    dirs = params.get('multi_probing', list())
    assert len(dirs) != 0, 'No directories were found.'

    for log_dir in dirs:
        probe_prefix = params['data'].get('probe_prefix', None)
        prefixed_path = os.path.join(log_dir, probe_prefix)
        tarfiles = glob.glob(prefixed_path + '*-ep*.pth.tar')
        epoch = 0

        temp_params = copy.deepcopy(params)
        temp_params['logging']['log_dir'] = log_dir

        for tarfile in sorted(tarfiles):
            logger.info('working on file %s ...' % str(tarfile))
            temp_params['logging']['pretrained_model_path'] = os.path.basename(tarfile)

            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            match_ = re.search(r'ep(\d+)\.', tarfile)
            if match_:
                epoch = int(match_.group(1))
            else:
                epoch += 1

            temp_params['pretrain_checkpoint_epoch'] = epoch

            basename = os.path.basename(os.path.normpath(log_dir))
            eval_output = os.path.join(log_dir, 'o-k-NN-jepa-' + basename + '.out')

            logger.addHandler(logging.StreamHandler())
            logger.addHandler(logging.FileHandler(eval_output))

            temp_params['logging']['save_path'] += f'-ep{epoch}'
            basename = os.path.basename(os.path.normpath(log_dir))
            temp_params['logging']['log_file'] = 'stats-kNN-' + basename + '.csv'

            knn_prober = LinearProbe(temp_params, logger)
            knn_prober.eval_knn()
            logger.info('\n')

    logger.info(f'Time taken to complete whole task: {time.perf_counter() - global_time_start:.2f}s')


if __name__ == '__main__':
    args = parser.parse_args()
    process_main(args.fname)