# Imports
import os
import functools
import math
import numpy as np
from tqdm import tqdm, trange


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

# Import my stuff
import inception_utils
import utils
import losses
import layers
#import train_fns
import train_fns
from argparse import ArgumentParser
from sync_batchnorm import patch_replication_callback
import json

import os
import io

def this_parser():
  usage = 'Parser for all scripts.'
  parser = ArgumentParser(description=usage)
  
  ### Dataset/Dataloader stuff ###
  parser.add_argument(
    '--experiment_name', type=str, default='',
    help='Experiment name'
         '(default: %(default)s)')
  parser.add_argument(
    '--weights_path', type=str, default='weights',
    help='weights path'
         '(default: %(default)s)')
  parser.add_argument(
    '--logs_path', type=str, default='logs',
    help='logs path'
         '(default: %(default)s)')
  parser.add_argument(
    '--accumulate_stats', action='store_true', default=False,
    help='accumulate stats (default: %(default)s)')
  parser.add_argument(
    '--name_suffix', type=str, default=None,
    help='weights path'
         '(default: %(default)s)')
  return parser


def run(config_parser):
  # Geting config
  experiment_name = model_name = config_parser['experiment_name']
  model_path      = '%s/%s'%(config_parser['weights_path'], model_name)
  logs_path       = '%s/%s'%(config_parser['logs_path'],    model_name)

  config_path     = '%s/metalog.txt'%logs_path
  new_file        = 'saved_stuff'
  save_path       = '%s/%s'%(model_path,new_file)

  if not os.path.exists(save_path):
    os.mkdir(save_path)
    
  device = 'cuda'

  file = open(config_path, 'r')
  all_file = file.read()
  fs1 = all_file.find('{')
  fs2 = all_file.find('}')
  config = all_file[fs1:fs2+1]
  import ast
  config = config.replace(", 'G_activation': ReLU()" , "")
  config = config.replace(", 'D_activation': ReLU()" , "")
  config = ast.literal_eval(config)

  config['samples_root'] = 'samples_test'

  config['skip_init']    = True
  #config['no_optim']     = True


  # Loading Model
  config['weights_root'] = config_parser['weights_path']


  model = __import__(config['model'])
  utils.seed_rng(config['seed'])
  # Prepare root folders if necessary
  utils.prepare_root(config)

  G = model.Generator(**config).to(device)
  D = model.Discriminator(**config).to(device)
  if config['is_encoder']:
    E = model.Encoder(**{**config, 'D': D}).to(device)
  Prior  = layers.Prior(**config).to(device)   
  GE = model.G_E(G,E,Prior)

  utils.load_weights(G, None, '',
                config['weights_root'], model_name, 
                config_parser['name_suffix'],
                G if config['ema'] else None, 
                E = None if not config['is_encoder'] else E,
                Prior = Prior if not config['prior_type'] == 'default' else None)

  # Sample functions	
  sample = functools.partial(utils.sample, G=G, Prior = Prior, config=config)

  # Accumulate stats?
  samples_name  = 'samples'
  if config_parser['accumulate_stats']:
    samples_name = 'samples_acc'
    utils.accumulate_standing_stats(G,
                           Prior, config['n_classes'],
                           config['num_standing_accumulations'])

  if config_parser['name_suffix'] is not None:
    samples_name += config_parser['name_suffix']

  # Sample and save in npz
  config['sample_npz']     = True
  config['sample_num_npz'] = 50000
  G_batch_size             = Prior.bs
  # Sample a number of images and save them to an NPZ, for use with TF-Inception

  if config['sample_npz']:
  # Lists to hold images and labels for images
    x, y = [], []
    print('Sampling %d images and saving them to npz...' % config['sample_num_npz'])
    for i in trange(int(np.ceil(config['sample_num_npz'] / float(G_batch_size)))):
      with torch.no_grad():
        images, labels = sample()
      x += [np.uint8(255 * (images.cpu().numpy() + 1) / 2.)]
      y += [labels.cpu().numpy()]
    x = np.concatenate(x, 0)[:config['sample_num_npz']]
    y = np.concatenate(y, 0)[:config['sample_num_npz']]    
    print('Images shape: %s, Labels shape: %s' % (x.shape, y.shape))
    npz_filename = '%s/%s/%s.npz' % (config['weights_root'], experiment_name, samples_name)
    print('Num %d'% len(x))
    print('Saving npz to %s...' % npz_filename)
    np.savez(npz_filename, **{'x' : x, 'y' : y})


  # Reconstruction metrics
  D_batch_size = (config['batch_size'] * config['num_D_steps']
                  * config['num_D_accumulations'])
  config_aux = config.copy()
  config_aux['augment'] = False
  dataloader_noaug = utils.get_data_loaders(**{**config_aux, 'batch_size': D_batch_size})


  if config_parser['accumulate_stats']:
     utils.accumulate_standing_stats_E(E, Prior, dataloader_noaug, device, config)
  test_acc, test_acc_iter, error_rec = train_fns.test_accuracy(GE, dataloader_noaug, device, config['D_fp16'], config)


  json_metric_name = samples_name + '_json'
  if not os.path.isfile('%s/%s.json' % (model_path, json_metric_name)):
    metric_dict = {}
    metric_dict['test_acc' ]  = test_acc
    metric_dict['error_rec']  = error_rec
    json.dump(metric_dict, open('%s/%s.json' % (model_path, json_metric_name), 'w'))
  else:
    metric_dict = json.load(open('%s/%s.json' % (model_path, json_metric_name)))
    metric_dict['inception_mean' ]  = test_acc
    metric_dict['inception_std'  ]  = error_rec
    json.dump(metric_dict, open('%s/%s.json' % (model_path, json_metric_name), 'w'))



def main():
  # parse command line and run    
  parser = this_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)
  
if __name__ == '__main__':    
  main()
