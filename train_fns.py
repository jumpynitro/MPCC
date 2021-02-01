''' train_fns.py
Functions for the main loop of training different conditional image models
'''
import torch
import torch.nn as nn
import torchvision
import os
import numpy as np

import utils
import losses


# Dummy training function for debugging
def dummy_training_function():
  def train(x, y):
    return {}
  return train


def GAN_training_function(G, D, GD, Prior, ema, state_dict, config, my_loss, E = None):
  def train(x, y, this_iter):
    G.optim.zero_grad()
    D.optim.zero_grad()
    if E is not None:
      E.optim.zero_grad()
    if not (config['prior_type'] == 'default'):
      Prior.optim.zero_grad()

    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    counter = 0
    
    ######### Discriminator ##############
    # Optionally toggle D and G's "require_grad"
    if config['toggle_grads']:
      if E is not None:
        utils.toggle_grad(E, False)
      if not (config['prior_type'] == 'default'):
        utils.toggle_grad(Prior, False)
      utils.toggle_grad(D, True)
      utils.toggle_grad(G, False)
      
    for step_index in range(config['num_D_steps']):
      # If accumulating gradients, loop multiple times before an optimizer step
      D.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        z_, y_ = Prior.sample_()
        D_fake, D_real = GD(z_[:config['batch_size']], y_[:config['batch_size']], 
                            x[counter], y[counter], train_G=False, 
                            split_D=config['split_D'], is_Enc = False)
         
        # Compute components of D's loss, average them, and divide by 
        # the number of gradient accumulations
        D_loss_real, D_loss_fake = my_loss.discriminator_loss(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
        D_loss.backward()
        counter += 1
        
      # Optionally apply ortho reg in D
      if config['D_ortho'] > 0.0:
        # Debug print to indicate we're using ortho reg in D.
        print('using modified ortho reg in D')
        utils.ortho(D, config['D_ortho'])
      
      D.optim.step()
    
    ########## Generator ################3
    # Optionally toggle "requires_grad"
    if config['toggle_grads']:
      if not (config['prior_type'] == 'default') and not this_iter%config['update_GMM_every_n']:
        utils.toggle_grad(Prior, True)
      utils.toggle_grad(D, False)
      utils.toggle_grad(G, True)
    # Zero G's gradients by default before training G, for safety
    G.optim.zero_grad()
    if not (config['prior_type'] == 'default'):
      Prior.optim.zero_grad()
    # If accumulating gradients, loop multiple times
    for accumulation_index in range(config['num_G_accumulations']):    
      z_, y_ = Prior.sample_()
      D_fake = GD(z_, y_, train_G=True, split_D=config['split_D'])
      G_loss = my_loss.generator_loss(D_fake)
      (G_loss  / float(config['num_G_accumulations'])).backward()

    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:
      print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G, config['G_ortho'], 
                  blacklist=[param for param in G.shared.parameters()])
    G.optim.step()
    if not (config['prior_type'] == 'default') and not this_iter%config['update_GMM_every_n']:
      Prior.optim.step()

    ############# Encoder ##########    
    if E is not None:
      # Optionally toggle "requires_grad"
      if  config['toggle_grads']:
        utils.toggle_grad(D, False)
        utils.toggle_grad(G, False)
        utils.toggle_grad(E, True)

      counter = 0

      for step_index in range(config['num_E_steps']):
        # Zero G's gradients by default before training G, for safety
        E.optim.zero_grad()
        if not (config['prior_type'] == 'default'):
          Prior.optim.zero_grad()

        # If accumulating gradients, loop multiple times
        for accumulation_index in range(config['num_E_accumulations']):    
          z_, y_     = Prior.sample_()
          z_mu, z_lv = GD(z_, y_, train_G=False, split_D=config['split_D'], is_Enc = True)
          z_p        = z_ if not config['is_latent_detach'] else z_.detach()
          E_loss     = my_loss.log_likelihood(z_p, z_mu, z_lv) /float(config['lambda_encoder'])
          total_loss = E_loss
          if not (config['prior_type'] == 'default') and not this_iter%config['update_GMM_every_n'] and step_index == 0:
            log_y_pred     = Prior.latent_classification(z_)
            Prior_loss     = my_loss.classification_loss(log_y_pred, y_)/ float(config['num_E_accumulations'])
            total_loss    += Prior_loss
            if config['is_loss3'] != 0:
              if config['is_loss3'] == -1:
                loss3       = torch.sum((1/float(config['lambda_encoder']))*my_loss.log_gaussian(Prior.lv_c)/Prior.n_classes)
                total_loss += loss3
              else:
                loss3       = torch.sum(config['is_loss3']*my_loss.log_gaussian(Prior.lv_c)/(Prior.n_classes*Prior.dim_z))
                total_loss += loss3
          MSE_loss       = torch.mean( torch.sum((z_ - z_mu).pow(2), dim = 1) )
          (total_loss/float(config['num_E_accumulations'])).backward()
          counter += 1
         
        # Optionally apply modified ortho reg in G
        if config['E_ortho'] > 0.0:
          print('using modified ortho reg in E') # Debug print to indicate we're using ortho reg in G
          # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
          utils.ortho(E, config['E_ortho'])
        E.optim.step()
        if not (config['prior_type'] == 'default') and not this_iter%config['update_GMM_every_n'] and step_index == 0:
          acc_samples = torch.mean((y_ == log_y_pred.argmax(1)).float())
          Prior.optim.step()

    # If we have an ema, update it, regardless of if we test with it or not
    if config['ema']:
      ema.update(state_dict['itr'])
    
    out = {'G_loss': float(G_loss.item()), 
            'D_loss_real': float(D_loss_real.item()),
            'D_loss_fake': float(D_loss_fake.item())}
    # Return G's loss and the components of D's loss.
    if not (config['prior_type'] == 'default') and not this_iter%config['update_GMM_every_n']:
      out['P_acc_samples']     = float(acc_samples.item())
    if E is not None:
      out['E_log_likelihood']  = float(E_loss.item())
      out['E_MSE_loss']        = float(MSE_loss.item())
    return out
  return train
  
''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''
def save_and_sample(G, D, G_ema, Prior, fixed_z, fixed_y,
                    state_dict, config, experiment_name, E = None):
  utils.save_weights(G, D, state_dict, config['weights_root'],
                     experiment_name, None, G_ema if config['ema'] else None,
                     E = None if not config['is_encoder'] else E,
                     Prior  = None if config['prior_type'] == 'default' else Prior)
  # Save an additional copy to mitigate accidental corruption if process
  # is killed during a save (it's happened to me before -.-)
  if config['num_save_copies'] > 0:
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name,
                       'copy%d' %  state_dict['save_num'],
                       G_ema if config['ema'] else None,
                       E = None if not config['is_encoder'] else E,
                       Prior = None if config['prior_type'] == 'default' else Prior)
    state_dict['save_num'] = (state_dict['save_num'] + 1 ) % config['num_save_copies']
    
  # Use EMA G for samples or non-EMA?
  which_G = G_ema if config['ema'] and config['use_ema'] else G
  
  # Accumulate standing statistics?
  if config['accumulate_stats']:
    utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                           Prior, config['n_classes'],
                           config['num_standing_accumulations'])
  
  # Save a random sample sheet with fixed z and y      
  with torch.no_grad():
    if config['parallel']:
      fixed_Gz =  nn.parallel.data_parallel(which_G,
                (Prior.obtain_latent_from_z_y(fixed_z, fixed_y), which_G.shared(fixed_y)))
    else:
      fixed_Gz = which_G(Prior.obtain_latent_from_z_y(fixed_z, fixed_y), which_G.shared(fixed_y))
  if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
    os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
  image_filename = '%s/%s/fixed_samples%d.jpg' % (config['samples_root'], 
                                                  experiment_name,
                                                  state_dict['itr'])
  torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                             nrow=int(fixed_Gz.shape[0] **0.5), normalize=True)
  # For now, every time we save, also save sample sheets
  utils.sample_sheet(which_G, Prior,
                     classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
                     num_classes=config['n_classes'],
                     samples_per_class=10, parallel=config['parallel'],
                     samples_root=config['samples_root'],
                     experiment_name=experiment_name,
                     folder_number=state_dict['itr'])

  # Also save interp sheets
  if not (config['prior_type'] == 'GMM' and config['G_shared']):
    for fix_z, fix_y in zip([False, False, True], [False, True, False]):
      utils.interp_sheet(which_G, Prior,
                       num_per_sheet=16,
                       num_midpoints=8,
                       num_classes=config['n_classes'],
                       parallel=config['parallel'],
                       samples_root=config['samples_root'],
                       experiment_name=experiment_name,
                       folder_number=state_dict['itr'],
                       sheet_number=0,
                       fix_z=fix_z, fix_y=fix_y, device='cuda')




def cluster_acc_torch(Y_pred, Y):
    #import pdb
    #pdb.set_trace()
    #assert Y_pred.size() == Y.size()
    D = int(torch.max(torch.max(Y_pred), torch.max(Y))+1)
    ww = torch.zeros(D,D)
    for i in range(Y_pred.size()[0]):
        ww[int(Y_pred[i]), int(Y[i])] += 1
    accuracy = torch.sum(torch.max(ww, dim = 1)[0])/torch.sum(ww)
    return accuracy

def cluster_acc(Y_pred, Y):
    D = int(np.max((np.max(Y_pred), np.max(Y)))+1)
    w = np.zeros((D,D))
    for i in range(len(Y_pred)):
        w[int(Y_pred[i]), int(Y[i])] += 1
    #print(w)
    accuracy = np.sum(np.max(w,1))/np.sum(w)
    return accuracy

''' Calculate accuracy'''
def test_accuracy(G_E, dataloader, device, D_fp16, config, obtain_y = False):
  #import pdb
  #pdb.set_trace()
  total_acc        = 0
  total_acc_iter   = 0
  total_y          = []
  total_y_pred     = []
  total_dist_gauss = []
  error_rec        = 0
  total_data       = 0
  for x, y in dataloader[0]:
    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    counter = 0
    for step_index in range(config['num_D_steps']):
      for accumulation_index in range(config['num_D_accumulations']):
        if D_fp16:
          x_counter, y_counter = x[counter].to(device).half(), y[counter].to(device)
        else:
          x_counter, y_counter = x[counter].to(device), y[counter].to(device)
        with torch.no_grad():
          if config['parallel']:
            o = nn.parallel.data_parallel(G_E.E, x_counter)
          else:
            o = G_E.E(x_counter)
          log_y_pred     = G_E.P.latent_classification(o[0])
          _ , dist_gauss = G_E.P.gmm_membressy2(o[0], mse_norm = True)

          if not config['is_not_rec']:
            if config['parallel']:
              x_pred     = nn.parallel.data_parallel(G_E.G, (o[0], G_E.G.shared(log_y_pred.argmax(1).long())))
            else: 
              x_pred     = G_E.G(o[0], G_E.G.shared(log_y_pred.argmax(1).long()))

            this_error   = torch.mean(x_counter - x_pred).pow(2).item()
            error_rec   += this_error*x_counter.size(0) 
            total_data  += x_counter.size(0) 
        counter += 1 
        total_y.append(y_counter.data.cpu().numpy())
        total_y_pred.append(log_y_pred.argmax(1).data.cpu().numpy())
        total_dist_gauss.append(dist_gauss.data.cpu().numpy())
        total_acc_iter += cluster_acc_torch(log_y_pred.argmax(1), y_counter).item()
        #counter +=1


  total_y          = np.concatenate(total_y)
  total_y_pred     = np.concatenate(total_y_pred)
  total_acc        = cluster_acc(total_y_pred, total_y)
  total_dist_gauss = np.concatenate(total_dist_gauss)
  if not config['is_not_rec']:
    error_rec = error_rec / total_data
  else:
    error_rec = 0
  total_acc_iter = total_acc_iter/(len(dataloader[0])*config['num_D_steps']*config['num_D_accumulations'])
  #pdb.set_trace()
  if not obtain_y:
    return total_acc, total_acc_iter, error_rec
  else:
    return total_y, total_y_pred, total_dist_gauss


# Sample function for sample sheets
def test_p_acc(G_E, device, config):

  z_ = utils.Distribution(torch.randn(config['num_batch_prior_acc'], config['dim_z'], requires_grad=False))
  z_.init_distribution('normal', mean=0, var= 1.0)
  z_ = z_.to(device,torch.float16 if config['G_fp16'] else torch.float32)     
  if config['G_fp16']:
    z_ = z_.half()
  y = torch.arange(0, config['n_classes'], device='cuda').repeat(config['num_batch_prior_acc']//config['n_classes'])
  p_mse = 0
  p_lik = 0

  for i in range(config['num_iter_prior_acc']):
    z_.sample_()
    latent = G_E.P.obtain_latent_from_z_y(z_, y)
    with torch.no_grad():
      if config['parallel']:
        o = nn.parallel.data_parallel(G_E.G, (latent, G_E.G.shared(y)))
      else:
        o = G_E.G(latent, G_E.G.shared(y))

      if config['parallel']:
        z_mu, z_var = nn.parallel.data_parallel(G_E.E, o)
      else:
        z_mu, z_var = G_E.E(o)

      actual_mse        = (latent - z_mu).pow(2)
      actual_likelihood = actual_mse/z_var
      p_mse += torch.mean(actual_mse)
      p_lik += torch.mean(actual_likelihood)

  p_mse = p_mse/config['num_iter_prior_acc']
  p_lik = p_lik/config['num_iter_prior_acc']
  return p_mse.item(), p_lik.item()
''' This function runs the inception metrics code, checks if the results
    are an improvement over the previous best (either in IS or FID, 
    user-specified), logs the results, and saves a best_ copy if it's an 
    improvement. '''
def test(G, D, G_ema, Prior, state_dict, config, sample, get_inception_metrics,
         experiment_name, test_log, E = None, test_acc = None):
  print('Gathering inception metrics...')
  if config['accumulate_stats']:
    utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                           Prior, config['n_classes'],
                           config['num_standing_accumulations'])
  if (config['dataset'] in ['C10']):
    IS_mean, IS_std, FID = get_inception_metrics(sample, 
                                               config['num_inception_images'],
                                               num_splits=10)
    print('Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (state_dict['itr'], IS_mean, IS_std, FID))
    # If improved over previous best metric, save approrpiate copy
    if ((config['which_best'] == 'IS' and IS_mean > state_dict['best_IS'])
      or (config['which_best'] == 'FID' and FID < state_dict['best_FID'])):
      print('%s improved over previous best, saving checkpoint...' % config['which_best'])
      utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name, 'best%d' % state_dict['save_best_num'],
                       G_ema if config['ema'] else None,
                       E = None if not config['is_encoder'] else E,
                       Prior  = None if config['prior_type'] == 'default' else Prior)
      state_dict['save_best_num'] = (state_dict['save_best_num'] + 1 ) % config['num_best_copies']
  else:
    IS_mean, IS_std, FID = 0, 0, 0

  if test_acc is not None:
    _, _, error_rec, _, _ = test_acc
    if error_rec < state_dict['best_error_rec']:
      utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name, 'best_error_rec',
                       G_ema if config['ema'] else None,
                       E = None if not config['is_encoder'] else E,
                       Prior  = None if config['prior_type'] == 'default' else Prior)
    state_dict['best_error_rec'] = min(state_dict['best_error_rec'], error_rec)

  state_dict['best_IS'] = max(state_dict['best_IS'], IS_mean)
  state_dict['best_FID'] = min(state_dict['best_FID'], FID)
  # Log results to file
  if test_acc is None:
    test_log.log(itr=int(state_dict['itr']), IS_mean=float(IS_mean),
                 IS_std=float(IS_std), FID=float(FID))
  else:
    test_acc, test_acc_iter, error_rec, p_mse, p_lik = test_acc
    test_log.log(itr=int(state_dict['itr']), IS_mean=float(IS_mean),
                 IS_std=float(IS_std), FID=float(FID), Acc = float(test_acc),
                 Acc_iter = float(test_acc_iter), error_rec = float(error_rec),
                 p_mse = float(p_mse), p_lik = float(p_lik))
