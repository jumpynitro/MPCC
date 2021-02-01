import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
# DCGAN loss
class Loss_obj(object):
  def __init__(self, loss_type = 'hinge_dis', **kwargs):
    if   loss_type == 'hinge_dis':
      self.generator_loss     = self.loss_hinge_gen
      self.discriminator_loss = self.loss_hinge_dis
    elif loss_type == 'dc_gan':
      self.generator_loss     = self.loss_dcgan_dis
      self.discriminator_loss = self.loss_dcgan_gen
    elif loss_type == 'wgan':
      self.lambda_term        = 10
      self.generator_loss     = self.loss_wgan_dis
      self.discriminator_loss = self.loss_wgan_gen

    self.cross_entropy       = nn.CrossEntropyLoss()
    self.classification_loss = self.log_sum_exp_loss

  def loss_dcgan_dis(self, dis_fake, dis_real):
    L1 = torch.mean(F.softplus(-dis_real))
    L2 = torch.mean(F.softplus(dis_fake))
    return L1, L2

  def loss_dcgan_gen(self, dis_fake):
    loss = torch.mean(F.softplus(-dis_fake))
    return loss

  # Hinge Loss
  def loss_hinge_dis(self, dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake
  # def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
    # loss = torch.mean(F.relu(1. - dis_real))
    # loss += torch.mean(F.relu(1. + dis_fake))
    # return loss
  def loss_hinge_gen(self, dis_fake):
    loss = -torch.mean(dis_fake)
    return loss

  # Wasserstein Loss
  def loss_wgan_dis(self, dis_fake, dis_real):
    loss_real = - torch.mean(dis_real)
    loss_fake =   torch.mean(dis_fake)
    
    return loss_real, loss_fake
  # def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
    # loss = torch.mean(F.relu(1. - dis_real))
    # loss += torch.mean(F.relu(1. + dis_fake))
    # return loss
  def loss_wgan_gen(self, dis_fake):
    loss = -torch.mean(dis_fake)
    return loss

  def calculate_gradient_penalty(self, real_images, fake_images):
    eta = torch.FloatTensor(real_images.size(0),1,1,1).uniform_(0,1)
    eta = eta.expand(real_images.size(0), real_images.size(1), real_images.size(2), real_images.size(3))
    eta = eta.cuda()


    interpolated = eta * real_images + ((1 - eta) * fake_images)

    interpolated = interpolated.cuda()

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = self.D(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(),
                               create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
    return grad_penalty


  def log_likelihood(self, z, mu, lv):
    return torch.mean(0.5 * torch.sum((z - mu).pow(2) / lv.exp() + lv + np.log(math.pi), dim=1))

  def log_sum_exp_loss(self, y_pred, y):
    return -y_pred[torch.arange(y_pred.size(0)), y].mean()

  def log_gaussian(self, lv):
    aux = - 0.5 * (lv + np.log(2*math.pi) + 1)
    return aux.sum(dim=- 1)

    

