import os
import numpy as np
import math
import sys
import pdb
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, size_average=False)  / y.shape[0]
    return l_kl

def adjust_learning_rate(optimizer, epoch, learing_rate):
    if epoch < 800:
        lr = learing_rate
    elif epoch < 1600:
        lr = 0.1*learing_rate
    else:
        lr = 0.01*learing_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def train_model(net, teacher, generator, data_test_loader, device, criterion, optimizer_G, optimizer_S, lr_G=0.2, lr_S= 2e-3, one_hot=1, information_entropy=10, activation=0.1, batch_size=32, img_size=32, latent_dim=100, n_epochs=200, dataset='MNIST'):
    accr = 0
    accr_best = 0
    batches_done = 0
    for epoch in range(n_epochs):

        total_correct = 0
        avg_loss = 0.0
        if dataset != 'MNIST':
            adjust_learning_rate(optimizer_S, epoch, lr_S)

        for i in range(120):
            net.train()
            z = Variable(torch.randn(batch_size, latent_dim)).to(device)
            optimizer_G.zero_grad()
            optimizer_S.zero_grad()        
            gen_imgs = generator(z)
            outputs_T, features_T = teacher(gen_imgs, out_feature=True)   
            pred = outputs_T.data.max(1)[1]
            loss_activation = -features_T.abs().mean()
            loss_one_hot = criterion(outputs_T,pred)
            softmax_o_T = F.softmax(outputs_T, dim = 1).mean(dim = 0)
            loss_information_entropy = (softmax_o_T * torch.log(softmax_o_T)).sum()
            loss = loss_one_hot * one_hot + loss_information_entropy * information_entropy + \
                    loss_activation * activation
            loss_kd = kdloss(net(gen_imgs.detach()), outputs_T.detach()) 
            loss += loss_kd       
            loss.backward()
            optimizer_G.step()
            optimizer_S.step() 
            if i == 1:
                print ("[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_a: %f] [loss_kd: %f]" % (epoch, n_epochs,loss_one_hot.item(), loss_information_entropy.item(), loss_activation.item(), loss_kd.item()))

        with torch.no_grad():
            for i, (images, labels) in enumerate(data_test_loader):
                images = images.cuda()
                labels = labels.cuda()
                net.eval()
                output = net(images)
                avg_loss += criterion(output, labels).sum()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()

        avg_loss = avg_loss / (len(data_test_loader)*batch_size)
        print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), float(total_correct) / (len(data_test_loader)*batch_size)))
        accr = round(float(total_correct) / (len(data_test_loader)*batch_size), 4)
        if accr > accr_best:
            torch.save(net,'student')
            accr_best = accr