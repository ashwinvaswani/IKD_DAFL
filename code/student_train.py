import argparse
import os
import numpy as np
import math
import sys
import pdb

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import resnet
from generator import Generator
import utils
import vgg16

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','cifar10','cifar100'])
parser.add_argument('--data', type=str, default='data')
parser.add_argument('--teacher_dict', type=str, default='teacher_resnet50_200ep_cifar10.pt')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
parser.add_argument('--lr_G', type=float, default=0.2, help='learning rate')
parser.add_argument('--lr_S', type=float, default=2e-3, help='learning rate')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--oh', type=float, default=1, help='one hot loss')
parser.add_argument('--ie', type=float, default=10, help='information entropy loss')
parser.add_argument('--a', type=float, default=0.1, help='activation loss')

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

generator = Generator(args.channels,args.img_size,args.latent_dim).to(device)
criterion = nn.CrossEntropyLoss().to(device)

if args.dataset == 'MNIST' :
    teacher = resnet.ResNet50(num_classes=10).to(device)
    teacher.load_state_dict(torch.load(args.teacher_dict))
    teacher.eval()
    
    data_test = MNIST(args.data,
                      train=False,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                          ]))           
    data_test_loader = DataLoader(data_test, batch_size=64, num_workers=1, shuffle=False)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G)
#     optimizer_S = torch.optim.Adam(net.parameters(), lr=args.lr_S)
    
elif args.dataset == 'cifar10' :
    teacher = resnet.ResNet50(num_classes=10).to(device)
    teacher.load_state_dict(torch.load(args.teacher_dict))
    teacher.eval()
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]) 
    data_test = CIFAR10(args.data,
                        train=False,
                        transform=transform_test)
    
    data_test_loader = DataLoader(data_test, batch_size=args.batch_size, num_workers=0)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G)

#     optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)

elif args.dataset == 'cifar100' :
    teacher = resnet.ResNet50(num_classes=100).to(device)
    teacher.load_state_dict(torch.load(args.teacher_dict))
    teacher.eval()
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]) 
    data_test = CIFAR100(args.data,
                        train=False,
                        transform=transform_test)
    
    data_test_loader = DataLoader(data_test, batch_size=args.batch_size, num_workers=0)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G)

#     optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)


original_parameters = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
classifier_parameters = [512,4096,4096]
cnt = 0
# net = vgg16.vgg16(original_parameters,classifier_parameters).to(device)
net = resnet.ResNet18(num_classes=10).to(device)
optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)
accr_best, cnt = utils.train_model(net, teacher, generator, data_test_loader, device, criterion, optimizer_G, optimizer_S, args.lr_G, args.lr_S, args.oh, args.ie, args.a, args.batch_size, args.img_size, args.latent_dim, args.n_epochs, args.dataset, cnt)
#         print()
#         print("Best accuracy currently : {}".format(accr_best))
# for depth in range(7):
#     last_number = vgg16.get_output_nodes(original_parameters)
#     classifier_params = [last_number,4096,4096]
#     if depth == 0:
#         print("Original network: ")
#         net = vgg16.vgg16(original_parameters,classifier_parameters).to(device)
#         # print(net)
#         optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)
# #         net, hist = train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=EPOCHS)
#         accr_best, cnt = utils.train_model(net, teacher, generator, data_test_loader, device, criterion, optimizer_G, optimizer_S, args.lr_G, args.lr_S, args.oh, args.ie, args.a, args.batch_size, args.img_size, args.latent_dim, args.n_epochs, args.dataset, cnt)
#         print()
#         print("Best accuracy currently : {}".format(accr_best))
#         print("\n############################################################################\n")

#     # For odd case :
#     elif depth % 2 == 1:
#         print("At depth " + str(depth) + ": ")
#         print("Dividing filters by 2: ")
#         original_parameters = [int(i/2) if type(i) == int else i for i in original_parameters]
#         classifier_parameters = [int(i/2) if type(i) == int else i for i in classifier_parameters]
#         print(original_parameters)
#         net = vgg16.vgg16(original_parameters,classifier_parameters).to(device)
#         last_layer = [net.features[i] for i in range(len(net.features))][-1]
#         if str(last_layer) ==  'ReLU(inplace=True)':
#             net.features = nn.Sequential(*[net.features[i] for i in range(len(net.features) -1)])
#         # print(net)
#         optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)
# #         net, hist = train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=EPOCHS)
#         accr_best, cnt = utils.train_model(net, teacher, generator, data_test_loader, device, criterion, optimizer_G, optimizer_S, args.lr_G, args.lr_S, args.oh, args.ie, args.a, args.batch_size, args.img_size, args.latent_dim, args.n_epochs, args.dataset, cnt)
#         print()
#         print("Best accuracy currently : {}".format(accr_best))
#         print("\n############################################################################\n")

#     # For even case :
#     elif depth % 2 == 0:
#         print("At depth " + str(depth) + ": ")
#         print("Removing last layer: ")
#         net = vgg16.vgg16(original_parameters,classifier_parameters).to(device)
#         second_last_layer = [net.features[i] for i in range(len(net.features)-1)][-1]
#         if str(second_last_layer) ==  'ReLU(inplace=True)':
#             net.features = nn.Sequential(*[net.features[i] for i in range(len(net.features) -2)])
#         else:
#             net.features = nn.Sequential(*[net.features[i] for i in range(len(net.features) -1)])
#         original_parameters = original_parameters[:-1]
#         # print(net)
#         optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)
# #         net, hist = train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=EPOCHS)
#         accr_best, cnt = utils.train_model(net, teacher, generator, data_test_loader, device, criterion, optimizer_G, optimizer_S, args.lr_G, args.lr_S, args.oh, args.ie, args.a, args.batch_size, args.img_size, args.latent_dim, args.n_epochs, args.dataset, cnt)
#         print()
#         print("Best accuracy currently : {}".format(accr_best))
#         print("\n############################################################################\n")

#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

# import argparse
# import os
# import numpy as np
# import math
# import sys
# import pdb

# import torchvision.transforms as transforms

# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torch.autograd import Variable

# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# from torchvision.datasets.mnist import MNIST
# from lenet import LeNet5Half
# from torchvision.datasets import CIFAR10
# from torchvision.datasets import CIFAR100
# import resnet

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','cifar10','cifar100'])
# parser.add_argument('--data', type=str, default='/cache/data/')
# parser.add_argument('--teacher_dir', type=str, default='/cache/models/')
# parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
# parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
# parser.add_argument('--lr_G', type=float, default=0.2, help='learning rate')
# parser.add_argument('--lr_S', type=float, default=2e-3, help='learning rate')
# parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
# parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
# parser.add_argument('--channels', type=int, default=1, help='number of image channels')
# parser.add_argument('--oh', type=float, default=1, help='one hot loss')
# parser.add_argument('--ie', type=float, default=10, help='information entropy loss')
# parser.add_argument('--a', type=float, default=0.1, help='activation loss')
# parser.add_argument('--output_dir', type=str, default='/cache/models/')

# opt = parser.parse_args()

# img_shape = (opt.channels, opt.img_size, opt.img_size)

# cuda = True 

# accr = 0
# accr_best = 0

# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()

#         self.init_size = opt.img_size // 4
#         self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_size**2))

#         self.conv_blocks0 = nn.Sequential(
#             nn.BatchNorm2d(128),
#         )
#         self.conv_blocks1 = nn.Sequential(
#             nn.Conv2d(128, 128, 3, stride=1, padding=1),
#             nn.BatchNorm2d(128, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         self.conv_blocks2 = nn.Sequential(
#             nn.Conv2d(128, 64, 3, stride=1, padding=1),
#             nn.BatchNorm2d(64, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
#             nn.Tanh(),
#             nn.BatchNorm2d(opt.channels, affine=False) 
#         )

#     def forward(self, z):
#         out = self.l1(z)
#         out = out.view(out.shape[0], 128, self.init_size, self.init_size)
#         img = self.conv_blocks0(out)
#         img = nn.functional.interpolate(img,scale_factor=2)
#         img = self.conv_blocks1(img)
#         img = nn.functional.interpolate(img,scale_factor=2)
#         img = self.conv_blocks2(img)
#         return img
        
# generator = Generator().cuda()
# teacher = resnet.ResNet50(num_classes=10).to(device)
# teacher.load_state_dict(torch.load(args.teacher_dict))
# teacher.eval()
# criterion = torch.nn.CrossEntropyLoss().cuda()

# original_parameters = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
# classifier_parameters = [512,4096,4096]
# cnt = 0
# net = vgg16.vgg16(original_parameters,classifier_parameters).to(device)

# teacher = nn.DataParallel(teacher)
# generator = nn.DataParallel(generator)

# def kdloss(y, teacher_scores):
#     p = F.log_softmax(y, dim=1)
#     q = F.softmax(teacher_scores, dim=1)
#     l_kl = F.kl_div(p, q, size_average=False)  / y.shape[0]
#     return l_kl

# if opt.dataset == 'MNIST':    
#     # Configure data loader   
#     data_test = MNIST(opt.data,
#                       train=False,
#                       transform=transforms.Compose([
#                           transforms.Resize((32, 32)),
#                           transforms.ToTensor(),
#                           transforms.Normalize((0.1307,), (0.3081,))
#                           ]))           
#     data_test_loader = DataLoader(data_test, batch_size=64, num_workers=1, shuffle=False)

#     # Optimizers
#     optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_G)
#     optimizer_S = torch.optim.Adam(net.parameters(), lr=opt.lr_S)

# if opt.dataset != 'MNIST':  
#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
#     if opt.dataset == 'cifar10': 
        
#         data_test = CIFAR10(opt.data,
#                           train=False,
#                           transform=transform_test)
#     if opt.dataset == 'cifar100': 
#         data_test = CIFAR100(opt.data,
#                           train=False,
#                           transform=transform_test)
#     data_test_loader = DataLoader(data_test, batch_size=opt.batch_size, num_workers=0)

#     # Optimizers
#     optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_G)

#     optimizer_S = torch.optim.SGD(net.parameters(), lr=opt.lr_S, momentum=0.9, weight_decay=5e-4)


# def adjust_learning_rate(optimizer, epoch, learing_rate):
#     if epoch < 800:
#         lr = learing_rate
#     elif epoch < 1600:
#         lr = 0.1*learing_rate
#     else:
#         lr = 0.01*learing_rate
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
        
        
# # ----------
# #  Training
# # ----------

# batches_done = 0
# for epoch in range(opt.n_epochs):

#     total_correct = 0
#     avg_loss = 0.0
#     if opt.dataset != 'MNIST':
#         adjust_learning_rate(optimizer_S, epoch, opt.lr_S)

#     for i in range(120):
#         net.train()
#         z = Variable(torch.randn(opt.batch_size, opt.latent_dim)).cuda()
#         optimizer_G.zero_grad()
#         optimizer_S.zero_grad()        
#         gen_imgs = generator(z)
#         outputs_T, features_T = teacher(gen_imgs, out_feature=True)   
#         pred = outputs_T.data.max(1)[1]
#         loss_activation = -features_T.abs().mean()
#         loss_one_hot = criterion(outputs_T,pred)
#         softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
#         loss_information_entropy = (softmax_o_T * torch.log(softmax_o_T)).sum()
#         loss = loss_one_hot * opt.oh + loss_information_entropy * opt.ie + loss_activation * opt.a
#         loss_kd = kdloss(net(gen_imgs.detach()), outputs_T.detach()) 
#         loss += loss_kd       
#         loss.backward()
#         optimizer_G.step()
#         optimizer_S.step() 
#         if i == 1:
#             print ("[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_a: %f] [loss_kd: %f]" % (epoch, opt.n_epochs,loss_one_hot.item(), loss_information_entropy.item(), loss_activation.item(), loss_kd.item()))
            
#     with torch.no_grad():
#         for i, (images, labels) in enumerate(data_test_loader):
#             images = images.cuda()
#             labels = labels.cuda()
#             net.eval()
#             output = net(images)
#             avg_loss += criterion(output, labels).sum()
#             pred = output.data.max(1)[1]
#             total_correct += pred.eq(labels.data.view_as(pred)).sum()

#     avg_loss /= len(data_test)
#     print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), float(total_correct) / len(data_test)))
#     accr = round(float(total_correct) / len(data_test), 4)
#     if accr > accr_best:
#         torch.save(net,opt.output_dir + 'student')
#         accr_best = accr
