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
parser.add_argument('--depth', type=int, default=1, help='Depth till which parameters should be halved')
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
parser.add_argument('--student', type=str, default='vgg16', choices=['resnet18','resnet34','resnet50','resnet101','resnet152','vgg16'], help='Base student model for IKD')

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

generator = Generator(args.channels,args.img_size,args.latent_dim).to(device)
criterion = nn.CrossEntropyLoss().to(device)
original_params = [64,64,128,256,512]
if args.dataset == 'MNIST' :
    teacher = resnet.ResNet50(original_params,num_classes=10).to(device)
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
    
elif args.dataset == 'cifar10' :
    teacher = resnet.ResNet50(original_params,num_classes=10).to(device)
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

elif args.dataset == 'cifar100' :
    teacher = resnet.ResNet50(original_params,num_classes=100).to(device)
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


original_parameters = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
classifier_parameters = [512,4096,4096]
cnt = 0

#net = vgg16.vgg16(original_parameters,classifier_parameters).to(device)
#net = resnet.ResNet18(num_classes=10).to(device)
#optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)


for depth in range(args.depth):
    last_number = utils.get_output_nodes(original_parameters)
    classifier_params = [last_number,4096,4096]
    if depth == 0:
        print("Original network: ")

        # TODO
        original_parameters = [64,64,128,256,512]

        output_model = utils.get_model(args.student,original_parameters)
        # If model is vgg
        if(output_model == 0):
            original_parameters = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
            classifier_parameters = [512,4096,4096]
            net = vgg16.vgg16(original_parameters,classifier_parameters).to(device)
        #If model is resnet
        else:
            net = output_model

        # print(net)
        optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)
        accr_best, cnt = utils.train_model(net, teacher, generator, data_test_loader, device, criterion, optimizer_G, optimizer_S, args.lr_G, args.lr_S, args.oh, args.ie, args.a, args.batch_size, args.img_size, args.latent_dim, args.n_epochs, args.dataset, cnt)
        print()
        print("Best accuracy currently : {}".format(accr_best))
        print("\n############################################################################\n")
    else:
        print("At depth " + str(depth) + ": ")
        original_parameters = [int(i/2) for i in original_parameters]
        print(original_parameters)
        output_model = utils.get_model(args.student,original_parameters)
        # If model is vgg
        if(output_model == 0):
            classifier_parameters = [int(i/2) if type(i) == int else i for i in classifier_parameters]
            net = vgg16.vgg16(original_parameters,classifier_parameters).to(device)
        #If model is resnet
        else:
            net = output_model
        # print(net)
        optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)
        accr_best, cnt = utils.train_model(net, teacher, generator, data_test_loader, device, criterion, optimizer_G, optimizer_S, args.lr_G, args.lr_S, args.oh, args.ie, args.a, args.batch_size, args.img_size, args.latent_dim, args.n_epochs, args.dataset, cnt)
        print()
        print("Best accuracy currently : {}".format(accr_best))
        print("\n############################################################################\n")
'''
    # For odd case :
    elif depth % 2 == 1:
        print("At depth " + str(depth) + ": ")
        print("Dividing filters by 2: ")
        original_parameters = [int(i/2) if type(i) == int else i for i in original_parameters]
        classifier_parameters = [int(i/2) if type(i) == int else i for i in classifier_parameters]
        print(original_parameters)
        output_model = utils.get_model(args.student)
        # If model is vgg
        if(output_model == 0):
            net = vgg16.vgg16(original_parameters,classifier_parameters).to(device)
        #If model is resnet
        else:
            net = output_model
        last_layer = [net.features[i] for i in range(len(net.features))][-1]
        if str(last_layer) ==  'ReLU(inplace=True)':
            net.features = nn.Sequential(*[net.features[i] for i in range(len(net.features) -1)])
        # print(net)
        optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)
#         net, hist = train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=EPOCHS)
        accr_best, cnt = utils.train_model(net, teacher, generator, data_test_loader, device, criterion, optimizer_G, optimizer_S, args.lr_G, args.lr_S, args.oh, args.ie, args.a, args.batch_size, args.img_size, args.latent_dim, args.n_epochs, args.dataset, cnt)
        print()
        print("Best accuracy currently : {}".format(accr_best))
        print("\n############################################################################\n")

    # For even case :
    elif depth % 2 == 0:
        print("At depth " + str(depth) + ": ")
        print("Removing last layer: ")
        output_model = utils.get_model(args.student)
        # If model is vgg
        if(output_model == 0):
            net = vgg16.vgg16(original_parameters,classifier_parameters).to(device)
        #If model is resnet
        else:
            net = output_model
        second_last_layer = [net.features[i] for i in range(len(net.features)-1)][-1]
        if str(second_last_layer) ==  'ReLU(inplace=True)':
            net.features = nn.Sequential(*[net.features[i] for i in range(len(net.features) -2)])
        else:
            net.features = nn.Sequential(*[net.features[i] for i in range(len(net.features) -1)])
        original_parameters = original_parameters[:-1]
        # print(net)
        optimizer_S = torch.optim.SGD(net.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)
#         net, hist = train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=EPOCHS)
        accr_best, cnt = utils.train_model(net, teacher, generator, data_test_loader, device, criterion, optimizer_G, optimizer_S, args.lr_G, args.lr_S, args.oh, args.ie, args.a, args.batch_size, args.img_size, args.latent_dim, args.n_epochs, args.dataset, cnt)
        print()
        print("Best accuracy currently : {}".format(accr_best))
        print("\n############################################################################\n")
'''
