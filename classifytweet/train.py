import os
import re
import json
import numpy as np
import pandas as pd
#import pickle

# coding: utf8
#from onnx_coreml import convert
#import onnx
# from tqdm import tqdm_notebook, tqdm
#import s3fs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as tfs
from torchvision.datasets import *
from torchvision.models import resnet18

from resolve import paths

def read_config_file(config_json):
    """
    This function reads in a json file like hyperparameters.json or resourceconfig.json
    :param config_json: this is a string path to the location of the file (for both sagemaker or local)
    :return: a python dict is returned
    """
    config_path = paths.config(config_json)
    if os.path.exists(config_path):
        json_data = open(config_path).read()
        return json.loads(json_data)


def entry_point():

#link s3 bucket here, put it in the function
#dataframe = pd.read_csv(paths.input(channel='training', filename="training.1600000.processed.noemoticon.csv"), error_bad_lines=False, encoding="ISO-8859-1", header=None).iloc[:, [0, 4, 5]].sample(frac=1).reset_index(drop=True)
    #datasource = "classifytweet/data/"
    #datasource =  paths.input(channel='training')
    #bucket = "ariane-test-c-cicd-pipeline-c6-input-data"
    #key = "input/data"
    #datasource = "s3://{}/{}/training/".format(bucket, key)
    #print(datasource)
    #epoch_step_2 = 3


    model_name = "modelName"
    epoch_step_1 = 2

    tr_tf = tfs.Compose([
        tfs.Resize(256),  # tfs.Resize(256),
        tfs.RandomCrop(224),  # tfs.CenterCrop(224)
        tfs.ColorJitter(0.2, 0.2, 0.2, 0.2),
        tfs.RandomHorizontalFlip(),
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    va_tf = tfs.Compose([
        tfs.Resize(224),
        tfs.CenterCrop(224),
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])


    tr_data = ImageFolder(datasource, transform=tr_tf)
    va_data = ImageFolder(datasource, transform=va_tf)
    print(tr_data.classes)
# TODO: Warum car & no-car jeweils 1000, sonst 700?
# AW PLP:

    np.random.seed(42)
    perm = np.random.permutation(len(tr_data))
    tr_size = int(len(tr_data) * 0.8)
    tr_sampler = SubsetRandomSampler(perm[:tr_size])
    va_sampler = SubsetRandomSampler(perm[tr_size:])
    tr_loader = DataLoader(tr_data, batch_size=64, sampler=tr_sampler)
    va_loader = DataLoader(va_data, batch_size=64, sampler=va_sampler)


    model = resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(tr_data.classes))



    def train(loader, model, criterion, optimizer):
        model.train()
        n_pred, n_true = 0, 0
        pbar = loader  # tqdm_notebook(loader) # WH
        for data in pbar:
            x, y = data
            optimizer.zero_grad()
            p = model(x)
            loss = criterion(p, y)
            n_pred += len(p)
            n_true += int(sum(torch.max(p, 1)[1] == y))
            text = 'Train: ' + str(round(1.0*n_true/n_pred*100, 1))+'%'
            print(text)
            # pBar.set_description(text)
            loss.backward()
            optimizer.step()
            # break


    def validate(loader, model, criterion):
        model.eval()
        n_pred, n_true = 0, 0
        pbar = loader  # tqdm_notebook(loader) # WH
        for data in pbar:
            x, y = data
            p = model(x)
            loss = criterion(p, y)
            n_pred += len(p)
            n_true += int(sum(torch.max(p, 1)[1] == y))
            text = 'Valid: ' + str(round(1.0*n_true/n_pred*100, 1))+'%'
            print(text)
            # pBar.set_description(text)
            # break


    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)



    for i in range(epoch_step_1):
        print('Epoch', i+1)
        train(tr_loader, model, criterion, optimizer)
        validate(va_loader, model, criterion)



    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)

    for i in range(epoch_step_2):
        print('Epoch', i+1)
        train(tr_loader, model, criterion, optimizer)
        validate(va_loader, model, criterion)


    """
    Saving
    """
    torch.save(model.state_dict(), model_name+'.pt')
    model.load_state_dict(torch.load(model_name+'.pt'))
    model.eval()

    '''
    img = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, img, model_name+".onnx",
                    input_names=['img'], output_names=['clf'])
    onnx_model = onnx.load(model_name+".onnx")


    scale = 1.0/(0.226*255)
    args = dict(
        is_bgr=False,
        red_bias=-(0.485*255) * scale,
        green_bias=-(0.456*255) * scale,
        blue_bias=-(0.406*255) * scale,
        image_scale=scale
    )



    coreml_model = convert(onnx_model, image_input_names=['img'], mode='classifier', preprocessing_args=args,
                        class_labels=tr_data.classes)  # ['car', 'golf', 'no-car', 'passat', 'polo', 'touareg', 'up'])

    coreml_model.author = 'LHIND'
    coreml_model.short_description = model_name
    coreml_model.input_description['img'] = 'Image'
    coreml_model.output_description['clf'] = 'Confidence and Label'
    coreml_model.output_description['classLabel'] = 'v'

    coreml_model.save(model_name+'.mlmodel')
'''
if __name__ == '__main__':
    entry_point()