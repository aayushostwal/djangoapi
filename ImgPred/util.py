import configparser
import os
import time
import urllib
import urllib.request

import torch
from PIL import Image
from django import forms
from torchvision import transforms

from django.http import HttpResponse
from django.shortcuts import render

config = configparser.ConfigParser()
config.read('config')


class InputFile(forms.Form):
    file = forms.FileField()


class InputURL(forms.Form):
    url = forms.URLField()


def save_input_file(f):
    # will save the uploaded image in a specific directory
    out_path = config['PATHS']['out_path']
    os.makedirs(out_path, exist_ok=True)
    out_file = os.path.join(out_path, "image_{}.{}".format(str(time.time()).split(".")[0], f.name.split(".")[-1]))
    with open(out_file, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return out_file


def save_input_url(url):
    out_path = config['PATHS']['out_path']
    os.makedirs(out_path, exist_ok=True)
    out_file = os.path.join(out_path, "image_{}.{}".format(str(time.time()).split(".")[0], 'jpeg'))
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent',
                          'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, out_file)
    return out_file


def predict_image(filepath):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.eval()
    input_image = Image.open(filepath)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    with open(config['PATHS']['classes'], "r") as f:
        categories = [s.strip() for s in f.readlines()]
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    out = {}
    for i in range(top5_prob.size(0)):
        out[categories[top5_catid[i]]] = top5_prob[i].item()
    return out
