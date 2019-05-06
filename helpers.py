# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 00:31:43 2018

@author: pingc
"""
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def gram_matrix(y):
    # The input of tensor has size of (batch_size, channels, height, width),
    # need to be resized before compute gram matrix. And then 
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, h * w)
    # swap channel and h*w
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    # The gram represents the correlation between channels -- features
    return gram

def load_image(filename, device, size=256):
    # load single image with given size and scale factor
    
    
#     content_trans = transforms.Compose([transforms.Resize(480),
#                                     transforms.CenterCrop(480),
#                                     transforms.Resize(size),
#                                     transforms.ToTensor(),
#                                     tensor_normalizer()])
    content_trans = transforms.Compose([transforms.Resize(size),
                                    transforms.ToTensor(),
                                    tensor_normalizer()])
    
    image = Image.open(filename).convert('RGB')
    image = content_trans(image).unsqueeze(0).to(device)
    
    return image


def gram_style(style_image, vgg):
    # input should be output of load_image
    features = vgg(style_image)
    gram = [gram_matrix(y) for y in features]
    
    return gram


def show_tensor(img):
    # a handy function to show tensor image
    if img.device.type == 'cuda': img = img.cpu()
    if img.requires_grad == True: img = img.detach()
    if img.ndimension() == 4: npimg = img.squeeze().numpy()
    else: npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    

def imsave(stylized_img, filename):
    
    image = stylized_img.cpu().clone()  
    image = image.squeeze(0)      
    denormalizer = tensor_denormalizer()
    image = denormalizer(image)
    image.data.clamp_(0, 1)
    toPIL = transforms.ToPILImage()
    image = toPIL(image)
    
    image.save(filename)

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    denormalizer = tensor_denormalizer()
    image = denormalizer(image)
    image.data.clamp_(0, 1)
    toPIL = transforms.ToPILImage()
    image = toPIL(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated
    
def normalize_batch(batch):
    # From Johnson's code
    # normalize using ImageNet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

def tensor_normalizer():
    #print("use normalizer")
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def tensor_denormalizer():
    return transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                               transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])

def save_models(style_name, style_subnet, enhance_subnet, refine_subnet):
    print("Saving models...")
    style_subnet.eval()
    enhance_subnet.eval()
    refine_subnet.eval()

    save_model_filename1 = 'saved_models/' + style_name + '_St.pt'
    save_model_filename2 = 'saved_models/' + style_name + '_En.pt'
    save_model_filename3 = 'saved_models/' +style_name + '_Re.pt'

    torch.save(style_subnet.state_dict(), save_model_filename1)
    torch.save(enhance_subnet.state_dict(), save_model_filename2)  
    torch.save(refine_subnet.state_dict(), save_model_filename3)
    style_subnet.train()
    enhance_subnet.train()
    refine_subnet.train()

# def subnet_loss19(stylized_img, resized_img, gram_style, content_weight, style_weight, loss_net, loss_func, device):
        
#         feat_stylize_img = loss_net(stylized_img.to(device))
#         feat_resized_img= loss_net(resized_img.to(device))
#         # content loss
#         content_loss = content_weight * loss_func(feat_stylize_img.relu4_2, feat_resized_img.relu4_2)
        
#         # style_loss
#         style_loss = 0.
        
#         for i in range(0,4):
#             gm_y = gram_matrix(feat_stylize_img[i])
#             gm_s = gram_style[i]
#             style_loss += loss_func(gm_y, gm_s[:len(stylized_img), :, :])
        
        
#         style_loss *= style_weight
        
        
#         return content_loss, style_loss
    
# def subnet_loss16(stylized_img, resized_img, gram_style, content_weight, style_weight, loss_net, loss_func, device):
    
#         feat_stylize_img = loss_net(stylized_img.to(device))
#         feat_resized_img= loss_net(resized_img.to(device))
#         # content loss
#         content_loss = content_weight * loss_func(feat_stylize_img.relu2_2, feat_resized_img.relu2_2)
        
#         # style_loss
#         style_loss = 0.
#         for ft_y, gm_s in zip(feat_stylize_img, gram_style):
#             gm_y = gram_matrix(ft_y)
#             style_loss += loss_func(gm_y, gm_s[:len(stylized_img), :, :])
                
#         style_loss *= style_weight
        

        
#         return content_loss, style_loss
        
