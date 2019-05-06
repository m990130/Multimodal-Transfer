from collections import namedtuple

import torch
from torchvision import models
from helpers import *

'''
    class and function need to be used for loss caculation.
'''


# https://github.com/ceshine/fast-neural-style/blob/master/fast_neural_style/loss_network.py
# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg):
        super(LossNetwork, self).__init__()
        self.vgg = vgg
        self.layer_name_mapping = {
            '1': "relu1_1", '3': "relu1_2",
            '6': "relu2_1", '8': "relu2_2",
            '11': "relu3_1", '13': "relu3_2", '15': "relu3_3", '17': "relu3_4",
            '20': "relu4_1", '22': "relu4_2", '24': "relu4_3", '26': "relu4_4",
            '29': "relu5_1", '31': "relu5_2", '33': "relu5_3", '35': "relu5_4"
        }
        
        # extract different layers' activation by change the namedtuple below
        self.StyleOutput = namedtuple("StyleOutput", ["relu1_1", "relu2_1", "relu3_1", "relu4_1"])
        self.ContentOutput = namedtuple("ContentOutput", ["relu4_2"])
    
    def forward(self, x, mode):
        if mode == 'style':
            layers = ['1', '6', '11', '20']
        elif mode == 'content':
            layers = ['22']
        else:
            print("Invalid mode. Select between 'style' and 'content'")
        output = {}
        for name, module in self.vgg._modules.items():
            x = module(x)
            if name in layers:
                output[self.layer_name_mapping[name]] = x
        if mode == 'style':
            return self.StyleOutput(**output)
        else:
            return self.ContentOutput(**output)
        
def getLosses(generated_img, resized_input_img, content_weight, style_weight, loss_net, loss_func, gram_style):
    '''
        function wrap the calculation style and content loss during training. Since there are 3 stages, but only
        few thing changes between them.
        
        input: 
                generated_img: generated image from the current subnetwork.
                
                resized_input_img: resized raw input has the right resolution.
                
                content_weight, style_weight: hyper parameters need to specify before training.
                
                loss_net: loss network to be used. I use VGG19 here.
                
                loss_func: loss function to caculate the distance between feature/ gram matrix. 
                           I use torch.nn.MSELoss() here.
                
                gram_style: gram matrix of the style image with corresponding resolutino, which 
                            needs to be computed before training
                            
        output:
                content_loss and style_loss from current subnetwork.
    '''
    
    # function wrap the calculation style and content loss during training. Since there are 3 stages, but only
    # few thing changes between them.  
    
    
    # Compute features
    generated_style_features = loss_net(generated_img, 'style')
    #print(len(loss_network(generated_img, 'content')))
    generated_content_features = loss_net(generated_img, 'content')[0] # return a list containing 1 element
    with torch.no_grad():
        target_content_features = loss_net(resized_input_img, 'content')[0]
    
    # Content loss
    # generated_content_features and content_loss should require loss while target_content_features shouldn't
    content_loss = content_weight * loss_func(generated_content_features, target_content_features)

    
    # Style loss
    style_loss = 0.
    for m in range(len(generated_style_features)):
        gram_s = gram_style[m]
        gram_y = gram_matrix(generated_style_features[m])
        style_loss += style_weight * loss_func(gram_y, gram_s.expand_as(gram_y))
    
    return content_loss, style_loss




# class Vgg16(torch.nn.Module):
#     def __init__(self, requires_grad=False):
#         super(Vgg16, self).__init__()
#         vgg_pretrained_features = models.vgg16(pretrained=True).features
#         self.slice1 = torch.nn.Sequential()
#         self.slice2 = torch.nn.Sequential()
#         self.slice3 = torch.nn.Sequential()
#         self.slice4 = torch.nn.Sequential()
#         for x in range(4):
#             self.slice1.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(4, 9):
#             self.slice2.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(9, 16):
#             self.slice3.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(16, 23):
#             self.slice4.add_module(str(x), vgg_pretrained_features[x])
#         if not requires_grad:
#             for param in self.parameters():
#                 param.requires_grad = False

#     def forward(self, X):
#         h = self.slice1(X)
#         h_relu1_2 = h
#         h = self.slice2(h)
#         h_relu2_2 = h
#         h = self.slice3(h)
#         h_relu3_3 = h
#         h = self.slice4(h)
#         h_relu4_3 = h
#         vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
#         out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
#         return out
    
    
# class Vgg19(torch.nn.Module):
#     def __init__(self, requires_grad=False):
#         super(Vgg19, self).__init__()
#         vgg19_pretrained_features = models.vgg19(pretrained=True).features
        
#         self.slice1 = torch.nn.Sequential()
#         self.slice2 = torch.nn.Sequential()
#         self.slice3 = torch.nn.Sequential()
#         self.slice4 = torch.nn.Sequential()
#         self.slice5 = torch.nn.Sequential()
#         # relu1_1
#         for x in range(2):
#             self.slice1.add_module(str(x), vgg19_pretrained_features[x])
#         # relu2_1
#         for x in range(2, 7):
#             self.slice2.add_module(str(x), vgg19_pretrained_features[x])
#         # relu3_1
#         for x in range(7, 12):
#             self.slice3.add_module(str(x), vgg19_pretrained_features[x])
#         # relu4_1
#         for x in range(12, 21):
#             self.slice4.add_module(str(x), vgg19_pretrained_features[x])
#         # relu4_2
#         for x in range(21, 23):
#             self.slice5.add_module(str(x), vgg19_pretrained_features[x])
        
#         if not requires_grad:
#             for param in self.parameters():
#                 param.requires_grad = False
                
                
#     def forward(self, X):
#         h = self.slice1(X)
#         h_relu1_1 = h
#         h = self.slice2(h)
#         h_relu2_1 = h
#         h = self.slice3(h)
#         h_relu3_1 = h
#         h = self.slice4(h)
#         h_relu4_1 = h
#         h = self.slice5(h)
#         h_relu4_2 = h
        
#         vgg19_outputs = namedtuple("Vgg19Outputs", ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu4_2'])
#         out = vgg19_outputs(h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1, h_relu4_2)
#         return out