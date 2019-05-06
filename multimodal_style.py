# -*- coding: utf-8 -*-

import torch
from torchvision import transforms, models
from torch.optim import Adam
from torchvision import datasets
from torch.utils.data import DataLoader

import sys
import time
import argparse

from helpers import *
from style_subnet import Style_Subnet
from enhance_subnet import Enhance_Subnet
from refine_subnet import Refine_Subnet
from loss_utils import LossNetwork, getLosses
from collections import namedtuple

def train(args):
   
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print('using device:', device)
    
    
    LR = args.lr
    EPOCH = args.epochs
    content_weight = [1, 1, 1]
    style_weight = args.style_weight
    log_interval = args.log_interval
    batch_size= args.batch_size
    style_root = args.style_dir
    style_name = args.style_name
    data_root = args.content_dir
    
    print('training style:', style_name)
    
    # load training set
    content_trans = transforms.Compose([transforms.Resize(480),
                                        transforms.CenterCrop(480),
                                        transforms.Resize(256),
                                        transforms.ToTensor(),
                                        tensor_normalizer()])

    train_dataset = datasets.ImageFolder(root=data_root, transform=content_trans)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)    


    # # define the loss network
    print("Loading VGG..")
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    loss_network = LossNetwork(vgg).to(device).eval()
    del vgg


    # load style image

    style256 = load_image(style_root, device, 256)

    style512 = load_image(style_root, device, 512)

    style1024 = load_image(style_root, device, 1024)


    """ Before training, compute the features of every resolution of the style image """

    print("Computing style features..")
    with torch.no_grad(): 
        style_loss_features_256 = loss_network(style256, 'style')
        style_loss_features_512 = loss_network(style512, 'style')
        style_loss_features_1024 = loss_network(style1024, 'style')

    gram_style_256 = [gram_matrix(y) for y in style_loss_features_256]
    gram_style_512 = [gram_matrix(y) for y in style_loss_features_512]
    gram_style_1024 = [gram_matrix(y) for y in style_loss_features_1024]


    assert style_loss_features_256[0].requires_grad == False, "Style_features requires grad!"
    assert gram_style_256[0].requires_grad == False, "gram requires grad!"


    """ Prepare Training """

    mse_loss = torch.nn.MSELoss()
    max_iterations = min(args.maxiter, len(train_dataset))

    style_subnet = Style_Subnet().to(device)
    enhance_subnet = Enhance_Subnet().to(device)
    refine_subnet = Refine_Subnet().to(device)


    # init optimizer
    optimizer = torch.optim.Adam(list(style_subnet.parameters()) + 
                                 list(enhance_subnet.parameters()) +
                                 list(refine_subnet.parameters()), lr=LR)



    style_subnet.train().cuda()
    enhance_subnet.train().cuda()
    refine_subnet.train().cuda()


    print("start training with style weights: {}".format(style_weight))

    for epoch in range(EPOCH):
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0

        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch

            if (batch_id + 1) % 2000 == 0 and batch_id != 0:            
                LR = LR * 0.8
                print("---------------------LR decrease-----------------------")
                optimizer = Adam(list(style_subnet.parameters()) 
                                 + list(enhance_subnet.parameters()) 
                                 + list(refine_subnet.parameters()) , LR)

            optimizer.zero_grad()


            #--------------------------Style Subnet-------------------------#
            style_out, resized_img1 = style_subnet(x.to(device))

            s_content_loss, s_style_loss = getLosses(style_out, 
                                                     resized_img1,
                                                     content_weight[0], 
                                                     float(style_weight[0]),
                                                     loss_network,
                                                     mse_loss,
                                                     gram_style_256)

            #--------------------------Enhance Subnet-------------------------#
            enhance_out, resized_img2 = enhance_subnet(style_out)

            e_content_loss, e_style_loss = getLosses(enhance_out, 
                                                    resized_img2,
                                                    content_weight[1], 
                                                    float(style_weight[1]),
                                                    loss_network,
                                                    mse_loss,
                                                    gram_style_512)
            #--------------------------Refine Subnet--------------------------#
            refine_out, resized_img3 = refine_subnet(enhance_out)

            r_content_loss, r_style_loss = getLosses(refine_out, 
                                                     resized_img3,
                                                     content_weight[2], 
                                                     float(style_weight[2]),
                                                     loss_network,
                                                     mse_loss,
                                                     gram_style_1024) 

            #--------------------------Back Prop and update---------------------#



            total_loss = 1 * (s_content_loss + s_style_loss) \
                        + 0.5 * (e_content_loss + e_style_loss) \
                        + 0.25 * (r_content_loss + r_style_loss) 


            total_loss.backward()
            optimizer.step()

            agg_content_loss += (s_content_loss.item()+ e_content_loss.item() + r_content_loss.item())
            agg_style_loss += (s_style_loss.item() + e_style_loss.item() + r_style_loss.item())

            if (batch_id + 1) % log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.5f}\tstyle: {:.5f}\ttotal: {:.5f}".format(
                    time.ctime(), epoch + 1, count, len(train_loader.dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)
                #agg_content_loss, agg_style_loss = 0., 0.
                print('\ns_content:{:.5f}, s_style:{:.5f}, en_content:{:.5f}, en_style:{:.5f}, re_content:{:.5f}, re_style:{:.5f}\n'.format(
                         s_content_loss, s_style_loss, e_content_loss, e_style_loss, r_content_loss, r_style_loss))


            if (batch_id + 1) % 900 == 0:
                save_models('./CheckPoint/' + style_name, style_subnet, enhance_subnet, refine_subnet)

            #if part:
            if count > max_iterations:
                break


    save_models(style_name, style_subnet, enhance_subnet, refine_subnet)


def stylize(args):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    style_subnet = Style_Subnet().to(device).eval()
    enhance_subnet = Enhance_Subnet().to(device).eval()
    refine_subnet = Refine_Subnet().to(device).eval()
    
    style_subnet.load_state_dict(torch.load("pretrained_models/"+args.model_name+'_St.pt'))
    enhance_subnet.load_state_dict(torch.load("pretrained_models/"+args.model_name+'_En.pt'))
    refine_subnet.load_state_dict(torch.load("pretrained_models/"+args.model_name+'_Re.pt'))
    
    if args.single_image:
        
        with torch.no_grad():           
            x = load_image(args.content_dir, device)
            x = x.repeat(1, 1, 1, 1)
            # the second is the resized x

            output, _ = style_subnet(x.to(device))
            output, _ = enhance_subnet(output)
            output, _ = refine_subnet(output)

            imsave(output, args.result_dir+args.model_name+"/result.jpg")
        
    else:
        content_trans = transforms.Compose([#transforms.Resize(480),
                                    #transforms.CenterCrop(480),
                                    transforms.Resize(256),
                                    transforms.ToTensor(),
                                    tensor_normalizer()])

        dataset = datasets.ImageFolder(args.content_dir, transform=content_trans)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                
                output, _ = style_subnet(x.to(device))
                output, _ = enhance_subnet(output)
                output, _ = refine_subnet(output)

                imsave(output, args.result_dir+args.model_name+"/"+str(i)+".jpg")
        


def main():
    
    main_arg_parser = argparse.ArgumentParser(description="parser for MultiModal-Style-Transfer")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=1,
                                  help="number of training epochs, default is 1")
    train_arg_parser.add_argument("--maxiter", type=int, default=8000,
                                  help="maximum iterations per epoch, default is 8000")
    train_arg_parser.add_argument("--batch-size", type=int, default=1,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--content-dir", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-dir", type=str, required=True,
                                  help="path to style-image")
    train_arg_parser.add_argument("--style-name", type=str, required=True,
                                  help="prefix of your model name")
    train_arg_parser.add_argument("--style-weight", nargs='+', default=[6e5*0.2, 6e5*4*0.1, 6e4*0.3],
                                  help="weights for style-loss of 3 sub networks, ")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")    
    train_arg_parser.add_argument("--log-interval", type=int, default=200,
                                  help="number of images after which the training loss is logged, default is 500")


    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--single-image", type=int, default=1,
                                 help="stylize single image or a image folder")
    eval_arg_parser.add_argument("--content-dir", type=str, required=True,
                                 help="path to content images you want to stylize")
    eval_arg_parser.add_argument("--result-dir", type=str, default="stylized_imgs/",
                                 help="path to stylized image, make sure the path exists")
    eval_arg_parser.add_argument("--model-name", type=str, default="marc",
                                 help="name of model to be used for stylizing the images.")


    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)

    if args.subcommand == "train":
        torch.cuda.empty_cache()
        train(args)
    else:
        stylize(args)
    

if __name__ == "__main__":
    main()
    
    





