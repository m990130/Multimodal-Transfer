# Multimodal-Transfer

## Description
This depo contains a pytorch implemantation of MultiModal Style Transfer, see http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_Multimodal_Transfer_A_CVPR_2017_paper.pdf

Some pretrained models and their correspoinding result are in *pretrained_models/* and *stylized_imgs/* respectively.

## Stylize image
To stylize the image, you need a trained model(contains 3 subnetwork: style-, enhance- and refine network), single image or a image folder for content image. There are several pretrained models in folder *pretrained_models/*. The subnetworks belonging to the same style are named with same prefix, e.g. **Pollock_St.pt, Pollock_En.pt and Pollock_Re.pt** represent the three subnetworks mentioned above, that trained form the painting *ref_style_images/Pollock-number-one-moma-November-31-1950-1950.jpg*.


**!!! plz make sure all the directories already exsist before you run it!!!**
### To stylize a single image
use **single-image = 1** to specify only stylize a single image. **model-name** is the prefix in *pretrained_models/*, e.g. **model-name = Pollock**. The stylized image will be stored as *result-dir/model-name/result.jpg*
```bash
$ python multimodal_style.py eval \
  --single-image 1 \
  --content-dir path/to/<content_image.jpg> \
  --result-dir path/to/<stylized_img.jpg>
  --model-name <model_prefix_in_pretrained_model>
```
### To stylize more than one images
use **single-image = 0** to specify this setting. Another difference is to set the **content-dir** as the path to the folder containing another subfolder, where the content images are, e.g. you want to stylize all images in *stylized_imgs/0_Original/image_folder*, then you need to set the **content-dir** as *stylized_imgs/0_Original/*. This is a bit unconvenient :D

In this mode, the stylized images will be stored as *result-dir/model-name/str(i).jpg*, where i is the index of the image in DataLoader.


## Training a model
For training I used the COCO dataset, with **epoch=1** and **maxiter=8000**, so the model basically got trained on 8000 images from COCO. The training took around 2.5 hours in such setting. Of course, the epoch and maxiter (how many iterations per epoch, or how many images were seen during each epoch) counld be larger, but I found that most of the style after 8000 iterations didn't change much. Maybe a better way to is to change the parameters in **style-weight**, which needs a lot of adjustment.
```bash
$ python train.py train \
  --epochs 1 \
  --maxiter 8000 \
  --batch-size 1 \
  --content-dir path/to/training_data \  # should be path to a folder containing a subfolder 
  --style-dir ref_style_images/Dali-illumined-pleasure-1929.jpg \
  --style-name Dali \  # the trained model will be saved with this prefix
  --style-weight 120000 240000 18000 \  # should be 3 interger with space in between here
  --lr 0.001 \
  --log-interval 200
```
