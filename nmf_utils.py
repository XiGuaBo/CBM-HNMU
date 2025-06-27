import numpy as np
import cv2
import torch
import torch.nn as nn

from torchvision import transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from torchsummary import summary

from PIL import Image

import matplotlib.pyplot as plt

from craft.craft_torch import Craft, torch_to_numpy
from math import ceil

import shutil, os, sys

# nb_crops = 10

to_pil = transforms.ToPILImage()

def show(img, **kwargs):
  img = np.array(img)
  if img.shape[0] == 3:
    img = img.transpose(1, 2, 0)

  img -= img.min();img /= img.max()
  plt.imshow(img, **kwargs); plt.axis('off')

'''
def NMF_EXTRACTOR(model, device = 'cuda', dataset="flower102", target_cls=0):
  # model info
  # print (model.fe.pretrained_cfg)
  # summary(model,(3,256,256))
  # processing
  config = resolve_data_config({}, model=model.fe)
  transform = create_transform(**config)
  # cut the model in two
  g = nn.Sequential(*(list(model.children())[:3])) # input to penultimate layer
  h = lambda x: model.header(x) # penultimate layer to logits
  # loading some images of alpine sea holly_0 !
  import json
  with open("Dataset/{}/idx2class.json".format(dataset),'r') as f:
    idx2class = json.loads(f.read())
  f.close()
  assert str(target_cls) in idx2class.keys(), "Target_cls idx is invalid !"
  target_cls = str(target_cls)
  images = np.load('excute/{}/{}_{}.npz'.format(dataset,idx2class[target_cls],target_cls))['arr_0'].astype(np.uint8)
  images = np.transpose(images,(0,2,3,1))
  craft = Craft(input_to_latent = g,
              latent_to_logit = h,
              number_of_concepts = 10,
              patch_size = 64,
              batch_size = 64,
              device = device)
  images_preprocessed = torch.stack([transform(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))) for img in images], 0)
  print (images_preprocessed.shape)
  crops, crops_u, w = craft.fit(images_preprocessed)
  crops = np.moveaxis(torch_to_numpy(crops), 1, -1)
  
  print (crops.shape, crops_u.shape, w.shape)
  return crops, crops_u, w, craft, transform

def NMF_EXCUTOR(crops, crops_u, craft, ts_image, target_cls, transform, dataset="flower102", idx=0, tar_zone="test", concepts_out=True): # @ ts_image is cv::Mat
  if not (len(ts_image.shape)==4):
      ts_image = np.expand_dims(ts_image,axis=0)
  ts_image = ts_image.astype(np.uint8)
  ts_images_preprocessed = torch.stack([transform(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))) for img in ts_image ], 0)
  importances = craft.estimate_importance(ts_images_preprocessed, class_id=target_cls) # 0 is the alpine sea holly_0 class id in flower102
  # images_u = craft.transform(ts_images_preprocessed)
  
  # print (images_u.shape)
  # top-5 importance concepts
  most_important_concepts = np.argsort(importances)[::-1][:5]
  for c_id in most_important_concepts:
    print("Concept", c_id, " has an importance value of ", importances[c_id])
  
  # normalize for importance
  importances = importances / np.sum(importances)
  
  best_crops = []
  
  for i, c_id in enumerate(most_important_concepts):
    best_crops_id = np.argsort(crops_u[:, c_id])[::-1][0]
    best_crop = crops[best_crops_id]
    best_crops.append(best_crop)
    if (concepts_out):
      plt.subplot(2, len(most_important_concepts), len(most_important_concepts) + i + 1)
      show(best_crop)
      plt.title("c_id:{}\nscore:{:.2f}".format(c_id,importances[c_id]))
  if (concepts_out):
    plt.subplot(2, len(most_important_concepts), 1)
    ts_image = cv2.cvtColor(ts_image.squeeze().astype(np.float32),cv2.COLOR_BGR2RGB)
    show(ts_image)
    import json
    with open("Dataset/{}/idx2class.json".format(dataset),'r') as f:
      idx2class = json.loads(f.read())
    f.close()
    plt.title("input\n{}".format(idx2class[str(target_cls)]))
    # shutil.rmtree("output/{}".format(dataset),ignore_errors=True)
    if not (os.path.exists("output/{}/{}".format(dataset,tar_zone))):
      os.makedirs("output/{}/{}".format(dataset,tar_zone),exist_ok=True)
    if not (os.path.exists("output/{}/{}/s{}_ex_concepts.png".format(dataset,tar_zone,idx))):
      plt.savefig("output/{}/{}/s{}_ex_concepts.png".format(dataset,tar_zone,idx))
  return np.array(best_crops), importances[most_important_concepts]
'''

# '''
from matplotlib.colors import ListedColormap
import matplotlib
import colorsys
def get_alpha_cmap(cmap):
  if isinstance(cmap, str):
    cmap = plt.get_cmap(cmap)
  else:
    c = np.array((cmap[0]/255.0, cmap[1]/255.0, cmap[2]/255.0))

    cmax = colorsys.rgb_to_hls(*c)
    cmax = np.array(cmax)
    cmax[-1] = 1.0

    cmax = np.clip(np.array(colorsys.hls_to_rgb(*cmax)), 0, 1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [c,cmax])

  alpha_cmap = cmap(np.arange(256))
  alpha_cmap[:,-1] = np.linspace(0, 0.85, 256)
  alpha_cmap = ListedColormap(alpha_cmap)

  return alpha_cmap

cmaps = [
  get_alpha_cmap((54, 197, 240)),
  get_alpha_cmap((210, 40, 95)),
  get_alpha_cmap((236, 178, 46)),
  get_alpha_cmap((15, 157, 88)),
  get_alpha_cmap((84, 25, 85))
]

def NMF_EXTRACTOR(model, device = 'cuda', dataset="flower102", arch="nfresnet50", target_cls=0):
  # model info
  # print (model.fe.pretrained_cfg)
  # summary(model,(3,256,256))
  # processing
  config = resolve_data_config({}, model=model.fe)
  transform = create_transform(**config)
  # cut the model in two
  g = nn.Sequential(*(list(model.children())[:3])) # input to penultimate layer
  h = lambda x: model.header(x) # penultimate layer to logits
  # loading some images of alpine sea holly_0 !
  import json
  with open("Dataset/{}/idx2class.json".format(dataset),'r') as f:
    idx2class = json.loads(f.read())
  f.close()
  assert str(target_cls) in idx2class.keys(), "Target_cls idx is invalid !"
  target_cls = str(target_cls)
  images = np.load('excute/{}/{}/{}_{}.npz'.format(arch,dataset,idx2class[target_cls].replace('/', '|'),target_cls))['arr_0'].astype(np.uint8)
  images = np.transpose(images,(0,2,3,1))
  craft = Craft(input_to_latent = g,
              latent_to_logit = h,
              number_of_concepts = 10,
              patch_size = 64,
              batch_size = 64,
              device = device)
  # images_preprocessed = torch.stack([transform(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))) for img in images], 0)
  images_preprocessed = torch.stack([torch.tensor(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),dtype=torch.float32) for img in images], 0)
  images_preprocessed = images_preprocessed.permute(0,3,1,2)
  print (images_preprocessed.shape)
  crops, crops_u, w = craft.fit(images_preprocessed)
  crops = np.moveaxis(torch_to_numpy(crops), 1, -1)
  
  print (crops.shape, crops_u.shape, w.shape)
  return crops, crops_u, w, craft, transform

def NMF_EXCUTOR(crops, crops_u, craft, ts_image, target_cls, transform, dataset="flower102", idx=0, tar_zone="test", concepts_out=True, fg="bb"): # @ ts_image is cv::Mat
  if not (len(ts_image.shape)==4):
      ts_image = np.expand_dims(ts_image,axis=0)
  ts_image = ts_image.astype(np.uint8)
  # ts_images_preprocessed = torch.stack([transform(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))) for img in ts_image ], 0)
  ts_images_preprocessed = torch.stack([torch.tensor(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),dtype=torch.float32) for img in ts_image ], 0)
  ts_images_preprocessed = ts_images_preprocessed.permute(0,3,1,2)
  importances = craft.estimate_importance(ts_images_preprocessed, class_id=target_cls) # 0 is the alpine sea holly_0 class id in flower102
  
  # concepts attr needed
  images_u = craft.transform(ts_images_preprocessed)
  print (images_u.shape)
  # concepts attr needed

  # top-5 importance concepts
  most_important_concepts = np.argsort(importances)[::-1][:5]
  for c_id in most_important_concepts:
    print("Concept", c_id, " has an importance value of ", importances[c_id])
  
  # normalize for importance
  importances = importances / np.sum(importances)
  
  best_crops = []
  
  for i, c_id in enumerate(most_important_concepts):
    best_crops_id = np.argsort(crops_u[:, c_id])[::-1][0]
    best_crop = crops[best_crops_id]
    best_crops.append(best_crop)
    if (concepts_out):
      plt.subplot(2, len(most_important_concepts), len(most_important_concepts) + i + 1)
      
      # concepts attr needed
      cmap = cmaps[i]
      p = 5
      mask = np.zeros(best_crop.shape[:-1])
      mask[:p, :] = 1.0
      mask[:, :p] = 1.0
      mask[-p:, :] = 1.0
      mask[:, -p:] = 1.0
      # concepts attr needed
      
      show(best_crop)
      show(mask, cmap=cmap)
      plt.title("c_id:{}\nscore:{:.2f}".format(c_id,importances[c_id]))
  if (concepts_out):
    plt.subplot(2, len(most_important_concepts), 1)
    ts_image = cv2.cvtColor(ts_image.squeeze().astype(np.float32),cv2.COLOR_BGR2RGB)
    show(ts_image)
    
    # concepts attr needed
    for i, c_id in enumerate(most_important_concepts):

      cmap = cmaps[i]
      heatmap = images_u[0,:, :, c_id]

      # sigma = np.percentile(images_u[:,:,:,c_id].flatten(), 20)
      # heatmap = heatmap * np.array(heatmap > sigma, np.float32)
      heatmap = cv2.resize(heatmap[:, :, None], (224, 224))
      show(heatmap, cmap=cmap, alpha=0.7)
    # concepts attr needed
    
    import json
    with open("Dataset/{}/idx2class.json".format(dataset),'r') as f:
      idx2class = json.loads(f.read())
    f.close()
    plt.title("input\n{}".format(idx2class[str(target_cls)]))
    # shutil.rmtree("output/{}".format(dataset),ignore_errors=True)
    if not (os.path.exists("output/{}/{}/{}".format(dataset,tar_zone,fg))):
      os.makedirs("output/{}/{}/{}".format(dataset,tar_zone,fg),exist_ok=True)
    plt.savefig("output/{}/{}/{}/s{}_ex_concepts.png".format(dataset,tar_zone,fg,idx))
  return np.array(best_crops), importances[most_important_concepts]
  # '''