import torch
from torch import nn
import numpy as np
import pickle

import os,sys,pathlib
import matplotlib.pyplot as plt
import cv2 as cv

import shutil
from network import *
import json

# net switch : ["nfresnet50" , "vit" , "resnext26" , "botnet26t" , "rexnet100" , "gcvit" , "deit" , "convit" , "cait"]
net_sw = sys.argv[1]
# dataset switch : ["flower102" , "cifar10" , "cifar100" , "cub" , "aircraft"]
dataset_sw = sys.argv[2]

class_nums_dict = {"flower102":102 , "cifar10":10 , "cifar100":100 , "cub":200 , "aircraft":100}
df_dict = {"flower102":".jpg" , "cifar10":"", "cifar100":"" , "cub":"" , "aircraft":".jpg"}

if (len(sys.argv)<3):
    dataset_sw = "flower102"
    
if (len(sys.argv)>=4):
    data_root = sys.argv[3]
else:
    data_root = "Dataset"

if (net_sw == "nfresnet50"):
    model = NFRES50(class_nums_dict[dataset_sw])
    sample_size = 256
elif (net_sw == "resnext26"):
    model = RESNEXT26(class_nums_dict[dataset_sw])
    sample_size = 256
elif (net_sw == "botnet26t"):
    model = BOTNET26t(class_nums_dict[dataset_sw])
    sample_size = 256
elif (net_sw == "rexnet100"):
    model = REXNET100(class_nums_dict[dataset_sw])
    sample_size = 256
elif (net_sw == "vit"):
    model = Vit(class_nums_dict[dataset_sw])
    sample_size = 224
elif (net_sw == "gcvit"):
    model = GCVIT(class_nums_dict[dataset_sw])
    sample_size = 224
elif (net_sw == "cait"):
    model = Cait_Small(class_nums_dict[dataset_sw])
    sample_size = 224
elif (net_sw == "convit"):
    model = Convit_Base(class_nums_dict[dataset_sw])
    sample_size = 224
elif (net_sw == "deit"):
    model = Deit_Base(class_nums_dict[dataset_sw])
    sample_size = 224
else:
    # no choice will switch to nfresnet50 and flower102
    model = NFRES50(class_nums_dict[dataset_sw])
    sample_size = 256
    
batch_size = 16
seed = 20240403 # just random shuffle the data in corresponding dataset

def DataLoader(data_format=df_dict[dataset_sw],zone=["train","val","test"],data_root=data_root):
    ds_root = "{}/{}".format(data_root,dataset_sw)
    ds_images_root = os.path.join(ds_root,"images")
    splits_path = os.path.join(ds_root,"splits")
    ds_splits = {}
    splits = {}
    class_dict = {}
    dict_class = {}
    for dt in zone:
        splits[dt] = {}
        with open(os.path.join(splits_path,"class2images_{}.p").format(dt),"rb") as f:
            ds_splits[dt] = pickle.load(f)
        cls_list = sorted(ds_splits[dt],key=lambda x : x)
        for idx,cls in enumerate(cls_list):
            splits[dt][cls] = ds_splits[dt][cls]
            class_dict[cls] = idx
            dict_class[idx] = cls
            
    ds_images = {}
    ds_images_per_cls = {}
    ds_lables = {}
    for dt in zone:
        ds_images[dt] = []
        ds_images_per_cls[dt] = {}
        ds_lables[dt] = []
        for cls in splits[dt].keys():
            ds_images_per_cls[dt][cls] = np.array([cv.resize(cv.imread(os.path.join(ds_images_root,img_name+data_format),cv.IMREAD_COLOR),(sample_size,sample_size)) for img_name in splits[dt][cls]]).squeeze()
            ds_images[dt].extend(ds_images_per_cls[dt][cls])
            # print (ds_images_per_cls[dt][cls].shape)
            # lv = np.zeros((ds_images_per_cls[dt][cls].shape[0],len(cls_list)),dtype=np.int64)
            # lv[:,class_dict[cls]] = 1
            lv = np.array([class_dict[cls]],dtype=np.int64).repeat(ds_images_per_cls[dt][cls].shape[0])
            # print (lv.shape)
            ds_lables[dt].extend(lv)
            
            # print ("samples num",ds_images_per_cls[dt][cls].shape[0])
            # print ("true:",cls)
            # print ("label:",dict_class[np.argmax(lv)])
            # print ("label:",dict_class[lv[0]])
            # print ("label nums:",len(lv))
    
        ds_images[dt] = np.array(ds_images[dt],dtype=np.float32)
        ds_lables[dt] = np.array(ds_lables[dt],dtype=np.int64)
        
        np.random.seed(seed=seed)
        s = np.random.get_state()
        np.random.shuffle(ds_images[dt])
        np.random.set_state(s)
        np.random.shuffle(ds_lables[dt])
        
        print (ds_images[dt].shape)
        print (ds_lables[dt].shape)
    return ds_images , ds_lables , class_dict , dict_class # , ds_images_per_cls

def val(ds,model):
    size = len(ds)
    loss_fn = nn.CrossEntropyLoss()
    model.eval().to("cuda")
    test_loss, correct = 0, 0
    n = 0
    current = 0
    P_CLS = np.zeros((class_nums_dict[dataset_sw],))
    P_CLS_A = np.zeros((class_nums_dict[dataset_sw],))
    with torch.no_grad():
        for batch, (X, y) in enumerate(ds):
            X, y = torch.tensor(X).to('cuda'), torch.tensor(y).to('cuda')
            X = X.permute(0, 3, 1, 2)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            for i,p in enumerate((pred.argmax(1) == y).type(torch.int)):
                if (p==1):
                    P_CLS[y[i]] += 1
                    correct += 1
                P_CLS_A[y[i]] += 1
            n += len(X)
            current += len(X)
            print(f"Val loss: {test_loss:>7f} ACC: {correct/n:>7f} [current: {current} / - ]")
    
    print ("ACC FOR CLASS :",P_CLS/P_CLS_A)

def torch_to_numpy(tensor):
  try:
    return tensor.detach().cpu().numpy()
  except:
    return np.array(tensor)

def val_collect_ps(ds,model):
    size = len(ds)
    loss_fn = nn.CrossEntropyLoss()
    model.eval().to("cuda")
    test_loss, correct = 0, 0
    n = 0
    current = 0
    ps_per_cls = {}
    with torch.no_grad():
        for batch, (X, y) in enumerate(ds):
            X, y = torch.tensor(X).to('cuda'), torch.tensor(y).to('cuda')
            X = X.permute(0, 3, 1, 2)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred = torch_to_numpy(pred.argmax(1))
            y = torch_to_numpy(y)
            X = torch_to_numpy(X)
            for i,p in enumerate(pred):
                if p==y[i]:
                    correct += 1
                    if not (dc[y[i]] in ps_per_cls.keys()):
                        ps_per_cls[dc[y[i]]] = [X[i]]
                    else:
                        ps_per_cls[dc[y[i]]].append(X[i])
                    
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            n += len(X)
            current += len(X)
            print(f"Val loss: {test_loss:>7f} ACC: {correct/n:>7f} [current: {current} / - ]")
    
    os.makedirs("excute/{}/{}".format(net_sw,dataset_sw),exist_ok=True)
    for cls in ps_per_cls.keys():
        out = np.array(ps_per_cls[cls])
        print (out.shape)
        np.savez("excute/{}/{}/{}_{}.npz".format(net_sw,dataset_sw,cls,cd[cls]),out)

def train(ds,model,epochs):
    size = len(ds['train'])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train().to("cuda")
    global_loss = 99999999999.
    for ep in range(epochs):
        mean_loss = 0.
        current = 0
        for batch, (X, y) in enumerate(ds['train']):
            # print (X.shape)
            # print (y.shape)
            X, y = torch.tensor(X).to('cuda'), torch.tensor(y).to('cuda')
            X = X.permute(0, 3, 1, 2)
            # exit()
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            mean_loss += loss/size
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # if batch % 100 == 0:
            loss = loss.item()
            current += len(X)
            mean_loss = mean_loss.item()
            print(f"loss: {loss:>7f} mloss: {mean_loss:>7f}  [{current:>5d}/ - ]")
            
        os.makedirs("weights/{}/".format(dataset_sw),exist_ok=True)
        if (global_loss > mean_loss):
            torch.save(model.state_dict(),"weights/{}/latest_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw))
            global_loss = mean_loss
        
        val(ds['val'],model)


if __name__ == "__main__":
    
    imgs,labs,cd,dc = DataLoader()
    
    with open("Dataset/{}/class2idx.json".format(dataset_sw),'w+') as f:
        f.write(json.dumps(cd))
    f.close()
    with open("Dataset/{}/idx2class.json".format(dataset_sw),'w+') as f:
        f.write(json.dumps(dc))
    f.close()
    # exit(0)
    
    ds = {}
    for dt in ["train","val","test"]:
        ds[dt] = []
        x_bs = []
        y_bs = []
        for idx in range(imgs[dt].shape[0]):
            x_bs.append(imgs[dt][idx])
            y_bs.append(labs[dt][idx])
            if (idx%batch_size==batch_size-1 or idx>=imgs[dt].shape[0]-1):
                ds[dt].append([np.array(x_bs),np.array(y_bs)])
                x_bs.clear()
                y_bs.clear()

    # s1. train baseline
    train(ds,model,50)
    # s2. val baseline
    val(ds['test'],model)
    # s3. collect True classified sample of baseline for Craft
    val_collect_ps(ds["train"],model)
