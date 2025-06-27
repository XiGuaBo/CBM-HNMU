from nmf_utils import *
import cv2 as cv
from train_base import DataLoader
from network import *
from network import PREDICTOR_NL as PREDICTOR

import clip
import torch

import json

from PIL import Image

import os,sys

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
    model_lock = NFRES50(class_nums_dict[dataset_sw])
    sample_size = 256
elif (net_sw == "resnext26"):
    model = RESNEXT26(class_nums_dict[dataset_sw])
    model_lock = RESNEXT26(class_nums_dict[dataset_sw])
    sample_size = 256
elif (net_sw == "botnet26t"):
    model = BOTNET26t(class_nums_dict[dataset_sw])
    model_lock = BOTNET26t(class_nums_dict[dataset_sw])
    sample_size = 256
elif (net_sw == "rexnet100"):
    model = REXNET100(class_nums_dict[dataset_sw])
    model_lock = REXNET100(class_nums_dict[dataset_sw])
    sample_size = 256
elif (net_sw == "vit"):
    model = Vit(class_nums_dict[dataset_sw])
    model_lock = Vit(class_nums_dict[dataset_sw])
    sample_size = 224
elif (net_sw == "gcvit"):
    model = GCVIT(class_nums_dict[dataset_sw])
    model_lock = GCVIT(class_nums_dict[dataset_sw])
    sample_size = 224
elif (net_sw == "cait"):
    model = Cait_Small(class_nums_dict[dataset_sw])
    model_lock = Cait_Small(class_nums_dict[dataset_sw])
    sample_size = 224
elif (net_sw == "convit"):
    model = Convit_Base(class_nums_dict[dataset_sw])
    model_lock = Convit_Base(class_nums_dict[dataset_sw])
    sample_size = 224
elif (net_sw == "deit"):
    model = Deit_Base(class_nums_dict[dataset_sw])
    model_lock = Deit_Base(class_nums_dict[dataset_sw])
    sample_size = 224
else:
    # no choice will switch to nfresnet50 and flower102
    model = NFRES50(class_nums_dict[dataset_sw])
    model_lock = NFRES50(class_nums_dict[dataset_sw])
    sample_size = 256
    

def PROCESS_TARCLS(tar_cls_str):
    tar_cls = []
    for cls in tar_cls_str.split(','):
        tar_cls.append(int(cls))
    return tar_cls

# Local Approximation
def CKD_TRAIN(epochs=200,loss_type="mse",load_predictor=False,tar_cls=None,kd_target="val"):
    # 1. train data prepare for linear predictor
    imgs,labs,cd,dc = DataLoader(zone=[kd_target])
    batch_size=1 # must be setted to 1
    ds = {}
    for dt in [kd_target]:
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
    del imgs,labs
    with open("Dataset/{}/selected_concepts.json".format(dataset_sw),'r') as f:
        cb = json.loads(f.read())
    f.close() 
    # 2. prepare models
    model.load_state_dict(torch.load("weights/{}/latest_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw)))
    model.eval().to("cuda")
    
    predictor = PREDICTOR(len(cb),len(tar_cls))
    if (load_predictor):
        predictor.load_state_dict(torch.load("weights/{}/latest_{}_predictor_nl_{}_cls_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw,loss_type,str(tar_cls)[1:-1])))
    predictor.train().to("cuda")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmodel, preprocess = clip.load('ViT-B/32', device)
    cbt = clip.tokenize(cb).to(device)
    print (len(cbt))
    with torch.no_grad():
        text_features = cmodel.encode_text(cbt)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    del cbt
    # print (text_features.shape)
    # 3. pre-compute
    all_crops = []
    all_crops_u = []
    all_w = []
    all_craft = []
    all_ts = []
    for c_idx in range(class_nums_dict[dataset_sw]):   
        crops, crops_u, w, craft, ts = NMF_EXTRACTOR(model,target_cls=c_idx,dataset=dataset_sw,arch=net_sw)
        all_crops.append(crops)
        all_crops_u.append(crops_u)
        all_w.append(w)
        all_craft.append(craft)
        all_ts.append(ts)
    # 4. train procedure
    # loss_kl = nn.KLDivLoss(reduction="batchmean")
    # loss_cl = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()
    
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-4)
    global_loss = 99999999999.
    for ep in range(epochs):
        mean_loss = 0.
        current = 0
        size = len(ds[kd_target])
        p_A = 1
        blance = 0
        for batch, (X, y) in enumerate(ds[kd_target]):
            if  not (int(y.squeeze()) in tar_cls):
                # continue
                if (blance > 0):
                    blance -= 1
                else:
                    continue
            else:
                blance += 1
            # p_A +=1
            # print (X.shape)
            # print (y.shape)
            if not (len(X.shape)==4):
                X = np.expand_dims(X,axis=0)
            TX = torch.tensor(X).to('cuda')
            TX = TX.permute(0, 3, 1, 2)
            # exit()
            # Compute prediction error
            pred = model(TX)
            # loss = loss_fn(pred, y)
            # mean_loss += loss/size
            y = int(y.squeeze())
            # Ty = torch.tensor(tar_cls.index(y)).to('cuda')
            # Ty = Ty.unsqueeze(0)
            best_concepts, most_important_concepts = NMF_EXCUTOR(all_crops[y],all_crops_u[y],all_craft[y],X,y,all_ts[y],idx=batch,tar_zone=kd_target,concepts_out=False,dataset=dataset_sw)
            for bc in best_concepts:
                bc -= bc.min()
                bc /= bc.max()
                bc *= 255.0
            with torch.no_grad():
                most_important_concepts = torch.tensor(most_important_concepts).to(device)
                image_features = torch.stack([cmodel.encode_image(preprocess(Image.fromarray(bc.astype(np.uint8))).unsqueeze(0).to(device)).squeeze() for bc in best_concepts], 0)
                image_features = image_features.T * most_important_concepts
                image_features = torch.tensor(image_features.T,dtype=torch.float16)
                image_features /= image_features.norm(dim=-1, keepdim=True)
    
            score = (100 * image_features @ text_features.T).softmax(dim=-1)
            score = torch.mean(score,dim=0) * 100
            # print (score.shape)
            top_k = score.to("cpu").numpy().argsort()[-5:][::-1]
            print (np.array(cb)[top_k])
            print (score.to("cpu").numpy()[top_k])
            print (score.to("cpu").numpy())
            
            pred_cls = torch_to_numpy(pred.argmax(1))
            print ("pred: ",dc[int(pred_cls)])
            print ("gd: ",dc[y])
                
            if not (score.shape == 2):
                score = score.unsqueeze(0)
            score = torch.tensor(score,dtype=torch.float32)
            cbm_pred = predictor(score)
            tar_pred = pred[:, tar_cls]
            # distillation loss
            loss = loss_mse(cbm_pred,tar_pred)
            # print (cbm_pred)
            # print (pred)
            # print (Ty)
            # print (loss)
            mean_loss += loss/p_A
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if batch % 100 == 0:
            loss = loss.item()
            current += len(X)
            mean_loss = mean_loss.item()
            print ("black box logits : ", pred[:, tar_cls].detach().to("cpu").numpy().squeeze())
            print ("cbm logits : ", cbm_pred.detach().to("cpu").numpy().squeeze())
            print ("black box pred : ", tar_cls[int(np.argmax(tar_pred.detach().to("cpu").numpy().squeeze()))])
            print ("cbm pred : ", tar_cls[int(np.argmax(cbm_pred.detach().to("cpu").numpy().squeeze()))])
            print(f"loss: {loss:>7f} mloss: {mean_loss:>7f}  [{current:>5d}/ - ]")

        # os.makedirs("weights/{}/".format(dataset_sw),exist_ok=True)
        if (global_loss > mean_loss):
            torch.save(predictor.state_dict(),"weights/{}/latest_{}_predictor_nl_{}_cls_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw,loss_type,str(tar_cls)[1:-1]))
            global_loss = mean_loss
            print ("saved weights for ep {} / Loss {:.3f}.".format(ep,mean_loss))

# PREDICTOR (CBM) TEST
def CKD_TEST(loss_type="mse",concepts_out=False,target="test",tar_cls=None):
    # 1. test data prepare for linear predictor
    imgs,labs,cd,dc = DataLoader(zone=[target])
    batch_size=1 # must be setted to 1
    ds = {}
    for dt in [target]:
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
    del imgs,labs
    with open("Dataset/{}/selected_concepts.json".format(dataset_sw),'r') as f:
        cb = json.loads(f.read())
    f.close() 
    # 2. prepare models
    model.load_state_dict(torch.load("weights/{}/latest_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw)))
    model.eval().to("cuda")
    
    predictor = PREDICTOR(len(cb),len(tar_cls))
    if (load_predictor):
        predictor.load_state_dict(torch.load("weights/{}/latest_{}_predictor_nl_{}_cls_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw,loss_type,str(tar_cls)[1:-1])))
    predictor.train().to("cuda")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmodel, preprocess = clip.load('ViT-B/32', device)
    cbt = clip.tokenize(cb).to(device)
    print (len(cbt))
    with torch.no_grad():
        text_features = cmodel.encode_text(cbt)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    del cbt
    # print (text_features.shape)
    # 3. pre-compute
    all_crops = []
    all_crops_u = []
    all_w = []
    all_craft = []
    all_ts = []
    for c_idx in range(class_nums_dict[dataset_sw]):   
        crops, crops_u, w, craft, ts = NMF_EXTRACTOR(model,target_cls=c_idx,dataset=dataset_sw,arch=net_sw)
        all_crops.append(crops)
        all_crops_u.append(crops_u)
        all_w.append(w)
        all_craft.append(craft)
        all_ts.append(ts)
        
    # val
    # loss_kl = nn.KLDivLoss(reduction="batchmean")
    # loss_cl = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()
    
    mean_loss = 0.
    current = 0
    size = len(ds[target])
    P_B = 0
    P_CBM = 0
    P_A = 0
    
    P_CBM_CLS = np.zeros((class_nums_dict[dataset_sw],))
    P_CBM_CLS_A = np.zeros((class_nums_dict[dataset_sw],))
    for batch, (X, y) in enumerate(ds[target]):
        # if  not (int(y.squeeze()) in tar_cls):
        #     continue
        with torch.no_grad():
            if not (len(X.shape)==4):
                X = np.expand_dims(X,axis=0)
            TX, Ty = torch.tensor(X).to('cuda'), torch.tensor(y).to('cuda')
            TX = TX.permute(0, 3, 1, 2)
            pred = model(TX)
            y = int(y.squeeze())
            best_concepts, most_important_concepts = NMF_EXCUTOR(all_crops[y],all_crops_u[y],all_craft[y],X,y,all_ts[y],idx=batch,tar_zone=target,concepts_out=concepts_out,dataset=dataset_sw)
            for bc in best_concepts:
                bc -= bc.min()
                bc /= bc.max()
                bc *= 255.0
            most_important_concepts = torch.tensor(most_important_concepts).to(device)
            image_features = torch.stack([cmodel.encode_image(preprocess(Image.fromarray(bc.astype(np.uint8))).unsqueeze(0).to(device)).squeeze() for bc in best_concepts], 0)
            image_features = image_features.T * most_important_concepts
            image_features = torch.tensor(image_features.T,dtype=torch.float16)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            score = (100 * image_features @ text_features.T).softmax(dim=-1)
            score = torch.mean(score,dim=0) * 100
            
            top_k = score.to("cpu").numpy().argsort()[-5:][::-1]
            print (np.array(cb)[top_k])
            print (score.to("cpu").numpy()[top_k])
            print (score.to("cpu").numpy())
            
            pred_cls = torch_to_numpy(pred.argmax(1))
            print ("pred: ",dc[int(pred_cls)])
            print ("gd: ",dc[y])
            
            if not (score.shape == 2):
                score = score.unsqueeze(0)
            score = torch.tensor(score,dtype=torch.float32)
            cbm_pred = predictor(score)
            
            soft_cbm_pred = torch.nn.functional.softmax(cbm_pred/2, dim=1)
            soft_pred = torch.nn.functional.softmax(pred/1, dim=1)
            
            tar_cbm_pred = torch.zeros((1,class_nums_dict[dataset_sw]),dtype=torch.float32).to("cuda")
            tar_cbm_pred[0][:] = soft_pred[0][:] 
            tar_cbm_pred[0][tar_cls] = 0.
            pr = 1 - tar_cbm_pred.sum()
            tar_cbm_pred[:,tar_cls] = soft_cbm_pred[0][:] * pr
            
            # tar_cbm_pred[0][:] = pred[0][:] 
            # tar_cbm_pred[0][tar_cls] = 0.
            # tar_cbm_pred[:,tar_cls] = cbm_pred[0][:]
            
            cbm_pred_cls = torch_to_numpy(tar_cbm_pred.argmax(1))
            print ("cbm pred: ",dc[int(cbm_pred_cls)])
            print ("gd: ",dc[y])
            
            if (pred_cls == y):
                P_B += 1
            if (cbm_pred_cls == y):
                P_CBM += 1
                P_CBM_CLS[y] += 1
                
            P_CBM_CLS_A[y] += 1
            P_A += 1
            
            loss = loss_mse(tar_cbm_pred,pred)
                
            mean_loss += loss/size
            loss = loss.item()
            current += len(X)
            mean_loss = mean_loss.item()
            # print ("black box logits : ", pred.to("cpu").numpy().squeeze())
            # print ("cbm logits : ", tar_cbm_pred.to("cpu").numpy().squeeze())
            print ("black box softmax : ", soft_pred.to("cpu").numpy().squeeze())
            print ("cbm softmax : ", tar_cbm_pred.to("cpu").numpy().squeeze())
            print ("black box pred : ", np.argmax(pred.to("cpu").numpy().squeeze()))
            print ("cbm pred : ", np.argmax(tar_cbm_pred.to("cpu").numpy().squeeze()))
            print(f"loss: {loss:>7f} mloss: {mean_loss:>7f}  [{current:>5d}/ - ]")
            
    print ("black box ACC : {:.2f}%".format(100*P_B/P_A))
    print ("CBM ACC : {:.2f}%".format(100*P_CBM/P_A))
    print ("CBM ACC FOR TAR CLASS :",P_CBM_CLS[tar_cls]/P_CBM_CLS_A[tar_cls])
    print ("CBM ACC FOR CLASS :",P_CBM_CLS/P_CBM_CLS_A)

# Concepts Intervention 
def CCBM_INTERVENT_nT_with_pF(loss_type="mse", inter_concepts_nums=10, val_target="val", tar_cls=None, int_target="val"):
    # 1. train data prepare for linear predictor
    imgs,labs,cd,dc = DataLoader(zone=[int_target,val_target])
    batch_size=1 # must be set to 1
    ds = {}
    for dt in [int_target,val_target]:
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
    del imgs,labs
    with open("Dataset/{}/selected_concepts.json".format(dataset_sw),'r') as f:
        cb = json.loads(f.read())
    f.close() 
    # 2. prepare models
    model.load_state_dict(torch.load("weights/{}/latest_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw)))
    model.eval().to("cuda")
    
    predictor = PREDICTOR(len(cb),len(tar_cls))
    predictor.load_state_dict(torch.load("weights/{}/latest_{}_predictor_nl_{}_cls_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw,loss_type,str(tar_cls)[1:-1])))
    predictor.train().to("cuda")
    
    predictor_org = PREDICTOR(len(cb),len(tar_cls))
    predictor_org.load_state_dict(torch.load("weights/{}/latest_{}_predictor_nl_{}_cls_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw,loss_type,str(tar_cls)[1:-1])))
    predictor_org.train().to("cuda")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmodel, preprocess = clip.load('ViT-B/32', device)
    cbt = clip.tokenize(cb).to(device)
    print (len(cbt))
    with torch.no_grad():
        text_features = cmodel.encode_text(cbt)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    del cbt
    # print (text_features.shape)
    # 3. pre-compute
    all_crops = []
    all_crops_u = []
    all_w = []
    all_craft = []
    all_ts = []
    for c_idx in range(class_nums_dict[dataset_sw]):   
        crops, crops_u, w, craft, ts = NMF_EXTRACTOR(model,target_cls=c_idx,dataset=dataset_sw,arch=net_sw)
        all_crops.append(crops)
        all_crops_u.append(crops_u)
        all_w.append(w)
        all_craft.append(craft)
        all_ts.append(ts)
        
    size = len(ds[int_target])
    current = 0
    cls_act = np.zeros((class_nums_dict[dataset_sw],len(cb)),dtype=np.float32)
    pf_act = np.zeros((class_nums_dict[dataset_sw],len(cb)),dtype=np.float32)
    # cctp_vote = np.zeros((len(cb),),dtype=np.float32)
    
    # 4. intervene
    for batch, (X, y) in enumerate(ds[int_target]):    
        # if not (int(y.squeeze()) in tar_cls):
        #     continue
        # cctp = torch.zeros((len(cb),),dtype=torch.float32).to(device)
        if not (len(X.shape)==4):
            X = np.expand_dims(X,axis=0)
        TX, Ty = torch.tensor(X).to('cuda'), torch.tensor(y).to('cuda')
        TX = TX.permute(0, 3, 1, 2)
        y = int(y.squeeze())
        best_concepts, most_important_concepts = NMF_EXCUTOR(all_crops[y],all_crops_u[y],all_craft[y],X,y,all_ts[y],idx=batch,tar_zone=int_target,concepts_out=False,dataset=dataset_sw)
        for bc in best_concepts:
            bc -= bc.min()
            bc /= bc.max()
            bc *= 255.0
        with torch.no_grad():
            most_important_concepts = torch.tensor(most_important_concepts).to(device)
            image_features = torch.stack([cmodel.encode_image(preprocess(Image.fromarray(bc.astype(np.uint8))).unsqueeze(0).to(device)).squeeze() for bc in best_concepts], 0)
            image_features = image_features.T * most_important_concepts
            image_features = torch.tensor(image_features.T,dtype=torch.float16)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        score = (100 * image_features @ text_features.T).softmax(dim=-1)
        score = torch.mean(score,dim=0) * 100
        if not (score.shape == 2):
            score = score.unsqueeze(0)
        score = torch.tensor(score,dtype=torch.float32,requires_grad=True)
        cbm_pred = predictor(score)
        cbm_pred_cls  = np.argmax(cbm_pred.squeeze().detach().to("cpu").numpy())
        
        if (y in tar_cls):
            # grad-based concepts
            cbm_pred.squeeze()[tar_cls.index(y)].backward(retain_graph=True)
            act = score[0] * predictor.weights.grad[:,tar_cls.index(y)] * predictor.weights[:,tar_cls.index(y)] 
            cls_act[y] += act.detach().to("cpu").numpy().squeeze() 
            predictor.weights.grad.data.zero_()
            if not (tar_cls[int(cbm_pred_cls)] == y):
                cbm_pred.squeeze()[int(cbm_pred_cls)].backward()
                act = score[0] * predictor.weights.grad[:,int(cbm_pred_cls)] * predictor.weights[:,int(cbm_pred_cls)] 
                pf_act[tar_cls[int(cbm_pred_cls)]] += act.detach().to("cpu").numpy().squeeze() 
                predictor.weights.grad.data.zero_()
            else:
                cbm_pred.squeeze()[int(cbm_pred_cls)].backward()
                act = score[0] * predictor.weights.grad[:,int(cbm_pred_cls)] * predictor.weights[:,int(cbm_pred_cls)] 
                pf_act[tar_cls[int(cbm_pred_cls)]] -= act.detach().to("cpu").numpy().squeeze() 
                predictor.weights.grad.data.zero_()
        else:
            cbm_pred.squeeze()[int(cbm_pred_cls)].backward(retain_graph=True)
            act = score[0] * predictor.weights.grad[:,int(cbm_pred_cls)] * predictor.weights[:,int(cbm_pred_cls)] 
            pf_act[tar_cls[int(cbm_pred_cls)]] += act.detach().to("cpu").numpy().squeeze() 
            predictor.weights.grad.data.zero_()
        
        current += len(X)
        print(f"[{current:>5d}/ - ]")
    
    # class level mask
    mask = torch.ones((len(cb),len(tar_cls)),dtype=torch.float32).to(device)
    for i in tar_cls:
        intervent_concepts_nT = cls_act[i].argsort()[:inter_concepts_nums//2]
        print ("class <{}> nT : ".format(i), np.array(cb)[intervent_concepts_nT])
        intervent_concepts_pF = pf_act[i].argsort()[::-1][:inter_concepts_nums//2]
        print ("class <{}> pF : ".format(i), np.array(cb)[intervent_concepts_pF])
        # intervent_concepts = np.concatenate([intervent_concepts_nT,intervent_concepts_pF])
        for ic in intervent_concepts_nT:
            # hard mask
            if (cls_act[i][ic] < 0):
                mask[ic,tar_cls.index(i)] = 0 
                
            # soft mask (grad based)
            # mask[ic,i] = torch.tensor(cls_act[i][ic]).to("cuda")
            # predictor.weights[ic][i] += mask[ic,i] / size
        for ic in intervent_concepts_pF:
            # if (cls_act[i][ic] < 0 and predictor.weights[ic][i] > 0): # fliter the global positve concepts
            # if (cls_act[i][ic] < 0 and np.abs(cls_act[i][ic]) > 0.5): # fliter the global positve concepts +
            # hard mask
            if (pf_act[i][ic] > 0):
                mask[ic,tar_cls.index(i)] = 0 
    
    predictor.weights = torch.nn.Parameter(predictor.weights * mask) # wij~ = 0
    torch.save(predictor.state_dict(),"weights/{}/latest_{}_intervented_predictor_nl_{}_cls_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw,loss_type,str(tar_cls)[1:-1]))
    
    # 5. val the modified cbm
    # loss_kl = nn.KLDivLoss(reduction="batchmean")
    # loss_cl = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()
    
    mean_loss = 0.
    current = 0
    size = len(ds[val_target])
    P_B = 0
    P_CBM = 0
    P_A = 0
    
    P_CBM_CLS = np.zeros((class_nums_dict[dataset_sw],))
    P_CBM_CLS_A = np.zeros((class_nums_dict[dataset_sw],))
    for batch, (X, y) in enumerate(ds[val_target]):
        # if not (int(y.squeeze()) in tar_cls):
        #     continue
        with torch.no_grad():
            if not (len(X.shape)==4):
                X = np.expand_dims(X,axis=0)
            TX, Ty = torch.tensor(X).to('cuda'), torch.tensor(y).to('cuda')
            TX = TX.permute(0, 3, 1, 2)
            pred = model(TX)
            y = int(y.squeeze())
            best_concepts, most_important_concepts = NMF_EXCUTOR(all_crops[y],all_crops_u[y],all_craft[y],X,y,all_ts[y],idx=batch,tar_zone=val_target,concepts_out=False,dataset=dataset_sw)
            for bc in best_concepts:
                bc -= bc.min()
                bc /= bc.max()
                bc *= 255.0
            most_important_concepts = torch.tensor(most_important_concepts).to(device)
            image_features = torch.stack([cmodel.encode_image(preprocess(Image.fromarray(bc.astype(np.uint8))).unsqueeze(0).to(device)).squeeze() for bc in best_concepts], 0)
            image_features = image_features.T * most_important_concepts
            image_features = torch.tensor(image_features.T,dtype=torch.float16)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            score = (100 * image_features @ text_features.T).softmax(dim=-1)
            score = torch.mean(score,dim=0) * 100
            
            top_k = score.to("cpu").numpy().argsort()[-5:][::-1]
            print (np.array(cb)[top_k])
            print (score.to("cpu").numpy()[top_k])
            print (score.to("cpu").numpy())
            
            pred_cls = torch_to_numpy(pred.argmax(1))
            print ("pred: ",dc[int(pred_cls)])
            print ("gd: ",dc[y])
            
            if not (score.shape == 2):
                score = score.unsqueeze(0)
            score = torch.tensor(score,dtype=torch.float32)
            cbm_pred = predictor(score)
            cbm_pred_org = predictor_org(score)
            
            soft_cbm_pred_org = torch.nn.functional.softmax(cbm_pred_org/2, dim=1)
            soft_cbm_pred = torch.nn.functional.softmax(cbm_pred/2, dim=1)
            soft_pred = torch.nn.functional.softmax(pred/1, dim=1)
            
            tar_cbm_pred = torch.zeros((1,class_nums_dict[dataset_sw]),dtype=torch.float32).to("cuda")
            tar_cbm_pred[0][:] = soft_pred[0][:] 
            tar_cbm_pred[0][tar_cls] = 0.
            pr = 1 - tar_cbm_pred.sum()
            tar_cbm_pred[0][tar_cls] = soft_cbm_pred[0][:] * pr
            
            tar_cbm_pred_org = torch.zeros((1,class_nums_dict[dataset_sw]),dtype=torch.float32).to("cuda")
            tar_cbm_pred_org[0][:] = soft_pred[0][:] 
            tar_cbm_pred_org[0][tar_cls] = soft_cbm_pred_org[0][:] * pr
            
            cbm_pred_cls = torch_to_numpy(tar_cbm_pred.argmax(1))
            cbm_pred_cls_org = torch_to_numpy(tar_cbm_pred_org.argmax(1))
            print ("cbm pred: ",dc[int(cbm_pred_cls)])
            print ("cbm org pred: ",dc[int(cbm_pred_cls_org)])
            print ("gd: ",dc[y])
            
            print ("cbm pred logits:",cbm_pred)
            print ("cbm pred org logits:",cbm_pred_org)
            print ("cbm pred softmax:",soft_cbm_pred)
            print ("cbm pred org softmax:",soft_cbm_pred_org)
            print ("black box logits:",pred[0][y])
            print ("black box softmax:",soft_pred[0][y])
            
            if (pred_cls == y):
                P_B += 1
            if (cbm_pred_cls == y):
                P_CBM += 1
                P_CBM_CLS[y] += 1
                
            P_CBM_CLS_A[y] += 1
            P_A += 1
            
            loss = loss_mse(soft_pred,tar_cbm_pred)
                
            mean_loss += loss/size
            loss = loss.item()
            current += len(X)
            mean_loss = mean_loss.item()
            # print ("black box logits : ", pred.to("cpu").numpy().squeeze())
            # print ("cbm logits : ", tar_cbm_pred.to("cpu").numpy().squeeze())
            print ("black box softmax : ", soft_pred.to("cpu").numpy().squeeze())
            print ("cbm softmax : ", tar_cbm_pred.to("cpu").numpy().squeeze())
            print ("black box pred : ", np.argmax(pred.to("cpu").numpy().squeeze()))
            print ("cbm pred : ", np.argmax(tar_cbm_pred.to("cpu").numpy().squeeze()))
            print(f"loss: {loss:>7f} mloss: {mean_loss:>7f}  [{current:>5d}/ - ]")
            
    print ("black box ACC : {:.2f}%".format(100*P_B/P_A))
    print ("CBM ACC : {:.2f}%".format(100*P_CBM/P_A))
    print ("CBM ACC FOR TAR CLASS :",P_CBM_CLS[tar_cls]/P_CBM_CLS_A[tar_cls])
    print ("CBM ACC FOR CLASS :",P_CBM_CLS/P_CBM_CLS_A)

# Knowledge Transfer
def RCKD_TRAIN(epochs=10, loss_type="mse", tar_cls=None, int_target="val"):
    # 1. train data prepare for linear predictor
    imgs,labs,cd,dc = DataLoader(zone=[int_target,"test"])
    batch_size=1 # must be set to 1
    ds = {}
    for dt in [int_target,"test"]:
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
    del imgs,labs
    with open("Dataset/{}/selected_concepts.json".format(dataset_sw),'r') as f:
        cb = json.loads(f.read())
    f.close() 
    # 2. prepare models
    model.load_state_dict(torch.load("weights/{}/latest_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw)))
    model.eval().to("cuda")
    
    model_lock.load_state_dict(torch.load("weights/{}/latest_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw)))
    model_lock.eval().to("cuda")
    
    predictor = PREDICTOR(len(cb),len(tar_cls))
    predictor.load_state_dict(torch.load("weights/{}/latest_{}_intervented_predictor_nl_{}_cls_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw,loss_type,str(tar_cls)[1:-1])))
    predictor.train().to("cuda")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmodel, preprocess = clip.load('ViT-B/32', device)
    cbt = clip.tokenize(cb).to(device)
    print (len(cbt))
    with torch.no_grad():
        text_features = cmodel.encode_text(cbt)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    del cbt
    # print (text_features.shape)
    # 3. pre-compute
    all_crops = []
    all_crops_u = []
    all_w = []
    all_craft = []
    all_ts = []
    for c_idx in range(class_nums_dict[dataset_sw]):   
        crops, crops_u, w, craft, ts = NMF_EXTRACTOR(model_lock,target_cls=c_idx,dataset=dataset_sw,arch=net_sw)
        all_crops.append(crops)
        all_crops_u.append(crops_u)
        all_w.append(w)
        all_craft.append(craft)
        all_ts.append(ts)
    # 4. train procedure
    # loss_kl = nn.KLDivLoss(reduction="batchmean")
    # loss_cl = nn.CrossEntropyLoss()
    # loss_mse = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-7)
    global_loss = 99999999999.
    for ep in range(epochs):
        mean_loss = 0.
        current = 0
        size = len(ds[int_target])
        for batch, (X, y) in enumerate(ds[int_target]):
            # if not (int(y.squeeze()) in tar_cls):
            #     continue
            # print (X.shape)
            # print (y.shape)
            if not (len(X.shape)==4):
                X = np.expand_dims(X,axis=0)
            TX, Ty = torch.tensor(X).to('cuda'), torch.tensor(y).to('cuda')
            TX = TX.permute(0, 3, 1, 2)
            # exit()
            # Compute prediction error
            pred = model(TX)
            # loss = loss_fn(pred, y)
            # mean_loss += loss/size
            y = int(y.squeeze())
            
            pred_cls = torch_to_numpy(pred.argmax(1))
            print ("pred: ",dc[int(pred_cls)])
            print ("gd: ",dc[y])
            
            with torch.no_grad():
                pred_org = model_lock(TX)
                best_concepts, most_important_concepts = NMF_EXCUTOR(all_crops[y],all_crops_u[y],all_craft[y],X,y,all_ts[y],idx=batch,tar_zone=int_target,concepts_out=False,dataset=dataset_sw)
                for bc in best_concepts:
                    bc -= bc.min()
                    bc /= bc.max()
                    bc *= 255.0 

                most_important_concepts = torch.tensor(most_important_concepts).to(device)
                image_features = torch.stack([cmodel.encode_image(preprocess(Image.fromarray(bc.astype(np.uint8))).unsqueeze(0).to(device)).squeeze() for bc in best_concepts], 0)
                image_features = image_features.T * most_important_concepts
                image_features = torch.tensor(image_features.T,dtype=torch.float16)
                image_features /= image_features.norm(dim=-1, keepdim=True)
    
                score = (100 * image_features @ text_features.T).softmax(dim=-1)
                score = torch.mean(score,dim=0) * 100
                # print (score.shape)
                top_k = score.to("cpu").numpy().argsort()[-5:][::-1]
                print (np.array(cb)[top_k])
                print (score.to("cpu").numpy()[top_k])
                print (score.to("cpu").numpy())
                
                if not (score.shape == 2):
                    score = score.unsqueeze(0)
                score = torch.tensor(score,dtype=torch.float32)
                cbm_pred = predictor(score)
                
            # class level
            soft_cbm_pred = torch.nn.functional.softmax(cbm_pred/2, dim=1)
            soft_pred = torch.nn.functional.softmax(pred/1.5, dim=1)
            soft_pred_org = torch.nn.functional.softmax(pred_org/1, dim=1)
            
            tar_cbm_pred = torch.zeros((1,class_nums_dict[dataset_sw]),dtype=torch.float32).to("cuda")
            tar_cbm_pred[0][:] = soft_pred_org[0][:] 
            tar_cbm_pred[0][tar_cls] = 0.
            pr = 1 - tar_cbm_pred.sum()
            tar_cbm_pred[0][tar_cls] = soft_cbm_pred[0][:] * pr
            
            cbm_pred_cls = torch_to_numpy(tar_cbm_pred.argmax(1))
            print ("cbm pred: ",dc[int(cbm_pred_cls)])
            print ("gd: ",dc[y])
            # distillation loss
            loss = pt_categorical_crossentropy(soft_pred,tar_cbm_pred)
            if (torch.isnan(loss)):
                continue
            # print (cbm_pred)
            # print (pred)
            # print (Ty)
            # print (loss)
            mean_loss += loss/size
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if batch % 100 == 0:
            loss = loss.item()
            current += len(X)
            mean_loss = mean_loss.item()
            print ("black box logits : ", pred.detach().to("cpu").numpy().squeeze())
            print ("target logits : ", tar_cbm_pred.detach().to("cpu").numpy().squeeze())
            # print ("black box softmax : ", nn.functional.softmax(pred,dim=1).detach().to("cpu").numpy().squeeze())
            # print ("cbm softmax : ", nn.functional.softmax(cbm_pred,dim=1).detach().to("cpu").numpy().squeeze())
            # print ("black box pred : ", np.argmax(pred.detach().to("cpu").numpy().squeeze()))
            # print ("cbm pred : ", np.argmax(cbm_pred.detach().to("cpu").numpy().squeeze()))
            print(f"loss: {loss:>7f} mloss: {mean_loss:>7f}  [{current:>5d}/ - ]")

        if (global_loss > mean_loss):
            torch.save(model.state_dict(),"weights/{}/latest_{}_intervented_{}_cls_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw,loss_type,str(tar_cls)[1:-1]))
            global_loss = mean_loss
            print ("saved weights for ep {} / Loss {:.3f}.".format(ep,mean_loss))
        
        # val
        mean_loss = 0.
        current = 0
        size = len(ds["test"])
        
        P_B = 0
        # P_CBM = 0
        P_CLS = np.zeros((class_nums_dict[dataset_sw],))
        P_CLS_A = np.zeros((class_nums_dict[dataset_sw],))
        for batch, (X, y) in enumerate(ds["test"]):
            with torch.no_grad():
                if not (len(X.shape)==4):
                    X = np.expand_dims(X,axis=0)
                TX, Ty = torch.tensor(X).to('cuda'), torch.tensor(y).to('cuda')
                TX = TX.permute(0, 3, 1, 2)
                pred = model(TX)
                y = int(y.squeeze())
                
                pred_cls = torch_to_numpy(pred.argmax(1))
                print ("pred: ",dc[int(pred_cls)])
                print ("gd: ",dc[y])
                
                if (pred_cls == y):
                    P_B += 1
                    P_CLS[y] += 1
                    
                P_CLS_A[y] += 1
                
                print ("black box logits : ", pred.to("cpu").numpy().squeeze())
                
        print ("black box ACC : {:.2f}%".format(100*P_B/size))
        # print ("CBM ACC : {:.2f}%".format(100*P_CBM/size))
        print ("black box ACC FOR TAR CLASS :",(P_CLS/P_CLS_A)[tar_cls])
        print ("black box ACC FOR CLASS :",P_CLS/P_CLS_A)

def pt_categorical_crossentropy(pred, label):
    """
    使用pytorch 来实现 categorical_crossentropy
    """
    # print(-label * torch.log(pred))
    return torch.sum(-label * torch.log(pred))

# random intervention
def CCBM_INTERVENT_RANDOM(loss_type="mse", inter_concepts_nums=10, tar_cls=None):
    # 1. train data prepare for linear predictor
    with open("Dataset/{}/selected_concepts.json".format(dataset_sw),'r') as f:
        cb = json.loads(f.read())
    f.close() 
    # 2. prepare models
    predictor = PREDICTOR(len(cb),len(tar_cls))
    predictor.load_state_dict(torch.load("weights/{}/latest_{}_predictor_nl_{}_cls_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw,loss_type,str(tar_cls)[1:-1])))
    predictor.train().to("cuda")
    
    # 3. pre-compute
    mask = torch.ones((len(cb),len(tar_cls)),dtype=torch.float32).to("cuda")
    for i in tar_cls:
        intervent_concepts = np.random.choice(len(cb),inter_concepts_nums,replace=False)
        for ic in intervent_concepts:
            # hard mask
            mask[ic,tar_cls.index(i)] = 0 
    
    predictor.weights = torch.nn.Parameter(predictor.weights * mask) # wij~ = 0
    torch.save(predictor.state_dict(),"weights/{}/latest_{}_random_intervented_predictor_nl_{}_cls_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw,loss_type,str(tar_cls)[1:-1]))
    
def RANDOM_RCKD_TRAIN(epochs=10, loss_type="mse", tar_cls=None, int_target="val"):
    # 1. train data prepare for linear predictor
    imgs,labs,cd,dc = DataLoader(zone=[int_target,"test"])
    batch_size=1 # must be setted to 1
    ds = {}
    for dt in [int_target,"test"]:
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
    del imgs,labs
    with open("Dataset/{}/selected_concepts.json".format(dataset_sw),'r') as f:
        cb = json.loads(f.read())
    f.close() 
    # 2. prepare models
    model.load_state_dict(torch.load("weights/{}/latest_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw)))
    model.eval().to("cuda")
    
    model_lock.load_state_dict(torch.load("weights/{}/latest_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw)))
    model_lock.eval().to("cuda")
    
    predictor = PREDICTOR(len(cb),len(tar_cls))
    predictor.load_state_dict(torch.load("weights/{}/latest_{}_random_intervented_predictor_nl_{}_cls_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw,loss_type,str(tar_cls)[1:-1])))
    predictor.train().to("cuda")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmodel, preprocess = clip.load('ViT-B/32', device)
    cbt = clip.tokenize(cb).to(device)
    print (len(cbt))
    with torch.no_grad():
        text_features = cmodel.encode_text(cbt)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    del cbt
    # print (text_features.shape)
    # 3. pre-compute
    all_crops = []
    all_crops_u = []
    all_w = []
    all_craft = []
    all_ts = []
    for c_idx in range(class_nums_dict[dataset_sw]):   
        crops, crops_u, w, craft, ts = NMF_EXTRACTOR(model_lock,target_cls=c_idx,dataset=dataset_sw,arch=net_sw)
        all_crops.append(crops)
        all_crops_u.append(crops_u)
        all_w.append(w)
        all_craft.append(craft)
        all_ts.append(ts)
    # 4. train procedure
    loss_kl = nn.KLDivLoss(reduction="batchmean")
    loss_cl = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-7)
    global_loss = 99999999999.
    for ep in range(epochs):
        mean_loss = 0.
        current = 0
        size = len(ds[int_target])
        for batch, (X, y) in enumerate(ds[int_target]):
            # if not (int(y.squeeze()) in tar_cls):
            #     continue
            # print (X.shape)
            # print (y.shape)
            if not (len(X.shape)==4):
                X = np.expand_dims(X,axis=0)
            TX, Ty = torch.tensor(X).to('cuda'), torch.tensor(y).to('cuda')
            TX = TX.permute(0, 3, 1, 2)
            # exit()
            # Compute prediction error
            pred = model(TX)
            # loss = loss_fn(pred, y)
            # mean_loss += loss/size
            y = int(y.squeeze())
            
            pred_cls = torch_to_numpy(pred.argmax(1))
            print ("pred: ",dc[int(pred_cls)])
            print ("gd: ",dc[y])
            
            with torch.no_grad():
                pred_org = model_lock(TX)
                best_concepts, most_important_concepts = NMF_EXCUTOR(all_crops[y],all_crops_u[y],all_craft[y],X,y,all_ts[y],idx=batch,tar_zone=int_target,concepts_out=False,dataset=dataset_sw)
                for bc in best_concepts:
                    bc -= bc.min()
                    bc /= bc.max()
                    bc *= 255.0 

                most_important_concepts = torch.tensor(most_important_concepts).to(device)
                image_features = torch.stack([cmodel.encode_image(preprocess(Image.fromarray(bc.astype(np.uint8))).unsqueeze(0).to(device)).squeeze() for bc in best_concepts], 0)
                image_features = image_features.T * most_important_concepts
                image_features = torch.tensor(image_features.T,dtype=torch.float16)
                image_features /= image_features.norm(dim=-1, keepdim=True)
    
                score = (100 * image_features @ text_features.T).softmax(dim=-1)
                score = torch.mean(score,dim=0) * 100
                # print (score.shape)
                top_k = score.to("cpu").numpy().argsort()[-5:][::-1]
                print (np.array(cb)[top_k])
                print (score.to("cpu").numpy()[top_k])
                print (score.to("cpu").numpy())
                
                if not (score.shape == 2):
                    score = score.unsqueeze(0)
                score = torch.tensor(score,dtype=torch.float32)
                cbm_pred = predictor(score)
                
            # class level
            soft_cbm_pred = torch.nn.functional.softmax(cbm_pred/2, dim=1)
            soft_pred = torch.nn.functional.softmax(pred/1.5, dim=1)
            soft_pred_org = torch.nn.functional.softmax(pred_org/1, dim=1)
            
            tar_cbm_pred = torch.zeros((1,class_nums_dict[dataset_sw]),dtype=torch.float32).to("cuda")
            tar_cbm_pred[0][:] = soft_pred_org[0][:] 
            tar_cbm_pred[0][tar_cls] = 0.
            pr = 1 - tar_cbm_pred.sum()
            tar_cbm_pred[0][tar_cls] = soft_cbm_pred[0][:] * pr
            
            cbm_pred_cls = torch_to_numpy(tar_cbm_pred.argmax(1))
            print ("cbm pred: ",dc[int(cbm_pred_cls)])
            print ("gd: ",dc[y])
            # distillation loss
            loss = pt_categorical_crossentropy(soft_pred,tar_cbm_pred)
            # print (cbm_pred)
            # print (pred)
            # print (Ty)
            # print (loss)
            mean_loss += loss/size
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if batch % 100 == 0:
            loss = loss.item()
            current += len(X)
            mean_loss = mean_loss.item()
            print ("black box logits : ", pred.detach().to("cpu").numpy().squeeze())
            print ("target logits : ", tar_cbm_pred.detach().to("cpu").numpy().squeeze())
            # print ("black box softmax : ", nn.functional.softmax(pred,dim=1).detach().to("cpu").numpy().squeeze())
            # print ("cbm softmax : ", nn.functional.softmax(cbm_pred,dim=1).detach().to("cpu").numpy().squeeze())
            # print ("black box pred : ", np.argmax(pred.detach().to("cpu").numpy().squeeze()))
            # print ("cbm pred : ", np.argmax(cbm_pred.detach().to("cpu").numpy().squeeze()))
            print(f"loss: {loss:>7f} mloss: {mean_loss:>7f}  [{current:>5d}/ - ]")

        if (global_loss > mean_loss):
            torch.save(model.state_dict(),"weights/{}/latest_{}_random_intervented_{}_cls_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw,loss_type,str(tar_cls)[1:-1]))
            global_loss = mean_loss
            print ("saved weights for ep {} / Loss {:.3f}.".format(ep,mean_loss))
        
        # val
        mean_loss = 0.
        current = 0
        size = len(ds["test"])
        
        P_B = 0
        # P_CBM = 0
        P_CLS = np.zeros((class_nums_dict[dataset_sw],))
        P_CLS_A = np.zeros((class_nums_dict[dataset_sw],))
        for batch, (X, y) in enumerate(ds["test"]):
            with torch.no_grad():
                if not (len(X.shape)==4):
                    X = np.expand_dims(X,axis=0)
                TX, Ty = torch.tensor(X).to('cuda'), torch.tensor(y).to('cuda')
                TX = TX.permute(0, 3, 1, 2)
                pred = model(TX)
                y = int(y.squeeze())
                
                pred_cls = torch_to_numpy(pred.argmax(1))
                print ("pred: ",dc[int(pred_cls)])
                print ("gd: ",dc[y])
                
                if (pred_cls == y):
                    P_B += 1
                    P_CLS[y] += 1
                    
                P_CLS_A[y] += 1
                
                print ("black box logits : ", pred.to("cpu").numpy().squeeze())
                
        print ("black box ACC : {:.2f}%".format(100*P_B/size))
        # print ("CBM ACC : {:.2f}%".format(100*P_CBM/size))
        print ("black box ACC FOR TAR CLASS :",(P_CLS/P_CLS_A)[tar_cls])
        print ("black box ACC FOR CLASS :",P_CLS/P_CLS_A)

# Global Approximation
def CKD_TRAIN_ALL(epochs=200,loss_type="mse",load_predictor=False,kd_target="val"):
    # 1. train data prepare for linear predictor tar_cls
    imgs,labs,cd,dc = DataLoader(zone=[kd_target])
    batch_size=1 # must be setted to 1
    ds = {}
    for dt in [kd_target]:
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
    del imgs,labs
    with open("Dataset/{}/selected_concepts.json".format(dataset_sw),'r') as f:
        cb = json.loads(f.read())
    f.close() 
    # 2. prepare models
    model.load_state_dict(torch.load("weights/{}/latest_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw)))
    model.eval().to("cuda")
    
    predictor = PREDICTOR(len(cb),len(tar_cls))
    if (load_predictor):
        predictor.load_state_dict(torch.load("weights/{}/latest_{}_predictor_nl_{}_cls_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw,loss_type,"all")))
    predictor.train().to("cuda")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmodel, preprocess = clip.load('ViT-B/32', device)
    cbt = clip.tokenize(cb).to(device)
    print (len(cbt))
    with torch.no_grad():
        text_features = cmodel.encode_text(cbt)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    del cbt
    # print (text_features.shape)
    # 3. pre-compute
    all_crops = []
    all_crops_u = []
    all_w = []
    all_craft = []
    all_ts = []
    for c_idx in range(class_nums_dict[dataset_sw]):   
        crops, crops_u, w, craft, ts = NMF_EXTRACTOR(model,target_cls=c_idx,dataset=dataset_sw,arch=net_sw)
        all_crops.append(crops)
        all_crops_u.append(crops_u)
        all_w.append(w)
        all_craft.append(craft)
        all_ts.append(ts)
    # 4. train procedure
    # loss_kl = nn.KLDivLoss(reduction="batchmean")
    # loss_cl = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()
    
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-4)
    global_loss = 99999999999.
    for ep in range(epochs):
        mean_loss = 0.
        current = 0
        size = len(ds[kd_target])
        p_A = 1
        blance = 0
        for batch, (X, y) in enumerate(ds[kd_target]):
            # p_A +=1
            # print (X.shape)
            # print (y.shape)
            if not (len(X.shape)==4):
                X = np.expand_dims(X,axis=0)
            TX = torch.tensor(X).to('cuda')
            TX = TX.permute(0, 3, 1, 2)
            # exit()
            # Compute prediction error
            pred = model(TX)
            # loss = loss_fn(pred, y)
            # mean_loss += loss/size
            y = int(y.squeeze())
            # Ty = torch.tensor(tar_cls.index(y)).to('cuda')
            # Ty = Ty.unsqueeze(0)
            best_concepts, most_important_concepts = NMF_EXCUTOR(all_crops[y],all_crops_u[y],all_craft[y],X,y,all_ts[y],idx=batch,tar_zone=kd_target,concepts_out=False,dataset=dataset_sw)
            for bc in best_concepts:
                bc -= bc.min()
                bc /= bc.max()
                bc *= 255.0
            with torch.no_grad():
                most_important_concepts = torch.tensor(most_important_concepts).to(device)
                image_features = torch.stack([cmodel.encode_image(preprocess(Image.fromarray(bc.astype(np.uint8))).unsqueeze(0).to(device)).squeeze() for bc in best_concepts], 0)
                image_features = image_features.T * most_important_concepts
                image_features = torch.tensor(image_features.T,dtype=torch.float16)
                image_features /= image_features.norm(dim=-1, keepdim=True)
    
            score = (100 * image_features @ text_features.T).softmax(dim=-1)
            score = torch.mean(score,dim=0) * 100
            # print (score.shape)
            top_k = score.to("cpu").numpy().argsort()[-5:][::-1]
            print (np.array(cb)[top_k])
            print (score.to("cpu").numpy()[top_k])
            print (score.to("cpu").numpy())
            
            pred_cls = torch_to_numpy(pred.argmax(1))
            print ("pred: ",dc[int(pred_cls)])
            print ("gd: ",dc[y])
                
            if not (score.shape == 2):
                score = score.unsqueeze(0)
            score = torch.tensor(score,dtype=torch.float32)
            cbm_pred = predictor(score)
            tar_pred = pred
            # distillation loss
            loss = loss_mse(cbm_pred,tar_pred)
            # print (cbm_pred)
            # print (pred)
            # print (Ty)
            # print (loss)
            mean_loss += loss/p_A
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if batch % 100 == 0:
            loss = loss.item()
            current += len(X)
            mean_loss = mean_loss.item()
            print ("black box logits : ", pred.detach().to("cpu").numpy().squeeze())
            print ("cbm logits : ", cbm_pred.detach().to("cpu").numpy().squeeze())
            print(f"loss: {loss:>7f} mloss: {mean_loss:>7f}  [{current:>5d}/ - ]")

        if (global_loss > mean_loss):
            torch.save(predictor.state_dict(),"weights/{}/latest_{}_predictor_nl_{}_cls_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw,loss_type,"all"))
            global_loss = mean_loss
            print ("saved weights for ep {} / Loss {:.3f}.".format(ep,mean_loss))

def CCBM_INTERVENT_ALL(loss_type="mse", inter_concepts_nums=10, int_target="val"):
    # 1. train data prepare for linear predictor
    imgs,labs,cd,dc = DataLoader(zone=[int_target])
    batch_size=1 # must be set to 1
    ds = {}
    for dt in [int_target]:
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
    del imgs,labs
    with open("Dataset/{}/selected_concepts.json".format(dataset_sw),'r') as f:
        cb = json.loads(f.read())
    f.close() 
    # 2. prepare models
    model.load_state_dict(torch.load("weights/{}/latest_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw)))
    model.eval().to("cuda")
    
    predictor = PREDICTOR(len(cb),len(tar_cls))
    predictor.load_state_dict(torch.load("weights/{}/latest_{}_predictor_nl_{}_cls_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw,loss_type,"all")))
    predictor.train().to("cuda")
    
    predictor_org = PREDICTOR(len(cb),len(tar_cls))
    predictor_org.load_state_dict(torch.load("weights/{}/latest_{}_predictor_nl_{}_cls_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw,loss_type,"all")))
    predictor_org.train().to("cuda")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmodel, preprocess = clip.load('ViT-B/32', device)
    cbt = clip.tokenize(cb).to(device)
    print (len(cbt))
    with torch.no_grad():
        text_features = cmodel.encode_text(cbt)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    del cbt
    # print (text_features.shape)
    # 3. pre-compute
    all_crops = []
    all_crops_u = []
    all_w = []
    all_craft = []
    all_ts = []
    for c_idx in range(class_nums_dict[dataset_sw]):   
        crops, crops_u, w, craft, ts = NMF_EXTRACTOR(model,target_cls=c_idx,dataset=dataset_sw,arch=net_sw)
        all_crops.append(crops)
        all_crops_u.append(crops_u)
        all_w.append(w)
        all_craft.append(craft)
        all_ts.append(ts)
        
    size = len(ds[int_target])
    current = 0
    cls_act = np.zeros((class_nums_dict[dataset_sw],len(cb)),dtype=np.float32)
    pf_act = np.zeros((class_nums_dict[dataset_sw],len(cb)),dtype=np.float32)
    # cctp_vote = np.zeros((len(cb),),dtype=np.float32)
    for batch, (X, y) in enumerate(ds[int_target]):    
        # if not (int(y.squeeze()) in tar_cls):
        #     continue
        # cctp = torch.zeros((len(cb),),dtype=torch.float32).to(device)
        if not (len(X.shape)==4):
            X = np.expand_dims(X,axis=0)
        TX, Ty = torch.tensor(X).to('cuda'), torch.tensor(y).to('cuda')
        TX = TX.permute(0, 3, 1, 2)
        y = int(y.squeeze())
        best_concepts, most_important_concepts = NMF_EXCUTOR(all_crops[y],all_crops_u[y],all_craft[y],X,y,all_ts[y],idx=batch,tar_zone=int_target,concepts_out=False,dataset=dataset_sw)
        for bc in best_concepts:
            bc -= bc.min()
            bc /= bc.max()
            bc *= 255.0
        with torch.no_grad():
            most_important_concepts = torch.tensor(most_important_concepts).to(device)
            image_features = torch.stack([cmodel.encode_image(preprocess(Image.fromarray(bc.astype(np.uint8))).unsqueeze(0).to(device)).squeeze() for bc in best_concepts], 0)
            image_features = image_features.T * most_important_concepts
            image_features = torch.tensor(image_features.T,dtype=torch.float16)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        score = (100 * image_features @ text_features.T).softmax(dim=-1)
        score = torch.mean(score,dim=0) * 100
        if not (score.shape == 2):
            score = score.unsqueeze(0)
        score = torch.tensor(score,dtype=torch.float32,requires_grad=True)
        cbm_pred = predictor(score)
        cbm_pred_cls  = np.argmax(cbm_pred.squeeze().detach().to("cpu").numpy())
        
        # grad-based concepts
        cbm_pred.squeeze()[y].backward(retain_graph=True)
        act = score[0] * predictor.weights.grad[:,y] * predictor.weights[:,y] 
        cls_act[y] += act.detach().to("cpu").numpy().squeeze() 
        predictor.weights.grad.data.zero_()
        if not (int(cbm_pred_cls) == y):
            cbm_pred.squeeze()[int(cbm_pred_cls)].backward()
            act = score[0] * predictor.weights.grad[:,int(cbm_pred_cls)] * predictor.weights[:,int(cbm_pred_cls)] 
            pf_act[int(cbm_pred_cls)] += act.detach().to("cpu").numpy().squeeze() 
            predictor.weights.grad.data.zero_()
        else:
            cbm_pred.squeeze()[int(cbm_pred_cls)].backward()
            act = score[0] * predictor.weights.grad[:,int(cbm_pred_cls)] * predictor.weights[:,int(cbm_pred_cls)] 
            pf_act[int(cbm_pred_cls)] -= act.detach().to("cpu").numpy().squeeze()
            predictor.weights.grad.data.zero_()
        
    current += len(X)
    print(f"[{current:>5d}/ - ]")
    
    mask = torch.ones((len(cb),class_nums_dict[dataset_sw]),dtype=torch.float32).to(device)
    for i in range(class_nums_dict[dataset_sw]):
        intervent_concepts_nT = cls_act[i].argsort()[:inter_concepts_nums//2]
        print ("class <{}> nf : ".format(i), np.array(cb)[intervent_concepts_nT])
        intervent_concepts_pF = pf_act[i].argsort()[::-1][:inter_concepts_nums//2]
        print ("class <{}> pf : ".format(i), np.array(cb)[intervent_concepts_pF])
        # intervent_concepts = np.concatenate([intervent_concepts_nT,intervent_concepts_pF])
        for ic in intervent_concepts_nf:
            # hard mask
            if (cls_act[i][ic] < 0):
                mask[ic,i] = 0 
                
        for ic in intervent_concepts_pf:
            # hard mask
            if (pf_act[i][ic] > 0):
                mask[ic,i] = 0 
    
    predictor.weights = torch.nn.Parameter(predictor.weights * mask) # wij~ = 0
    torch.save(predictor.state_dict(),"weights/{}/latest_{}_intervented_predictor_nl_{}_cls_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw,loss_type,"all"))

def RCKD_TRAIN_ALL(epochs=10,loss_type="mse", int_target="val"):
    # 1. train data prepare for linear predictor
    imgs,labs,cd,dc = DataLoader(zone=[int_target,"test"])
    batch_size=1 # must be setted to 1
    ds = {}
    for dt in [int_target,"test"]:
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
    del imgs,labs
    with open("Dataset/{}/selected_concepts.json".format(dataset_sw),'r') as f:
        cb = json.loads(f.read())
    f.close() 
    # 2. prepare models
    model.load_state_dict(torch.load("weights/{}/latest_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw)))
    model.eval().to("cuda")
    
    model_lock.load_state_dict(torch.load("weights/{}/latest_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw)))
    model_lock.eval().to("cuda")
    
    predictor = PREDICTOR(len(cb),len(tar_cls))
    predictor.load_state_dict(torch.load("weights/{}/latest_{}_random_intervented_predictor_nl_{}_cls_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw,loss_type,"all")))
    predictor.train().to("cuda")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmodel, preprocess = clip.load('ViT-B/32', device)
    cbt = clip.tokenize(cb).to(device)
    print (len(cbt))
    with torch.no_grad():
        text_features = cmodel.encode_text(cbt)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    del cbt
    # print (text_features.shape)
    # 3. pre-compute
    all_crops = []
    all_crops_u = []
    all_w = []
    all_craft = []
    all_ts = []
    for c_idx in range(class_nums_dict[dataset_sw]):   
        crops, crops_u, w, craft, ts = NMF_EXTRACTOR(model_lock,target_cls=c_idx,dataset=dataset_sw,arch=net_sw)
        all_crops.append(crops)
        all_crops_u.append(crops_u)
        all_w.append(w)
        all_craft.append(craft)
        all_ts.append(ts)
    # 4. train procedure
    # loss_kl = nn.KLDivLoss(reduction="batchmean")
    # loss_cl = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-7)
    global_loss = 99999999999.
    for ep in range(epochs):
        mean_loss = 0.
        current = 0
        size = len(ds[int_target])
        for batch, (X, y) in enumerate(ds[int_target]):
            # if not (int(y.squeeze()) in tar_cls):
            #     continue
            # print (X.shape)
            # print (y.shape)
            if not (len(X.shape)==4):
                X = np.expand_dims(X,axis=0)
            TX, Ty = torch.tensor(X).to('cuda'), torch.tensor(y).to('cuda')
            TX = TX.permute(0, 3, 1, 2)
            # exit()
            # Compute prediction error
            pred = model(TX)
            # loss = loss_fn(pred, y)
            # mean_loss += loss/size
            y = int(y.squeeze())
            
            pred_cls = torch_to_numpy(pred.argmax(1))
            print ("pred: ",dc[int(pred_cls)])
            print ("gd: ",dc[y])
            
            with torch.no_grad():
                pred_org = model_lock(TX)
                best_concepts, most_important_concepts = NMF_EXCUTOR(all_crops[y],all_crops_u[y],all_craft[y],X,y,all_ts[y],idx=batch,tar_zone=int_target,concepts_out=False,dataset=dataset_sw)
                for bc in best_concepts:
                    bc -= bc.min()
                    bc /= bc.max()
                    bc *= 255.0 

                most_important_concepts = torch.tensor(most_important_concepts).to(device)
                image_features = torch.stack([cmodel.encode_image(preprocess(Image.fromarray(bc.astype(np.uint8))).unsqueeze(0).to(device)).squeeze() for bc in best_concepts], 0)
                image_features = image_features.T * most_important_concepts
                image_features = torch.tensor(image_features.T,dtype=torch.float16)
                image_features /= image_features.norm(dim=-1, keepdim=True)
    
                score = (100 * image_features @ text_features.T).softmax(dim=-1)
                score = torch.mean(score,dim=0) * 100
                # print (score.shape)
                top_k = score.to("cpu").numpy().argsort()[-5:][::-1]
                print (np.array(cb)[top_k])
                print (score.to("cpu").numpy()[top_k])
                print (score.to("cpu").numpy())
                
                if not (score.shape == 2):
                    score = score.unsqueeze(0)
                score = torch.tensor(score,dtype=torch.float32)
                cbm_pred = predictor(score)
                
            # class level
            soft_cbm_pred = torch.nn.functional.softmax(cbm_pred/2, dim=1)
            soft_pred = torch.nn.functional.softmax(pred/1.5, dim=1)
    
            tar_cbm_pred = soft_cbm_pred
            
            cbm_pred_cls = torch_to_numpy(tar_cbm_pred.argmax(1))
            print ("cbm pred: ",dc[int(cbm_pred_cls)])
            print ("gd: ",dc[y])
            # distillation loss
            loss = pt_categorical_crossentropy(soft_pred,tar_cbm_pred)
            # print (cbm_pred)
            # print (pred)
            # print (Ty)
            # print (loss)
            mean_loss += loss/size
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if batch % 100 == 0:
            loss = loss.item()
            current += len(X)
            mean_loss = mean_loss.item()
            print ("black box logits : ", pred.detach().to("cpu").numpy().squeeze())
            print ("target logits : ", tar_cbm_pred.detach().to("cpu").numpy().squeeze())
            # print ("black box softmax : ", nn.functional.softmax(pred,dim=1).detach().to("cpu").numpy().squeeze())
            # print ("cbm softmax : ", nn.functional.softmax(cbm_pred,dim=1).detach().to("cpu").numpy().squeeze())
            # print ("black box pred : ", np.argmax(pred.detach().to("cpu").numpy().squeeze()))
            # print ("cbm pred : ", np.argmax(cbm_pred.detach().to("cpu").numpy().squeeze()))
            print(f"loss: {loss:>7f} mloss: {mean_loss:>7f}  [{current:>5d}/ - ]")

        if (global_loss > mean_loss):
            torch.save(model.state_dict(),"weights/{}/latest_{}_intervented_{}_cls_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw,loss_type,"all"))
            global_loss = mean_loss
            print ("saved weights for ep {} / Loss {:.3f}.".format(ep,mean_loss))
        
        # val
        mean_loss = 0.
        current = 0
        size = len(ds["test"])
        
        P_B = 0
        # P_CBM = 0
        P_CLS = np.zeros((class_nums_dict[dataset_sw],))
        P_CLS_A = np.zeros((class_nums_dict[dataset_sw],))
        for batch, (X, y) in enumerate(ds["test"]):
            with torch.no_grad():
                if not (len(X.shape)==4):
                    X = np.expand_dims(X,axis=0)
                TX, Ty = torch.tensor(X).to('cuda'), torch.tensor(y).to('cuda')
                TX = TX.permute(0, 3, 1, 2)
                pred = model(TX)
                y = int(y.squeeze())
                
                pred_cls = torch_to_numpy(pred.argmax(1))
                print ("pred: ",dc[int(pred_cls)])
                print ("gd: ",dc[y])
                
                if (pred_cls == y):
                    P_B += 1
                    P_CLS[y] += 1
                    
                P_CLS_A[y] += 1
                
                print ("black box logits : ", pred.to("cpu").numpy().squeeze())
                
        print ("black box ACC : {:.2f}%".format(100*P_B/size))

if __name__ == "__main__":
    
    # print ("DBG!")
    # print ("{}".format(len(sys.argv)))
    # print ("{} {}".format(sys.argv[4],PROCESS_TARCLS(sys.argv[5])))
    
    if (len(sys.argv) < 5):        
        print ("Help :")
        print ("For local  intervention : python CBM-HNMU.py <net_sw> <dataset_sw> <data_root> <ap/ci/kt> <tar_cls> <ic_nums>")
        print ("For random intervention : python CBM-HNMU.py <net_sw> <dataset_sw> <data_root> <rci/rkt> <tar_cls> <ic_nums>")
        print ("For global intervention : python CBM-HNMU.py <net_sw> <dataset_sw> <data_root> <aap/aci/akt> <ic_nums>")
        exit(0)
    elif (len(sys.argv) == 5) and (not (sys.argv[4] in ["aap","aci","akt"])):
        print ("Help :")
        print ("For local  intervention : python CBM-HNMU.py <net_sw> <dataset_sw> <data_root> <ap/ci/kt> <tar_cls> <ic_nums>")
        print ("For random intervention : python CBM-HNMU.py <net_sw> <dataset_sw> <data_root> <rci/rkt> <tar_cls> <ic_nums>")
        print ("For global intervention : python CBM-HNMU.py <net_sw> <dataset_sw> <data_root> <aap/aci/akt> <ic_nums>")
        exit(0)
    else:
        if (sys.argv[4] in ["aap","aci","akt"]):
            if (sys.argv[4] == "aap"):
                CKD_TRAIN_ALL()
            elif (sys.argv[4] == "aci"):
                if (len(sys.argv) > 5):
                    CCBM_INTERVENT_ALL(inter_concepts_nums=PROCESS_TARCLS(sys.argv[5]))
                else:
                    CCBM_INTERVENT_ALL(inter_concepts_nums=10)
            else:
                RCKD_TRAIN_ALL()
                
        if (sys.argv[4] in ["rci","rkt"]):
            if (len(sys.argv) < 6):
                print ("Help :")
                print ("For local  intervention : python CBM-HNMU.py <net_sw> <dataset_sw> <data_root> <ap/ci/kt> <tar_cls> <ic_nums>")
                print ("For random intervention : python CBM-HNMU.py <net_sw> <dataset_sw> <data_root> <rci/rkt> <tar_cls> <ic_nums>")
                print ("For global intervention : python CBM-HNMU.py <net_sw> <dataset_sw> <data_root> <aap/aci/akt> <ic_nums>")
                exit(0)
            if (sys.argv[4] == "rci"):
                if (len(sys.argv) > 6):
                    CCBM_INTERVENT_RANDOM(inter_concepts_nums==sys.argv[6],tar_cls=PROCESS_TARCLS(sys.argv[5]))
                else:
                    CCBM_INTERVENT_RANDOM(inter_concepts_nums==10,tar_cls=PROCESS_TARCLS(sys.argv[5]))
            else:
                RANDOM_RCKD_TRAIN(tar_cls=PROCESS_TARCLS(sys.argv[5]))
        
        if (sys.argv[4] in ["ap","ci","kt"]):
            if (len(sys.argv) < 6):
                print ("Help :")
                print ("For local  intervention : python CBM-HNMU.py <net_sw> <dataset_sw> <data_root> <ap/ci/kt> <tar_cls> <ic_nums>")
                print ("For random intervention : python CBM-HNMU.py <net_sw> <dataset_sw> <data_root> <rci/rkt> <tar_cls> <ic_nums>")
                print ("For global intervention : python CBM-HNMU.py <net_sw> <dataset_sw> <data_root> <aap/aci/akt> <ic_nums>")
                exit(0)
            if (sys.argv[4] == "ap"):
                CKD_TRAIN(tar_cls=PROCESS_TARCLS(sys.argv[5]))
            elif (sys.argv[4] == "ci"):
                if (len(sys.argv) > 6):
                    CCBM_INTERVENT_nT_with_pF(inter_concepts_nums==sys.argv[6],tar_cls=PROCESS_TARCLS(sys.argv[5]))
                else:
                    CCBM_INTERVENT_nT_with_pF(inter_concepts_nums==10,tar_cls=PROCESS_TARCLS(sys.argv[5]))
            else:
                RCKD_TRAIN(tar_cls=PROCESS_TARCLS(sys.argv[5]))
        