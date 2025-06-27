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
    model_int = NFRES50(class_nums_dict[dataset_sw])
    sample_size = 256
elif (net_sw == "resnext26"):
    model = RESNEXT26(class_nums_dict[dataset_sw])
    model_int = RESNEXT26(class_nums_dict[dataset_sw])
    sample_size = 256
elif (net_sw == "botnet26t"):
    model = BOTNET26t(class_nums_dict[dataset_sw])
    model_int = BOTNET26t(class_nums_dict[dataset_sw])
    sample_size = 256
elif (net_sw == "rexnet100"):
    model = REXNET100(class_nums_dict[dataset_sw])
    model_int = REXNET100(class_nums_dict[dataset_sw])
    sample_size = 256
elif (net_sw == "vit"):
    model = Vit(class_nums_dict[dataset_sw])
    model_int = Vit(class_nums_dict[dataset_sw])
    sample_size = 224
elif (net_sw == "gcvit"):
    model = GCVIT(class_nums_dict[dataset_sw])
    model_int = GCVIT(class_nums_dict[dataset_sw])
    sample_size = 224
elif (net_sw == "cait"):
    model = Cait_Small(class_nums_dict[dataset_sw])
    model_int = Cait_Small(class_nums_dict[dataset_sw])
    sample_size = 224
elif (net_sw == "convit"):
    model = Convit_Base(class_nums_dict[dataset_sw])
    model_int = Convit_Base(class_nums_dict[dataset_sw])
    sample_size = 224
elif (net_sw == "deit"):
    model = Deit_Base(class_nums_dict[dataset_sw])
    model_int = Deit_Base(class_nums_dict[dataset_sw])
    sample_size = 224
else:
    # no choice will switch to nfresnet50 and flower102
    model = NFRES50(class_nums_dict[dataset_sw])
    model_int = NFRES50(class_nums_dict[dataset_sw])
    sample_size = 256
    
model.load_state_dict(torch.load("weights/{}/latest_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw)))
model.eval().to("cuda")
    
def PROCESS_TARCLS(tar_cls_str):
    tar_cls = []
    for cls in tar_cls_str.split(','):
        tar_cls.append(int(cls))
    return tar_cls
    
def FIND_MIXED_SET(target="val"):
    # 1. test data prepare for linear predictor
    imgs,labs,cd,dc = DataLoader(data_format=df_dict[dataset_sw],zone=[target],data_root=data_root)
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
    
    # val
    current = 0
    size = len(ds[target])
    P_B = 0
    P_CLS = np.zeros((class_nums_dict[dataset_sw],))
    P_CLS_A = np.zeros((class_nums_dict[dataset_sw],))
    
    false_positive_map = np.zeros((class_nums_dict[dataset_sw],class_nums_dict[dataset_sw]))
    
    for batch, (X, y) in enumerate(ds[target]):
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
            
            current += len(X)
            
            if not (y == int(pred_cls)):
                false_positive_map[y][int(pred_cls)] += 1
            
            print(f"[{current:>5d}/ - ]")
            
    print ("black box ACC : {:.2f}%".format(100*P_B/size))
    print ("black box ACC FOR CLASS :",P_CLS/P_CLS_A)
    
    print ("Most False Positive Class : ", np.sum(false_positive_map,axis=0).squeeze())
    print ("Most False Positive Class-Top-4 : ", np.sum(false_positive_map,axis=0).squeeze().argsort()[::-1][:4])
    print ("False Positive Nums : ", np.sum(false_positive_map,axis=0).squeeze()[np.sum(false_positive_map,axis=0).squeeze().argsort()[::-1][:4]])
    
    for n in range(class_nums_dict[dataset_sw]):
        false_positive_map[n,n] = (P_CLS/P_CLS_A)[n]
    # np.save("fp",false_positive_map)
    
    mixed_pair = {}
    for s in range(class_nums_dict[dataset_sw]):
        for d in range(s,class_nums_dict[dataset_sw]):
            if (s==d):
                continue
            mixed_degree = (false_positive_map[s][d] + false_positive_map[d][s])/2
            key = (s,d)
            mixed_pair[key] = mixed_degree
    seq = sorted(mixed_pair,key=lambda x: mixed_pair[x],reverse=True)[:5]
    print (seq)
    for s in seq:
        print (mixed_pair[s])
    
def REASONABLE(target="test",loss_type="mse",tar_cls=None):
    # 1. train data prepare for linear predictor
    imgs,labs,cd,dc = DataLoader(data_format=df_dict[dataset_sw],zone=[target],data_root=data_root)
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
    model_int.load_state_dict(torch.load("weights/{}/latest_{}_intervented_{}_cls_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw,"mse",str(tar_cls)[1:-1])))
    model_int.eval().to("cuda")
    
    bb = model
    bbi = model_int
    
    p = PREDICTOR(len(cb),len(tar_cls))
    p.load_state_dict(torch.load("weights/{}/latest_{}_predictor_nl_{}_cls_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw,loss_type,str(tar_cls)[1:-1])))
    p.eval().to("cuda")
    
    pi = PREDICTOR(len(cb),len(tar_cls))
    pi.load_state_dict(torch.load("weights/{}/latest_{}_intervented_predictor_nl_{}_cls_{}_{}.pth".format(dataset_sw,dataset_sw,net_sw,loss_type,str(tar_cls)[1:-1])))
    pi.eval().to("cuda")
    
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
        crops, crops_u, w, craft, ts = NMF_EXTRACTOR(bb,target_cls=c_idx,dataset=dataset_sw,arch=net_sw)
        all_crops.append(crops)
        all_crops_u.append(crops_u)
        all_w.append(w)
        all_craft.append(craft)
        all_ts.append(ts)
        
    all_crops_i = []
    all_crops_u_i = []
    all_w_i = []
    all_craft_i = []
    all_ts_i = []
    for c_idx in range(class_nums_dict[dataset_sw]):   
        crops, crops_u, w, craft, ts = NMF_EXTRACTOR(bbi,target_cls=c_idx,dataset=dataset_sw,arch=net_sw)
        all_crops_i.append(crops)
        all_crops_u_i.append(crops_u)
        all_w_i.append(w)
        all_craft_i.append(craft)
        all_ts_i.append(ts)
        
    # val
    current = 0
    size = len(ds[target])
    P_B = 0
    P_B_I = 0
    P_CBM = 0
    P_CBM_I = 0

    P_B_CLS = np.zeros((class_nums_dict[dataset_sw],))
    P_B_CLS_A = np.zeros((class_nums_dict[dataset_sw],))
    P_B_I_CLS = np.zeros((class_nums_dict[dataset_sw],))
    P_B_I_CLS_A = np.zeros((class_nums_dict[dataset_sw],))
    P_CBM_CLS = np.zeros((class_nums_dict[dataset_sw],))
    P_CBM_CLS_A = np.zeros((class_nums_dict[dataset_sw],))
    P_CBM_I_CLS = np.zeros((class_nums_dict[dataset_sw],))
    P_CBM_I_CLS_A = np.zeros((class_nums_dict[dataset_sw],))
    
    # sample_collect = []
    bb_sample_collect = []
    # cbm_sample_collect = []
    
    # negative_bb_sample_collect = []
    # negative_concepts_match_collect = []
    
    concepts_match_collect_bb = []
    concepts_match_collect_bbi = []
    
    Coverage = np.zeros((class_nums_dict[dataset_sw],2))
    CVG = 0
    
    for batch, (X, y) in enumerate(ds[target]):
        with torch.no_grad():
            if not (len(X.shape)==4):
                X = np.expand_dims(X,axis=0)
            TX, Ty = torch.tensor(X).to('cuda'), torch.tensor(y).to('cuda')
            TX = TX.permute(0, 3, 1, 2)
            pred = bb(TX)
            pred_i = bbi(TX)
            y = int(y.squeeze())
            
            pred_cls = torch_to_numpy(pred.argmax(1))
            print ("pred: ",dc[int(pred_cls)])
            print ("gd: ",dc[y])
            
            pred_i_cls = torch_to_numpy(pred_i.argmax(1))
            print ("pred_i: ",dc[int(pred_i_cls)])
            print ("gd: ",dc[y])
            
            best_concepts, most_important_concepts = NMF_EXCUTOR(all_crops[y],all_crops_u[y],all_craft[y],X,y,all_ts[y],idx=batch,tar_zone=target,concepts_out=False,dataset=dataset_sw)
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
            cbm_pred = p(score)
            
            soft_cbm_pred = torch.nn.functional.softmax(cbm_pred/2, dim=1)
            soft_pred = torch.nn.functional.softmax(pred/1, dim=1)
        
            tar_pred = torch.zeros((1,class_nums_dict[dataset_sw]),dtype=torch.float32).to("cuda")
            tar_pred[0][:] = soft_pred[0][:] 
            tar_pred[0][tar_cls] = 0.
            pr = 1 - tar_pred.sum()
            tar_pred[0][tar_cls] = soft_cbm_pred[0][:] * pr
        
            cbm_pred_cls = torch_to_numpy(tar_pred.argmax(1))
            print ("cbm pred: ",dc[int(cbm_pred_cls)])
            print ("gd: ",dc[y])
            
            best_concepts_i, most_important_concepts_i = NMF_EXCUTOR(all_crops_i[y],all_crops_u_i[y],all_craft_i[y],X,y,all_ts_i[y],idx=batch,tar_zone=target,concepts_out=False,dataset=dataset_sw)
            for bc in best_concepts_i:
                bc -= bc.min()
                bc /= bc.max()
                bc *= 255.0 
            
            most_important_concepts_i = torch.tensor(most_important_concepts_i).to(device)
            image_features = torch.stack([cmodel.encode_image(preprocess(Image.fromarray(bc.astype(np.uint8))).unsqueeze(0).to(device)).squeeze() for bc in best_concepts_i], 0)
            image_features = image_features.T * most_important_concepts_i
            image_features = torch.tensor(image_features.T,dtype=torch.float16)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            score_i = (100 * image_features @ text_features.T).softmax(dim=-1)
            score_i = torch.mean(score,dim=0) * 100
            print (score.shape)
            top_k_i = score_i.to("cpu").numpy().argsort()[-5:][::-1]
            # print (np.array(cb)[top_k_i])
            # print (score_i.to("cpu").numpy()[top_k_i])
            # print (score_i.to("cpu").numpy())
            
            # if not (score_i.shape == 2):
            #     score_i = score_i.unsqueeze(0)
            # score_i = torch.tensor(score_i,dtype=torch.float32)
            # cbm_pred_i = pi(score_i)
            cbm_pred_i = pi(score)
            soft_cbm_pred_i = torch.nn.functional.softmax(cbm_pred_i/2, dim=1)
            
            
            tar_pred = torch.zeros((1,class_nums_dict[dataset_sw]),dtype=torch.float32).to("cuda")
            tar_pred[0][:] = soft_pred[0][:] 
            tar_pred[0][tar_cls] = 0.
            pr = 1 - tar_pred.sum()
            tar_pred[0][tar_cls] = soft_cbm_pred_i[0][:] * pr
        
            cbm_pred_i_cls = torch_to_numpy(tar_pred.argmax(1))
            print ("cbm pred i: ",dc[int(cbm_pred_i_cls)])
            print ("gd: ",dc[y])
            
            if (pred_cls == y):
                P_B += 1
                P_B_CLS[y] += 1
            if (pred_i_cls == y):
                P_B_I += 1
                P_B_I_CLS[y] += 1
            if (cbm_pred_cls == y):
                P_CBM += 1
                P_CBM_CLS[y] += 1
            if (cbm_pred_i_cls == y):
                P_CBM_I += 1
                P_CBM_I_CLS[y] += 1
                
            P_B_CLS_A[y] += 1
            P_B_I_CLS_A[y] += 1
            P_CBM_CLS_A[y] += 1
            P_CBM_I_CLS_A[y] += 1
            
            # if (int(pred_cls)==int(cbm_pred_cls)) and (not (int(pred_cls)==y)) and (int(cbm_pred_i_cls)==y):
            #     sample_collect.append([batch,int(pred_cls),dc[int(pred_cls)],y,dc[y]])
            
            if (not (int(pred_cls)==y)) and (int(pred_i_cls)==y):
                bb_sample_collect.append([batch,int(pred_cls),dc[int(pred_cls)],y,dc[y]])
                NMF_EXCUTOR(all_crops[y],all_crops_u[y],all_craft[y],X,y,all_ts[y],idx=batch,tar_zone=target,concepts_out=True,dataset=dataset_sw,fg='bb')
                NMF_EXCUTOR(all_crops_i[y],all_crops_u_i[y],all_craft_i[y],X,int(pred_i_cls),all_ts_i[y],idx=batch,tar_zone=target,concepts_out=True,dataset=dataset_sw,fg='bbi')
                concepts_match_collect_bb.append([batch,np.array(cb)[top_k].tolist()])
                concepts_match_collect_bbi.append([batch,np.array(cb)[top_k_i].tolist()])
                
                if (int(pred_cls) in tar_cls):
                    Coverage[int(pred_cls)][0] += 1
                elif (y in tar_cls):
                    Coverage[y][1] += 1
                    
                if (int(pred_cls) in tar_cls) or (y in tar_cls):
                    CVG += 1
                    
                
            if (not (int(cbm_pred_cls)==y)) and (int(cbm_pred_i_cls)==y):
                cbm_sample_collect.append([batch,int(cbm_pred_cls),dc[int(cbm_pred_cls)],y,dc[y]])
                
            # if ((int(pred_cls)==y)) and not (int(pred_i_cls)==y):
            #     negative_bb_sample_collect.append([batch,int(pred_cls),dc[int(pred_cls)],int(pred_i_cls),dc[int(pred_i_cls)]])
            #     NMF_EXCUTOR(all_crops[y],all_crops_u[y],all_craft[y],X,int(pred_cls),all_ts[y],idx=batch,tar_zone=target,concepts_out=True)
            #     # NMF_EXCUTOR(all_crops_i[y],all_crops_u_i[y],all_craft_i[y],X,int(pred_i_cls),all_ts_i[y],idx=batch,tar_zone=target,concepts_out=True)
            #     negative_concepts_match_collect.append([batch,np.array(cb)[top_k].tolist()])
            #     # negative_concepts_match_collect.append([batch,np.array(cb)[top_k_i].tolist()])
    
    # print (sample_collect)
    print (bb_sample_collect)
    print ("CP : {}".format(len(bb_sample_collect)))
    print ("Coverage for class :")
    for c in tar_cls:
        print (Coverage[c])
    print ("Coverage : {}".format(CVG))
    
    print (concepts_match_collect_bb)
    print (concepts_match_collect_bbi)
    
    # print (negative_bb_sample_collect)
    # print (negative_concepts_match_collect)
    
    bb_cls_acc = P_B_CLS/P_B_CLS_A
    bbi_cls_acc = P_B_I_CLS/P_B_I_CLS_A
    bbgap_cls_acc = np.array(bbi_cls_acc - bb_cls_acc)
    bbp = np.where(bbgap_cls_acc>0)
    bbn = np.where(bbgap_cls_acc<0)
    print ("positive cls for bb/i:",bbp)
    print ("gap : " , bbgap_cls_acc[bbp])
    print ("negetive cls for bb/i:",bbn)
    print ("gap : " , bbgap_cls_acc[bbn])
    
    print (P_B/len(ds[target]))
    print (P_B_I/len(ds[target]))
    print (bb_cls_acc[tar_cls])
    print (bbi_cls_acc[tar_cls])
    
    cbm_cls_acc = P_CBM_CLS/P_CBM_CLS_A
    cbm_i_cls_acc = P_CBM_I_CLS/P_CBM_I_CLS_A
    cbmgap_cls_acc = np.array(cbm_i_cls_acc - cbm_cls_acc)
    cbmp = np.where(cbmgap_cls_acc>0)
    cbmn = np.where(cbmgap_cls_acc<0)
    print ("positive cls for cbm:",cbmp)
    print ("gap : " , cbmgap_cls_acc[cbmp])
    print ("negetive cls for cbm:",cbmn)
    print ("gap : " , cbmgap_cls_acc[cbmn])
    
    print (P_CBM/len(ds[target]))
    print (P_CBM_I/len(ds[target]))
    print (cbm_cls_acc[tar_cls])
    print (cbm_i_cls_acc[tar_cls])
    
    for c in tar_cls:
        print ("{}-{} :".format(c,dc[c]))
        w = p.weights[:,tar_cls.index(c)].detach().to("cpu").numpy().squeeze()
        iw = np.argsort(w)[-100:]
        wi = pi.weights[:,tar_cls.index(c)].detach().to("cpu").numpy().squeeze()
        iwi = np.argsort(wi)[-100:]
        # print ("top-10 pre-intervented positive concepts:",iw)
        # print (np.array(cb)[iw])
        # print ("top-10 intervented positive concepts:",iwi)
        # print (np.array(cb)[iwi])
        # iiwi = np.where(wi==0)[0].tolist()
        # print ("all intervented concepts:",np.array(cb)[iiwi])
        icross = set(iw) - set(iw).intersection(set(iwi))
        icross = list(icross)
        icross.sort(key=lambda x: iw.tolist().index(x),reverse=True)
        print (icross)
        print ("changed concepts in top 100 : ",np.array(cb)[list(icross)])

if __name__ == "__main__":
    
    if (len(sys.argv) < 5):
        print ("Help : python Reasonable.py <net_sw> <dataset_sw> <data_root> <cc_select/reasonable> <tar_cls>")
        print ("       <tar_cls> must be set if <arg_4> set to reasonable.")
        print ("       tar_cls : cls_id,cls_id,cls_id,...,cls_id")
        exit(0)
    
    if (sys.argv[4]=="cc_select"):  
        FIND_MIXED_SET(target="val")
    elif (sys.argv[4]=="reasonable"):  
        REASONABLE(target="test",loss_type="mse",tar_cls=PROCESS_TARCLS(argv[5]))
    else:
        print ("Help : python Reasonable.py <net_sw> <dataset_sw> <data_root> <cc_select/reasonable> <tar_cls>")
        print ("       <tar_cls> must be set if <arg_4> set to reasonable.")
        print ("       tar_cls : cls_id,cls_id,cls_id,...,cls_id")
        print ("# * Please ensure weights floder exsist the target file!")