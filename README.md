# ICCV 2025
## Intervening in Black Box: Concept Bottleneck Model for Enhancing Human Neural Network Mutual Understanding (CBM-HNMU)
![CBM-HNMU Arch Figure 1](https://github.com/XiGuaBo/CBM-HNMU/blob/main/post_figs/figure1_own.png "CBM-HNMU Arch Figure 1")  
![CBM-HNMU Arch Figure 2](https://github.com/XiGuaBo/CBM-HNMU/blob/main/post_figs/figure1_own_v1.png "CBM-HNMU Arch Figure 2")

# requirement
clip==1.0  
jax==0.4.31  
jaxopt==0.8.3  
matplotlib==3.5.3  
numpy==1.19.5  
opencv_python==4.3.0.38  
Pillow==9.3.0  
Pillow==10.4.0  
scikit_learn==1.0.2  
scipy==1.7.3  
setuptools==65.6.3  
tensorflow_gpu==2.4.0  
timm==0.6.12  
torch==1.7.1  
torch_summary==1.4.5  
torchvision==0.8.2  

These are the main packages needed to be installed. For detailed, please refer to the requirement.txt.  
We will also provide the integrated environment of Anaconda3 in the future.  

# Dependence Url
Clip  : [url:] https://github.com/openai/CLIP  
Craft : [url:] https://github.com/deel-ai/Craft  

# Datasets
Please reference the README.md in Dataset.  

# Train Baselines (Fine-Tune)
net switch : ["nfresnet50" , "vit" , "resnext26" , "botnet26t" , "rexnet100" , "gcvit" , "deit" , "convit" , "cait"]  
dataset switch : ["flower102" , "cifar10" , "cifar100" , "cub" , "aircraft"]  

python train_base.py <net_sw> <dataset_sw> <data_root>  
eg: python train_base.py nfresnet50 flower102 ../YOUR_FLODER/Dataset  

# Confusing Categories Selection
python Reasonable.py <net_sw> <dataset_sw> <data_root> <cc_select>  

# Local Approximation
python CBM-HNMU.py <net_sw> <dataset_sw> <data_root> <ap> <tar_cls> <opt:ic_nums>  

# Concepts Intervention
python CBM-HNMU.py <net_sw> <dataset_sw> <data_root> <ci> <tar_cls> <opt:ic_nums>  

# Knowledge Transfer 
python CBM-HNMU.py <net_sw> <dataset_sw> <data_root> <kt> <tar_cls> <opt:ic_nums>  

# Visualization
python Reasonable.py <net_sw> <dataset_sw> <data_root> <reasonable> <tar_cls>  
