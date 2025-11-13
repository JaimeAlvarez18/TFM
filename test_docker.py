import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

for i in range(torch.cuda.device_count()):
    print(i,torch.cuda.get_device_name(i))
device ='cuda' if torch.cuda.is_available() else 'cpu'
import gc         
import torch.amp as amp
import math

# from utils.data_acquisition import data_set_N_with_nature,BIG_DATALOADER, read_files,test_dataset,data_set_binary_synth
# from utils.models import siamese_model
# from utils.losses import ContrastiveLoss,SupConLoss
import time
import numpy as np

from einops import rearrange

import sys
import copy

import numpy as np
from glob import glob
import random
random.seed(42)
import os
from torch.utils.data import Dataset
# import cv2
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import torch.amp as amp
device='cuda' if torch.cuda.is_available() else 'cpu'
import json
import gc  
import sys
# from utils.data_acquisition import data_set,images_Dataset,test_dataset,data_set_with_nature,data_set_binary_with_nature,data_set_binary_synth
import torch
from torch.utils.data import DataLoader
device = "cuda" if torch.cuda.is_available() else "cpu"
import gc
# from utils.data_acquisition import create_and_save_ALL_embeddings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score,roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


from tqdm import tqdm
from utils.models import Big_model,siamese_model
from torch import nn
import numpy as np
from sklearn.metrics import confusion_matrix
import torch.amp as amp
import time
import transformers
print("hola")