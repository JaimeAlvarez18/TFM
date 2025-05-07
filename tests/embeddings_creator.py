import sys
from utils.data_acquisition import data_set_with_nature,images_Dataset,test_dataset,data_set_binary_with_nature,data_set_binary_synth
import torch
from torch.utils.data import DataLoader
device = "cuda" if torch.cuda.is_available() else "cpu"
import gc
from utils.data_acquisition import create_and_save_ALL_embeddings
from sklearn.neighbors import KNeighborsClassifier


from tqdm import tqdm
from utils.models import Big_model,siamese_model
from torch import nn
import numpy as np
from sklearn.metrics import confusion_matrix
import torch.amp as amp
import time

if __name__ == "__main__":
    if len(sys.argv) == 2:
        route_encoder=sys.argv[1]
        
    BATCH_SIZE=182
    RESOLUTION=256
    MARGIN=1
    EMBEDDING_SIZE=128
    EFFICIENTNET_TYPE="efficientnet-b0"
    LOSS="SupConLoss"
    CLASSES=2
    path_embeddings=f'Models/Embeddings/embeddings_{EMBEDDING_SIZE}_{BATCH_SIZE}_SYNTH_{LOSS}_TODOS.npz'
    
    loader_data = data_set_with_nature('Datasets/GenImage/')
    train,val,test,y_train,y_val,y_test = loader_data.get_data()
    # train,y_train = loader_data.get_data()
    
    print("Creating dataloaders")
    train_dataset=test_dataset(train,y_train,device,RESOLUTION)
    train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=16,prefetch_factor=8)

    # test_data=test_dataset(test,y_test,device,RESOLUTION)
    # test_dataloader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=16,prefetch_factor=8)
    
    checkpoint=torch.load(route_encoder,weights_only=False)
    model= siamese_model(checkpoint["model_type"],device,EMBEDDING_SIZE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    embs1,classes =  create_and_save_ALL_embeddings(model,train_dataloader,path_embeddings)