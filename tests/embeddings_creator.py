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
from sklearn.model_selection import train_test_split
def set_memory_limit(gb):
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (gb * 1024**3, hard))

if __name__ == "__main__":
    # set_memory_limit(50)
    if len(sys.argv) == 2:
        route_encoder=sys.argv[1]
        
    BATCH_SIZE=32
    RESOLUTION=256
    MARGIN=1
    EMBEDDING_SIZE=128
    EFFICIENTNET_TYPE="efficientnet-b0"
    LOSS="SupConLoss"
    CLASSES=5
    path_embeddings=f'Models/Embeddings/MINIBATCH_128_2048_ForenSynths_Partial_Checkpoint.npz'
    print(path_embeddings)
    
    
    

    
    
    # loader_data = data_set_with_nature('GenImage_resized/')
    # train,val,test,y_train,y_val,y_test = loader_data.get_data()
    # train_dataset=test_dataset(train,y_train,device,RESOLUTION)
    # train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,prefetch_factor=4)

    loader_data = data_set_binary_synth('ForenSynths')
    train,y_train,_,_,test,y_test = loader_data.get_data()
    train,val,y_train,y_val = train_test_split(train,y_train,train_size=0.2,stratify=y_train,random_state=5)
    train_dataset=test_dataset(train,y_train,device,RESOLUTION)
    train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,prefetch_factor=1)
    print(np.unique(np.array(y_train),return_counts=True))
    
    print("Creating dataloaders")


    # test_data=test_dataset(test,y_test,device,RESOLUTION)
    # test_dataloader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=16,prefetch_factor=8)
    
    checkpoint=torch.load(route_encoder,weights_only=False)
    model= siamese_model(checkpoint["model_type"],device,EMBEDDING_SIZE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    embs1,classes =  create_and_save_ALL_embeddings(model,train_dataloader,path_embeddings,True)