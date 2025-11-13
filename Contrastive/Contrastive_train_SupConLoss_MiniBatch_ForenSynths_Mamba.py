# docker exec -it 3ead6b1f94d9 bash
# python -m Contrastive.Contrastive_train_SupConLoss_MiniBatch_ForenSynths_Mamba > New_ES5.log 2>&1 &

import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
import gc
from tqdm import tqdm
device ='cuda:0' #if torch.cuda.is_available() else 'cpu'

# print(torch.cuda.mem_get_info(device=device))
torch.cuda.set_device(0)  # Select GPU 0
torch.cuda.empty_cache()   # Clear cache
torch.cuda.reset_peak_memory_stats(0)  # Reset stats for GPU 0
torch.cuda.reset_accumulated_memory_stats(0)
with torch.cuda.device(0):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


import gc         
import torch.amp as amp
from transformers import AutoModelForImageClassification
import math

from utils.data_acquisition import data_set_N_with_nature,BIG_DATALOADER, read_files,test_dataset,data_set_binary_synth
from utils.models import siamese_model
from utils.losses import ContrastiveLoss,SupConLoss
import time
import numpy as np

from einops import rearrange

import sys
import copy

if __name__ == "__main__":
    # Constants:
    BATCH_SIZE=6000
    # BATCH_SIZE=4080
    RESOLUTION=256
    MARGIN=1
    EMBEDDING_SIZE=128
    EFFICIENTNET_TYPE="efficientnet-b0"
    # PATH_TO_SAVE=f'Models/Contrastive_Models/Contrastive_b0_MINIBATCH_{EMBEDDING_SIZE}_{BATCH_SIZE}_ForenSynths_Epoch.pth'
    # PATH_TO_SAVE_Epoch=f'Models/Contrastive_Models/Contrastive_Mamba_MINIBATCH_{BATCH_SIZE}_9_train_Checkpoint_1_GPU_ES4.pth'
    # PATH_TO_SAVE_Epoch=f'Models/Contrastive_Mamba_MINIBATCH_6000_adm_bigan_MJ_vqdm_sd_1_5_Partial_Checkpoint_1_GPU_ES2.pth'
    PATH_TO_SAVE_Epoch=f'Models/Contrastive_NEW_ES5.pth'
    # MAX_MINIBATCH_PROCESS=100
    MAX_MINIBATCH_PROCESS=500
    
    # SMALL_MINIBATCH=10  # Number of images to process in each minibatch (to avoid OOM errors)
    SMALL_MINIBATCH=20 # Number of images to process in each minibatch (to avoid OOM errors)
    

    retrain=False
    if len(sys.argv) > 1:
        retrain=True
        route=sys.argv[1]

    print("Getting data ...")
    # loader_data = data_set_binary_synth('ForenSynths')
    # train,y_train,_,_,test,y_test = loader_data.get_data()
# 
    loader_data = data_set_N_with_nature('GenImage_resized/')
    train, val, test, y_train, y_val, y_test = loader_data.get_data()
 


    

    print("Creating Dataloaders ...")
    train_dataset=BIG_DATALOADER(train,y_train,device,RESOLUTION)
    train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,prefetch_factor=1)

    # val_dataset=test_dataset(val,y_val,device,RESOLUTION)
    # val_dataloader=DataLoader(val_dataset,batch_size=16,shuffle=True,num_workers=2,prefetch_factor=1)

    # test_data=test_dataset(test,y_test,device,RESOLUTION)
    # test_dataloader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=12,prefetch_factor=8)

    del train,test,y_train,y_test,train_dataset#,test_data
    gc.collect()
    torch.cuda.empty_cache() 
    best=9999999.9
    print(f"Creating model... {retrain=}")
    batch_losses=[]
    if retrain:
        checkpoint=torch.load(route)
        best=checkpoint["best_loss"]
        batch_losses=checkpoint['batch_losses']
        model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-L3-256-21K", trust_remote_code=True, dtype="auto").to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model loaded for retraining...")
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # batch_losses1 = [t.item() for t in batch_losses]
        # del batch_losses
        # gc.collect()
        # torch.cuda.empty_cache()
        # batch_losses=batch_losses1

        print(batch_losses,best)
        print("Retraining on")
    else:
        model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-L3-256-21K", trust_remote_code=True, dtype="auto").to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = SupConLoss()
    scaler = amp.GradScaler()

    print("Training model...")
    EPOCHS=1
    train_loss=[]
    train_accuracy=[]
    val_loss=[]
    val_accuracy=[]
    n_batch=0

    
    for epoch in range(EPOCHS):
        # Set model to training mode
        model.train()

        # Initialize validation stats
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for index, (routes, labels) in tqdm(enumerate(train_dataloader), desc=f"Training Epoch {epoch + 1}/{EPOCHS}"):
            if n_batch>50:
                break
            print("-"*50)
            print(f"Starting new batch: {index}")
            previous=model.state_dict()

            routes=np.array(list(routes))
            #Reset optimizer
            optimizer.zero_grad()

            chunk_size = math.ceil(routes.shape[0] / MAX_MINIBATCH_PROCESS)
            chunks_routes= np.array_split(routes, chunk_size)
            chunks_labels= np.array_split(labels, chunk_size)

            all_embeddings=[]
            for batch_routes in chunks_routes:
                data = read_files(batch_routes)
                data = data.to(device)
                with torch.no_grad():
                    with amp.autocast(device_type='cuda'):
                        pred1 = model(data)['logits']
                        pred1=pred1.unsqueeze(1)
                    pred1=pred1.to('cpu')
                    all_embeddings.extend(pred1)
                del data, pred1
                gc.collect()
                torch.cuda.empty_cache()
            loss_tracker = []
            all_embeddings = torch.from_numpy(np.array(all_embeddings)).unsqueeze(0).to('cpu')
            all_embeddings = rearrange(all_embeddings, 'b n k c -> (b n) k c')

            chunk_size_small = math.ceil(routes.shape[0] / SMALL_MINIBATCH)
            chunks_routes_small= np.array_split(routes, chunk_size_small)
            chunks_labels_small= np.array_split(labels, chunk_size_small)


            for index1, (batch_routes, batch_labels) in enumerate(zip(chunks_routes_small,chunks_labels_small)):
                rep = all_embeddings.clone()

                data = read_files(batch_routes)
                data = data.to(device)
                with amp.autocast(device_type='cuda'):
                    preds=model(data)['logits'].unsqueeze(1).to('cpu')
                    start = index1 * SMALL_MINIBATCH
                    rep[start : start + preds.shape[0]] = preds
                    
                    del data
                    gc.collect()
                    torch.cuda.empty_cache()
                    labels=labels.to(device)
                    rep=rep.to(device)

                    loss = criterion(rep,labels)
                del rep,preds
                gc.collect()
                torch.cuda.empty_cache()
                
                loss_tracker.append(loss.item())

                scaler.scale(loss).backward()
                del loss
                gc.collect()
                torch.cuda.empty_cache()
            scaler.step(optimizer)  # Performs an optimizer step
            scaler.update()  # Updates the scaler for the next iteration
            loss = sum(loss_tracker)/len(loss_tracker)


            #Free memory
            del routes,labels,loss_tracker
            gc.collect()
            torch.cuda.empty_cache()

            batch_losses.append(loss)

            

            # Track running loss and accuracy
            # if loss.item()<best:
        
            
            if loss<best:
                print(f'Loss Batch {index}: {loss:.3f}. Saved')
                best=loss
            
                checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss":best,
                "batch_losses": batch_losses
                }
            else:
                print(f'Loss Batch {index}: {loss:.3f}. Not Saved')
                checkpoint = {
                "model_state_dict": previous,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss":best,
                "batch_losses": batch_losses
                }

            torch.save(checkpoint, PATH_TO_SAVE_Epoch)
            del loss
            gc.collect()
            torch.cuda.empty_cache()
            n_batch+=1
    

        # # Calculate training loss
        # train_loss_value = running_loss / len(train_dataloader)
        # train_loss.append(train_loss_value)
        # # Set model to evaluation mode

        

        # print('-'*60)
        # # Print results for the epoch
        # print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        # print(f"Train Loss: {train_loss_value:.4f}")
        # print('-'*60)
        # print()