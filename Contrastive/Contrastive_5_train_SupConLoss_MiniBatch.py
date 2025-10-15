
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
device ='cuda' if torch.cuda.is_available() else 'cpu'
import gc         
import torch.amp as amp
import math

from utils.data_acquisition import data_set_N_with_nature,BIG_DATALOADER, read_files,test_dataset
from utils.models import siamese_model
from utils.losses import ContrastiveLoss,SupConLoss
import time
import numpy as np

from einops import rearrange

import sys
import copy

if __name__ == "__main__":
    # Constants:
    BATCH_SIZE=8192
    RESOLUTION=256
    MARGIN=1
    EMBEDDING_SIZE=128
    EFFICIENTNET_TYPE="efficientnet-b0"
    PATH_TO_SAVE=f'Models/Contrastive_Models/Contrastive_b0_MINIBATCH_{EMBEDDING_SIZE}_{BATCH_SIZE}_sd1_5_sd1_4_BigGan_Epoch.pth'
    PATH_TO_SAVE_Epoch=f'Models/Contrastive_Models/Contrastive_b0_MINIBATCH_{EMBEDDING_SIZE}_{BATCH_SIZE}_sd1_5_sd1_4_BigGan_Partial_Checkpoint.pth'
    MAX_MINIBATCH_PROCESS=32  # Number of images to process in each minibatch (to avoid OOM errors)
    

    retrain=False
    if len(sys.argv) > 1:
        retrain=True
        route=sys.argv[1]

    print("Getting data ...")
    loader_data = data_set_N_with_nature('GenImage_resized/')
    train,val,test,y_train,y_val,y_test = loader_data.get_data()
    print(len(train),len(val),len(test),len(y_train),len(y_test),len(y_val))


    

    print("Creating Dataloaders ...")
    train_dataset=BIG_DATALOADER(train,y_train,device,RESOLUTION)
    train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,prefetch_factor=1)

    val_dataset=test_dataset(val,y_val,device,RESOLUTION)
    val_dataloader=DataLoader(val_dataset,batch_size=16,shuffle=True,num_workers=2,prefetch_factor=1)

    # test_data=test_dataset(test,y_test,device,RESOLUTION)
    # test_dataloader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=12,prefetch_factor=8)

    del train,val,test,y_train,y_test,y_val,train_dataset,val_dataset#,test_data
    gc.collect()
    torch.cuda.empty_cache() 
    best=9999999.9
    print("Creating model...")
    if retrain:
        checkpoint=torch.load(route)
        model=siamese_model(checkpoint["model_type"],device,EMBEDDING_SIZE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best=checkpoint["best_loss"]
    else:
        model=siamese_model(EFFICIENTNET_TYPE,device,EMBEDDING_SIZE).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = SupConLoss()
    scaler = amp.GradScaler()

    print("Training model...")
    EPOCHS=3
    train_loss=[]
    train_accuracy=[]
    best=999999
    val_loss=[]
    val_accuracy=[]

    for epoch in range(EPOCHS):
        # Set model to training mode
        model.train()

        # Initialize validation stats
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for index, (routes, labels) in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}")):

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
                        pred1 = model.predict_one_image(data)
                        pred1=pred1.unsqueeze(1)
                    pred1=pred1.to('cpu')
                    all_embeddings.append(pred1)
                del data, pred1
                gc.collect()
                torch.cuda.empty_cache()
            loss_tracker = []
            all_embeddings = torch.from_numpy(np.array(all_embeddings)).to('cpu')
            all_embeddings = rearrange(all_embeddings, 'b n k c -> (b n) k c')


            for index1, (batch_routes, batch_labels) in enumerate(zip(chunks_routes,chunks_labels)):
                rep = copy.deepcopy(all_embeddings)

                data = read_files(batch_routes)
                data = data.to(device)
                with amp.autocast(device_type='cuda'):
                    preds=model.predict_one_image(data).unsqueeze(1).to('cpu')
                    rep[index1*MAX_MINIBATCH_PROCESS:(index1+1)*MAX_MINIBATCH_PROCESS] = preds
                    del data
                    gc.collect()
                    torch.cuda.empty_cache()
                    model=model.to('cpu')
                    labels=labels.to(device)
                    rep=rep.to(device)

                    loss = criterion(rep,labels)
                labels=labels.to('cpu')
                model=model.to(device)
                del rep
                gc.collect()
                torch.cuda.empty_cache()
                loss_tracker.append(loss)

                scaler.scale(loss).backward()
            scaler.step(optimizer)  # Performs an optimizer step
            scaler.update()  # Updates the scaler for the next iteration
            loss = sum(loss_tracker)/len(loss_tracker)


            #Free memory
            del routes,labels,loss_tracker
            gc.collect()
            torch.cuda.empty_cache()


            

            # Track running loss and accuracy
            running_loss += loss.item()
            if index % 2 == 0:    # Print every 10 mini-batches
                print(f'Loss Batch {index}: {running_loss / (index + 1):.3f}')
                checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_type": model.type,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss":best
                }
                torch.save(checkpoint, PATH_TO_SAVE_Epoch)

        # Calculate training loss
        train_loss_value = running_loss / len(train_dataloader)
        train_loss.append(train_loss_value)
        # Set model to evaluation mode
        model.eval()

        # Initialize validation stats
        running_loss = 0.0
        val_correct = 0
        val_total = 0

        # Validation loop
        with torch.no_grad():
            for image1, label in tqdm(val_dataloader, desc=f"Validating Epoch {epoch + 1}/{EPOCHS}"):
                
                #Data to GPU
                image1 = image1.to(device)
                label = label.to(device)

                with amp.autocast(device_type=device):  # Automatically choose precision (float16 for ops that benefit)
                # Forward pass
                    pred1 = model.predict_one_image(image1)
                    pred1=pred1.unsqueeze(1)
                    loss = criterion(pred1,label)

                #Free memory
                del image1
                gc.collect()
                torch.cuda.empty_cache()
                scaler.scale(loss)

                # Calculate loss
                

                # Track running loss and accuracy

                if np.isnan(loss.item()) == False:
                    running_loss += loss.item()
                
                    

                val_total += label.size(0)

                #Free memory
                del pred1,label,loss
                gc.collect()
                torch.cuda.empty_cache()

        # Calculate validation loss and accuracy
        val_loss_value = running_loss / len(val_dataloader)
        val_loss.append(val_loss_value)
        if val_loss_value<=best:
            best=val_loss_value
        checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_type": model.type,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss":best
            }
        torch.save(checkpoint, PATH_TO_SAVE)

        print('-'*60)
        # Print results for the epoch
        print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss_value:.4f}")
        print(f"Val Loss: {val_loss_value:.4f}")
        print('-'*60)
        print()