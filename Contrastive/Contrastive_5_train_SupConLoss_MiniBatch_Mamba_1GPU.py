import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
target_name = "NVIDIA A40"  # or any substring of the GPU name

device = None
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    print(i,name)
    # if target_name.lower() in name.lower():
    #     device = torch.device(f"cuda:{i}")
    #     print(f"Selected {name} as {device}")
    #     break


import gc         
import torch.amp as amp
import math
from transformers import AutoModelForImageClassification

from utils.data_acquisition import data_set_N_with_nature,BIG_DATALOADER, read_files,test_dataset,data_set_binary_synth
from utils.models import siamese_model
from utils.losses import ContrastiveLoss,SupConLoss
import time
import numpy as np

from einops import rearrange

import sys
import copy

import transformers


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP



if __name__ == "__main__":
    dist.init_process_group("nccl")
    numbers_gpus = "0,1,2" 
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # Constants:
    BATCH_SIZE=8192
    RESOLUTION=256
    MARGIN=1
    EMBEDDING_SIZE=128
    EFFICIENTNET_TYPE="efficientnet-b0"
    PATH_TO_SAVE=f'Models/Contrastive_Models/Contrastive_Mamba_MINIBATCH_{EMBEDDING_SIZE}_{BATCH_SIZE}_ForenSynths_Epoch.pth'
    PATH_TO_SAVE_Epoch=f'Contrastive_Mamba_MINIBATCH_128_8192_sd1_5_sd1_4_Wukong_MJ_glide_Partial_Checkpoint'
    MAX_MINIBATCH_PROCESS=16  # Number of images to process in each minibatch (to avoid OOM errors)
    

    retrain=False
    if len(sys.argv) > 1:
        retrain=True
        route=sys.argv[1]

    # Load model directly


    print("Getting data ...")
    loader_data = data_set_N_with_nature('GenImage_resized/')
    train,val,test,y_train,y_val,y_test = loader_data.get_data()
 

    print("Creating Dataloaders ...")
    train_dataset=BIG_DATALOADER(train,y_train,device,RESOLUTION)

    train_sampler=torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=4,prefetch_factor=2)


    # val_dataset=test_dataset(val,y_val,device,RESOLUTION)
    # val_dataloader=DataLoader(val_dataset,batch_size=16,shuffle=True,num_workers=2,prefetch_factor=1)

    # test_data=test_dataset(test,y_test,device,RESOLUTION)
    # test_dataloader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=12,prefetch_factor=8)

    del train,test,y_train,y_test,train_dataset#,test_data
    gc.collect()
    torch.cuda.empty_cache() 
    best=9999999.9
    print("Creating model...")

    model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-L3-256-21K", trust_remote_code=True, dtype="auto").to(local_rank)
    # if os.path.exists(PATH_TO_SAVE_Epoch):
    #     ckpt = torch.load(PATH_TO_SAVE_Epoch, map_location="cpu")
    #     model.load_state_dict(ckpt["model_state_dict"])
    #     print("✅ Loaded checkpoint from", PATH_TO_SAVE_Epoch)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = SupConLoss()
    scaler = amp.GradScaler()


    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    size_mb = (param_size + buffer_size) / (1024 ** 2)
    print(f"Model size (in memory): {size_mb:.2f} MB")

    print("Training model...")
    EPOCHS=3
    train_loss=[]
    train_accuracy=[]
    best=999999
    val_loss=[]
    val_accuracy=[]
    with torch.amp.autocast(device_type='cuda'):
        for epoch in range(EPOCHS):
            train_sampler.set_epoch(epoch)
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
                    data = data.to(local_rank)
                    with torch.no_grad():
                        
                        pred1 = model(data)
                        pred1=pred1['logits']

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
                    data = data.to(local_rank)
                    preds=model(data)['logits'].unsqueeze(1).to('cpu')
                    rep[index1*MAX_MINIBATCH_PROCESS:(index1+1)*MAX_MINIBATCH_PROCESS] = preds
                    del data
                    gc.collect()
                    torch.cuda.empty_cache()

                    labels=labels.to(local_rank)
                    rep=rep.to(local_rank)
                    loss = criterion(rep,labels)
                    labels=labels.to('cpu')
                    del rep
                    gc.collect()
                    torch.cuda.empty_cache()
                    loss_tracker.append(loss)

                       # Backward pass and optimization
                    scaler.scale(loss).backward()  # Scaled loss for gradient computation

            # Unscales gradients and updates the optimizer step
                scaler.step(optimizer)  # Performs an optimizer step
                scaler.update()  # Updates the scaler for the next iteration
    # Updates the scaler for the next iteration
                loss = sum(loss_tracker)/len(loss_tracker)


                #Free memory
                del routes,labels,loss_tracker
                gc.collect()
                torch.cuda.empty_cache()


                

                # Track running loss and accuracy
                running_loss += loss.item()

                if loss.item()<best:
                    best=loss.item()
                    print(f'Loss Batch {index}: {running_loss / (index + 1):.3f}')
                    checkpoint = {
                    "model_state_dict": model.module.state_dict(),

                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss":best
                    }
                    torch.save(checkpoint, PATH_TO_SAVE_Epoch)

            # Calculate training loss
            train_loss_value = running_loss / len(train_dataloader)
            train_loss.append(train_loss_value)
            # Set model to evaluation mode


            # Initialize validation stats
            running_loss = 0.0
            val_correct = 0
            val_total = 0

            

            print('-'*60)
            # Print results for the epoch
            print(f"Epoch [{epoch + 1}/{EPOCHS}]")
            print(f"Train Loss: {train_loss_value:.4f}")
            print('-'*60)
            print()
        dist.destroy_process_group()