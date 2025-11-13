import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
device ='cuda' if torch.cuda.is_available() else 'cpu'
import gc         
import torch.amp as amp
import math
from transformers import AutoModelForImageClassification
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.data_acquisition import data_set_N_with_nature,BIG_DATALOADER, read_files,test_dataset
from utils.models import siamese_model
from utils.losses import ContrastiveLoss,SupConLoss
import time
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, distributed

from einops import rearrange
import os
import sys
import copy
def setup(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend="gloo", init_method="env://", world_size=world_size, rank=rank)

def cleanup():
    try:
        dist.barrier()
    except Exception:
        pass
    dist.destroy_process_group()

def all_gather_tensor_cpu(tensor: torch.Tensor):
    """
    Gather arbitrary-size first-dim tensor from all ranks, works on CPU.
    """
    # Ensure tensor is on CPU
    tensor = tensor.cpu()
    world_size = dist.get_world_size()

    # Get local batch size
    local_size = torch.tensor([tensor.size(0)], dtype=torch.long)
    sizes = [torch.tensor([0], dtype=torch.long) for _ in range(world_size)]

    # Gather sizes from all ranks
    dist.all_gather(sizes, local_size)
    sizes = [int(s.item()) for s in sizes]
    max_size = max(sizes)

    # Pad if necessary
    if local_size.item() < max_size:
        pad_shape = (max_size - local_size.item(),) + tensor.shape[1:]
        pad = torch.zeros(pad_shape, dtype=tensor.dtype)
        tensor_padded = torch.cat([tensor, pad], dim=0)
    else:
        tensor_padded = tensor

    # Gather all tensors
    gather_list = [torch.zeros_like(tensor_padded) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor_padded)

    # Remove padding and concatenate
    out_tensors = [g[:s] for g, s in zip(gather_list, sizes) if s > 0]
    return torch.cat(out_tensors, dim=0) if out_tensors else torch.empty((0,) + tensor.shape[1:], dtype=tensor.dtype)



if __name__ == "__main__":



    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # Constants:
    GLOBAL_BATCH=8192
    RESOLUTION=256
    MARGIN=1
    EMBEDDING_SIZE=128
    EFFICIENTNET_TYPE="efficientnet-b0"
    PATH_TO_SAVE=f'Models/Contrastive_Models/Contrastive_b0_MINIBATCH_{EMBEDDING_SIZE}_{GLOBAL_BATCH}_sd1_5_sd1_4_BigGan_Epoch.pth'
    PATH_TO_SAVE_Epoch=f"Models/Contrastive_Models/Contrastive_Mamba_MINIBATCH_128_8192_sd1_5_sd1_4_Wukong_MJ_glide_Partial_Checkpoint_2GPUS.pth"
    MAX_MINIBATCH_PROCESS=8  # Number of images to process in each minibatch (to avoid OOM errors)

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ.get("RANK", local_rank))
    print(f"Start rank {rank} local_rank {local_rank} world_size {world_size}")

    setup(rank, world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    per_gpu_batch = GLOBAL_BATCH // world_size
    

    retrain=False
    if len(sys.argv) > 1:
        retrain=True
        route=sys.argv[1]

    print("Getting data ...")
    loader_data = data_set_N_with_nature('GenImage_resized/')
    train,val,test,y_train,y_val,y_test = loader_data.get_data()
    print(np.unique(np.array(y_train),return_counts=True),np.unique(np.array(y_val),return_counts=True),np.unique(np.array(y_test),return_counts=True))


    model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-L3-256-21K",
                                                            trust_remote_code=True, dtype="auto")
    model.to(device)
    if os.path.exists(PATH_TO_SAVE_Epoch):
        ckpt = torch.load(PATH_TO_SAVE_Epoch, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        if rank == 0:
            print("✅ Loaded checkpoint from", PATH_TO_SAVE_Epoch)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    if rank == 0:
        print("✅ Model wrapped in DDP")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = SupConLoss().to(device)
    scaler = torch.amp.GradScaler()

    print("Creating Dataloaders ...")

    train_dataset = BIG_DATALOADER(train, y_train, device, RESOLUTION)
    train_sampler = distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=False,
    )

    # test_data=test_dataset(test,y_test,device,RESOLUTION)
    # test_dataloader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=12,prefetch_factor=8)

    del train,val,test,y_train,y_test,y_val,train_dataset#,test_data
    gc.collect()
    torch.cuda.empty_cache() 
    best=9999999.9
    print("Creating model...")
    
    criterion = SupConLoss()

    print("Training model...")
    EPOCHS=3
    train_loss=[]
    train_accuracy=[]
    best=999999
    val_loss=[]
    val_accuracy=[]

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        # Set model to training mode
        model.train()

        # Initialize validation stats
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for index, (routes, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}",total=len(train_loader))):

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
                    with torch.amp.autocast(device_type='cuda'):

                        pred1 = model(data)
                        pred1=pred1['logits']



                        pred1=pred1.to('cpu')
                        all_embeddings.extend(pred1)

                del data, pred1
                gc.collect()
            loss_tracker = []

            all_embeddings = torch.from_numpy(np.array(all_embeddings)).to('cpu')




            global_embeddings = all_gather_tensor_cpu(all_embeddings)

            global_labels = all_gather_tensor_cpu(labels.view(-1,1)).view(-1)



            for index1, (batch_routes, batch_labels) in enumerate(zip(chunks_routes,chunks_labels)):

                rep = copy.deepcopy(global_embeddings)

                data = read_files(batch_routes)
                data = data.to(local_rank)
                with torch.amp.autocast(device_type='cuda'):
                    preds=model(data)['logits'].to('cpu')
                rep[index1*MAX_MINIBATCH_PROCESS : index1*MAX_MINIBATCH_PROCESS + preds.size(0)] = preds
                del data
                gc.collect()

                global_labels=global_labels.to(local_rank)
                rep=rep.to(local_rank)
                with torch.amp.autocast(device_type='cuda'):
                    loss = criterion(rep.unsqueeze(1),global_labels)
                scaler.scale(loss).backward()

                del rep
                loss_tracker.append(loss)
                
    
            scaler.step(optimizer)
            scaler.update() # Performs an optimizer step
  # Updates the scaler for the next iteration
            loss = sum(loss_tracker)/len(loss_tracker)


            #Free memory
            del routes,labels,loss_tracker,global_embeddings,global_labels
            torch.cuda.empty_cache()

            gc.collect()


            

            # Track running loss and accuracy
            running_loss += loss.item()

            if loss.item()<best:
                best=loss.item()
                print(f'Loss Batch {index}: {running_loss / (index + 1):.3f}')
                checkpoint = {
                    "model_state_dict": model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best
                    }
                if not dist.is_initialized() or dist.get_rank() == 0:
                    torch.save(checkpoint, PATH_TO_SAVE_Epoch)


        # Calculate training loss
        train_loss_value = running_loss / len(train_loader)
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