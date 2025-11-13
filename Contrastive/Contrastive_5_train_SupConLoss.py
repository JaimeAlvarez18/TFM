# ddp_train.py
import os
import sys
import math
import gc
import copy
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, distributed
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForImageClassification
from utils.data_acquisition import data_set_N_with_nature, BIG_DATALOADER, read_files, test_dataset
from utils.losses import SupConLoss

def setup(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)

def cleanup():
    try:
        dist.barrier()
    except Exception:
        pass
    dist.destroy_process_group()

def all_gather_tensor(tensor: torch.Tensor, device: torch.device):
    """Gather arbitrary-size first-dim tensor from all ranks."""
    world_size = dist.get_world_size()
    local_size = torch.tensor([tensor.size(0)], device=device, dtype=torch.long)
    sizes = [torch.tensor([0], device=device, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    sizes = [int(s.item()) for s in sizes]
    max_size = max(sizes)

    if local_size.item() < max_size:
        pad_shape = (max_size - local_size.item(),) + tensor.shape[1:]
        pad = torch.zeros(pad_shape, device=device, dtype=tensor.dtype)
        tensor_padded = torch.cat([tensor, pad], dim=0)
    else:
        tensor_padded = tensor

    gather_list = [torch.zeros_like(tensor_padded) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor_padded)

    out_tensors = [g[:s] for g, s in zip(gather_list, sizes) if s > 0]
    return torch.cat(out_tensors, dim=0) if out_tensors else torch.empty((0,) + tensor.shape[1:], device=device, dtype=tensor.dtype)

def main():
    CHECKPOINT_PATH = "Models/Contrastive_Models/Contrastive_Mamba_MINIBATCH_128_8192_sd1_5_sd1_4_Wukong_MJ_glide_Partial_Checkpoint.pth"
    GLOBAL_BATCH = 8192
    RESOLUTION = 256
    EMBEDDING_SIZE = 128
    MAX_MINIBATCH_PROCESS = 4
    USE_AMP = True

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ.get("RANK", local_rank))
    print(f"Start rank {rank} local_rank {local_rank} world_size {world_size}")

    setup(rank, world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    per_gpu_batch = GLOBAL_BATCH // world_size
    if per_gpu_batch < 1:
        raise ValueError("GLOBAL_BATCH smaller than world_size.")
    if rank == 0:
        print(f"Using global batch {GLOBAL_BATCH} → per-GPU batch {per_gpu_batch}")

    loader_data = data_set_N_with_nature('GenImage_resized/')
    train, val, test, y_train, y_val, y_test = loader_data.get_data()

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

    model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-L3-256-21K",
                                                            trust_remote_code=True, dtype="auto")
    model.to(device)
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        if rank == 0:
            print("✅ Loaded checkpoint from", CHECKPOINT_PATH)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    if rank == 0:
        print("✅ Model wrapped in DDP")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = SupConLoss().to(device)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    EPOCHS = 3
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()

        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}") if rank == 0 else enumerate(train_loader)

        for batch_idx, (routes, labels) in pbar:
            routes = list(routes)
            local_batch_size = len(routes)
            if local_batch_size == 0:
                continue

            chunk_size = math.ceil(local_batch_size / MAX_MINIBATCH_PROCESS)
            chunked_routes = [routes[i*chunk_size:(i+1)*chunk_size] for i in range(math.ceil(local_batch_size / chunk_size))]

            optimizer.zero_grad(set_to_none=True)
            embeddings_chunks = []
            for chunk in chunked_routes:
                imgs = read_files(np.array(chunk))
                if isinstance(imgs, np.ndarray):
                    imgs = torch.from_numpy(imgs)
                imgs = imgs.to(device, non_blocking=True)

                with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
                    out = model(imgs)['logits']
                out = out.unsqueeze(1)
                embeddings_chunks.append(out)
                del imgs, out
                gc.collect()

            local_embeddings = torch.cat(embeddings_chunks, dim=0)
            labels = torch.as_tensor(labels, device=device, dtype=torch.long)

            emb_all = all_gather_tensor(local_embeddings, device)
            labels_all = all_gather_tensor(labels.view(-1, 1), device).view(-1)

            with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
                loss = criterion(emb_all, labels_all)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if rank == 0 and (batch_idx % 10 == 0):
                pbar.set_postfix({'batch_loss': loss.item(), 'avg_loss': running_loss / (batch_idx + 1)})

            del local_embeddings, emb_all, labels_all, labels, loss, embeddings_chunks
            torch.cuda.empty_cache()
            gc.collect()

        if rank == 0:
            avg_loss = running_loss / max(len(train_loader), 1)
            print(f"Epoch {epoch+1}/{EPOCHS} avg_loss: {avg_loss:.6f}")
            ckpt = {
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(ckpt, f"checkpoint_epoch_{epoch+1}.pth")

    cleanup()

if __name__ == "__main__":
    main()
