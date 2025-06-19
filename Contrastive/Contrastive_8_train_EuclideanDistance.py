import torch.multiprocessing as mp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
device ='cuda' if torch.cuda.is_available() else 'cpu'
import gc         
mp.set_start_method('spawn', force=True)
import torch.amp as amp

from utils.data_acquisition import data_set,images_Dataset
from utils.models import siamese_model
from utils.losses import ContrastiveLoss
import time

import sys




if __name__ == "__main__":
    # Constants:
    BATCH_SIZE=39
    RESOLUTION=256
    MARGIN=1
    EMBEDDING_SIZE=128
    EFFICIENTNET_TYPE="efficientnet-b0"
    PATH_TO_SAVE=f'Models/Contrastive_Models/Contrastive_b0_{EMBEDDING_SIZE}_{BATCH_SIZE}_8_EuclideanDistance{MARGIN}.pth'
    
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    retrain=False
    if len(sys.argv) > 1:
        retrain=True
        route=sys.argv[1]

    print("Getting data ...")
    loader_data = data_set('Datasets/GenImage/')
    train,val,test,y_train,y_val,y_test = loader_data.get_data()

    print('Preparing data ...')
    train,y_train=loader_data.make_pairs(train, y_train)
    val,y_val=loader_data.make_pairs(val, y_val)
    test,y_test=loader_data.make_pairs(test, y_test)

    print("Creating Dataloaders ...")
    train_dataset=images_Dataset(train,y_train,device,RESOLUTION)
    train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,prefetch_factor=2)

    val_dataset=images_Dataset(val,y_val,device,RESOLUTION)
    val_dataloader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,prefetch_factor=2)

    test_dataset=images_Dataset(test,y_test,device,RESOLUTION)
    test_dataloader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,prefetch_factor=2)

    del train,val,test,y_train,y_test,y_val,train_dataset,val_dataset,test_dataset
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
    criterion = ContrastiveLoss(margin=MARGIN)
    scaler = amp.GradScaler()

    print("Training model...")
    EPOCHS=1
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
        for image1, image2, label in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}"):

            #Reset optimizer
            optimizer.zero_grad()

            #Data to GPU
            image1 = image1.to(device)
            image2 = image2.to(device)
            label = label.to(device)

            with amp.autocast(device_type='cuda'):
                # Forward pass
                pred1,pred2 = model(image1, image2)
                # Calculate loss
                loss = criterion(pred1,pred2,label)

            #Free memory
            del image1,image2
            gc.collect()
            torch.cuda.empty_cache()

            

            # Backward pass and optimization
            scaler.scale(loss).backward()  # Scaled loss for gradient computation

            # Unscales gradients and updates the optimizer step
            scaler.step(optimizer)  # Performs an optimizer step
            scaler.update()  # Updates the scaler for the next iteration

            # Track running loss and accuracy
            running_loss += loss.item()

            #Free memory
            del label, loss,pred1,pred2
            gc.collect()
            torch.cuda.empty_cache()

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
            for image1, image2, label in tqdm(val_dataloader, desc=f"Validating Epoch {epoch + 1}/{EPOCHS}"):
                
                #Data to GPU
                image1 = image1.to(device)
                image2 = image2.to(device)
                label = label.to(device)

                with amp.autocast(device_type=device,):  # Automatically choose precision (float16 for ops that benefit)
                # Forward pass
                    pred1,pred2 = model(image1, image2)
                    loss = criterion(pred1,pred2,label)

                #Free memory
                del image1,image2
                gc.collect()
                torch.cuda.empty_cache()

                # Calculate loss
                scaler.scale(loss)

                # Track running loss and accuracy
                running_loss += loss.item()

                val_total += label.size(0)

                #Free memory
                del pred1,pred2,label,loss
                gc.collect()
                torch.cuda.empty_cache()

        # Calculate validation loss and accuracy
        val_loss_value = running_loss / len(val_dataloader)
        val_loss.append(val_loss_value)
        # if val_loss_value<=best:
        #     best=val_loss_value
        checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_type": model.type,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss":best
            }
        torch.save(checkpoint, PATH_TO_SAVE)
        
        print()
        print('-'*60)
        # Print results for the epoch
        print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss_value:.4f}")
        print(f"Val Loss: {val_loss_value:.4f}")
        print('-'*60)
        print()