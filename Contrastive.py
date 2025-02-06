import torch.multiprocessing as mp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
device ='cuda' if torch.cuda.is_available() else 'cpu'
import gc         
mp.set_start_method('spawn', force=True)

from data_acquisition import data_set,images_Dataset
from models import siamese_model
from losses import ContrastiveLoss

import sys

# import warnings
# warnings.filterwarnings('ignore')

if __name__ == "__main__":
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
    train_dataset=images_Dataset(train[:500000],y_train[:500000],device,256)
    train_dataloader=DataLoader(train_dataset,batch_size=39,shuffle=True,num_workers=4)

    val_dataset=images_Dataset(val[:1000],y_val[:1000],device,256)
    val_dataloader=DataLoader(val_dataset,batch_size=16,shuffle=True,num_workers=4)

    # test_dataset=images_Dataset(test[:5000],y_test[:5000],device,256)
    # test_dataloader=DataLoader(test_dataset,batch_size=64,shuffle=True)

    del train,val,test,y_train,y_test,y_val,train_dataset,val_dataset#,test_dataset
    gc.collect()
    torch.cuda.empty_cache() 
    best=9999999.9
    print("Creating model...")
    if retrain:
        checkpoint=torch.load(route)
        model=siamese_model(checkpoint["model_type"],device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best=checkpoint["best_loss"]
    else:
        model=siamese_model('efficientnet-b0',device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = ContrastiveLoss(margin=2)

    print("Training model...")
    EPOCHS=100
    train_loss=[]
    train_accuracy=[]
    best=0.047
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

            # Forward pass
            pred1,pred2 = model(image1, image2)

            #Free memory
            del image1,image2
            gc.collect()
            torch.cuda.empty_cache()

            # Calculate loss
            loss = criterion(pred1,pred2,label)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

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

                # Forward pass
                pred1,pred2 = model(image1, image2)

                #Free memory
                del image1,image2
                gc.collect()
                torch.cuda.empty_cache()

                # Calculate loss
                loss = criterion(pred1,pred2,label)

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

        #If this model is better than the best
        if val_loss_value <= best:
            best=val_loss_value
            checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "model_type": model.type,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss":best
                }
            torch.save(checkpoint, 'Models/Contrastive_Models/Best_Contrastive_model_b0_256_new_hoy.pth')
        checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "model_type": model.type,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss":best
                }
        torch.save(checkpoint, 'Models/Contrastive_Models/Last_Contrastive_model_b0_256_new_hoy.pth')

        print()
        print('-'*60)
        # Print results for the epoch
        print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss_value:.4f}")
        print(f"Val Loss: {val_loss_value:.4f}")
        print('-'*60)
        print()