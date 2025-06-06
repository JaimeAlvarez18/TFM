
import sys
import torch.multiprocessing as mp
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
device ='cuda' if torch.cuda.is_available() else 'cpu'
import gc         
mp.set_start_method('spawn', force=True)
from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.metrics import confusion_matrix
from utils.data_acquisition import data_set, images_Dataset,mix_dataset
from utils.models import Big_model
import numpy as np
import time
import torch.amp as amp


def print_confusion_matrix(cm):
    # Print confusion matrix in a nice format
    # Print header
    print(" " * 8, "Predicted")
    print(" " * 8, " ".join([f"{i:4}" for i in range(cm.shape[1])]))
    
    # Print each row
    for i, row in enumerate(cm):
        print(f"True {i:2} ", " ".join([f"{val:4}" for val in row]))

if __name__ == "__main__":
    retrain=False
    if len(sys.argv) == 2:
        retrain=False
        route_encoder=sys.argv[1]

    if len(sys.argv) == 3:
        retrain=True
        route_encoder=sys.argv[1]
        route_classifier=sys.argv[2]
        
    BATCH_SIZE=182
    RESOLUTION=256
    MARGIN=1
    EMBEDDING_SIZE=128
    EFFICIENTNET_TYPE="efficientnet-b0"
    LOSS='EuclideanDistance1'
    NUMBER=5
    PATH_TO_SAVE=f'Models/Classification_Models/Classification_pair_{EMBEDDING_SIZE}_{BATCH_SIZE}_{NUMBER}_{LOSS}.pth'    


    print("Getting data ...")
    loader_data = data_set('Datasets/GenImage/')
    train,val,test,y_train,y_val,y_test = loader_data.get_data()
    

    print('Preparing data ...')
    
    train1,y_train=loader_data.make_pairs(train, y_train)
    val1,y_val=loader_data.make_pairs(val,y_val)
    test1,y_test=loader_data.make_pairs(test, y_test)

    print("Creating dataloaders ...")
    
    train_dataset=images_Dataset(train1,y_train,device,RESOLUTION)
    train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=12,prefetch_factor=8)

    val_dataset=images_Dataset(val1,y_val,device,RESOLUTION)
    val_dataloader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=12,prefetch_factor=8)

    test_dataset=images_Dataset(test1,y_test,device,RESOLUTION)
    test_dataloader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=12,prefetch_factor=8)

    del train,val,test,y_train,y_test,y_val,train_dataset,val_dataset,test_dataset
    gc.collect()
    torch.cuda.empty_cache() 

    print("Creating model...")
    if retrain:
        checkpoint=torch.load(route_classifier)
        model= Big_model(route_encoder,device,EMBEDDING_SIZE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        model = Big_model(route_encoder,device,EMBEDDING_SIZE).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("Training model...")
    EPOCHS=3
    train_loss=[]
    train_accuracy=[]
    best=0.0
    val_loss=[]
    val_accuracy=[]
    scaler = amp.GradScaler()

    for epoch in range(EPOCHS):
        # Set model to training mode
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []  # List to store true labels
        all_preds = []   # List to store predictions

        # Training loop
        for image1, image2, label in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}"):
            
            #Reset optimizer
            optimizer.zero_grad()

            #Data to GPU
            image1 = image1.to(device)
            image2 = image2.to(device)
            label = label.long().to(device)
            
                        # Forward pass
            with amp.autocast(device_type='cuda'):

            # Forward pass
                pred = model(image1, image2)
                # Calculate loss
                loss = criterion(pred,label)


            #Free memory
            del image1,image2
            gc.collect()
            torch.cuda.empty_cache()


            scaler.scale(loss).backward()  # Scaled loss for gradient computation

            # Unscales gradients and updates the optimizer step
            scaler.step(optimizer)  # Performs an optimizer step
            scaler.update()  # Updates the scaler for the next iteration

            # Track running loss and accuracy
            running_loss += loss.item()
            total += label.size(0)
            correct += (torch.argmax(pred,1) == label).sum().item()

            all_labels.extend(label.cpu().numpy())  # Move to CPU for confusion matrix
            all_preds.extend(torch.argmax(pred, 1).cpu().numpy())  # Predictions
        

            #Free memory
            del label, loss,pred
            gc.collect()
            torch.cuda.empty_cache()


        train_loss_value = running_loss / len(train_dataloader)
        train_accuracy_value = 100 * correct / total
        train_loss.append(train_loss_value)
        train_accuracy.append(train_accuracy_value)
        tcm = confusion_matrix(all_labels, all_preds)

        # Set model to evaluation mode
        model.eval()

        # Initialize validation stats
        running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_labels = []  # List to store true labels
        all_preds = []   # List to store predictions

        # Validation loop
        with torch.no_grad():
            for image1, image2, label in tqdm(test_dataloader, desc=f"Validating Epoch {epoch + 1}/{EPOCHS}"):
                
                #Data to GPU
                image1 = image1.to(device)
                image2 = image2.to(device)
                label = label.long().to(device)

                with amp.autocast(device_type='cuda'):
                    # Forward pass
                    pred = model(image1, image2)
                    # Calculate loss
                    loss = criterion(pred,label)

                #Free memory
                del image1,image2
                gc.collect()
                torch.cuda.empty_cache()



                # Track running loss and accuracy
                running_loss += loss.item()
                val_total += label.size(0)
                val_correct += (torch.argmax(pred,1) == label).sum().item()
                all_labels.extend(label.cpu().numpy())  # Move to CPU for confusion matrix
                all_preds.extend(torch.argmax(pred, 1).cpu().numpy())  # Predictions

            
                

                #Free memory
                del pred,label,loss
                gc.collect()
                torch.cuda.empty_cache()


        # Calculate validation loss and accuracy
        val_loss_value = running_loss / len(val_dataloader)
        val_accuracy_value = 100 * val_correct / val_total
        val_loss.append(val_loss_value)
        val_accuracy.append(val_accuracy_value)
        vcm = confusion_matrix(all_labels, all_preds)
        if val_accuracy_value >= best:
            best=val_accuracy_value
            checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "encoder_path": model.path_to_encoder,
                    "encoder_type" : model.encoder_type,
                    "best_result":best

                }
            torch.save(checkpoint, PATH_TO_SAVE)
        

        print()
        print('-'*60)
        print('-'*60)
        # Print results for the epoch
        print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        print()
        print('*'*60)
        print("Train")
        print(f"Loss: {train_loss_value:.4f}. Accuracy: {train_accuracy_value}.")
        print(f"Training Confusion Matrix:")
        print_confusion_matrix(tcm)
        print()
        print('*'*60)
        print()
        print("Validation")
        print(f"Loss: {val_loss_value:.4f}. Accuracy: {val_accuracy_value}.")
        print(f"Validation Confusion Matrix:")
        print_confusion_matrix(vcm)
        print('-'*60)
        print('-'*60)
        print()