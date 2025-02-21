import torch.multiprocessing as mp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
device ='cuda' if torch.cuda.is_available() else 'cpu'
import gc         
mp.set_start_method('spawn', force=True)

from utils.data_acquisition import data_set,test_dataset
from utils.models import siamese_model
from utils.losses import ContrastiveLoss
from efficientnet_pytorch import EfficientNet
from torch import nn
import numpy as np
from sklearn.metrics import confusion_matrix

import sys

# import warnings
# warnings.filterwarnings('ignore')

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
    if len(sys.argv) > 1:
        retrain=True
        route=sys.argv[1]
        
    BATCH_SIZE=182
    RESOLUTION=256
    MARGIN=1
    EMBEDDING_SIZE=128
    EFFICIENTNET_TYPE="efficientnet-b0"
    PATH_TO_SAVE=f'Models/Supervised/Supervised_b0_{EMBEDDING_SIZE}_{BATCH_SIZE}.pth'
    
    print("Getting data ...")
    loader_data = data_set('Datasets/GenImage/')
    train,val,test,y_train,y_val,y_test = loader_data.get_data()


    print('Preparing data ...')


    print("Creating Dataloaders ...")
    train_dataset=test_dataset(train,y_train,device,RESOLUTION)
    train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=12,prefetch_factor=8)

    val_dataset=test_dataset(test,y_test,device,RESOLUTION)
    val_dataloader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=12,prefetch_factor=8)




    best=0
    print("Creating model...")
    if retrain:
        checkpoint=torch.load(route,weights_only=False)
        model = EfficientNet.from_name('efficientnet-b0')  # Initialize from scratch
        # Modify the classifier (output layer) of EfficientNet

        model._fc = nn.Linear(1280,len(np.unique(y_train)))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best=checkpoint["best_loss"]
    else:
        model = EfficientNet.from_name('efficientnet-b0',3)  # Initialize from scratch
        model._fc = nn.Linear(1280,len(np.unique(y_train)))
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("Training model...")
    EPOCHS=100
    train_loss=[]
    train_accuracy=[]
    best=0
    val_loss=[]
    val_accuracy=[]

    for epoch in range(EPOCHS):
        # Set model to training mode
        model.train()

        # Initialize validation stats
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []  # List to store true labels
        all_preds = []   # List to store predictions

        # Training loop
        for image1, label in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}"):

            #Reset optimizer
            optimizer.zero_grad()

            #Data to GPU
            image1 = image1.to(device)

            label = label.long().to(device)

            # Forward pass
            pred1 = model(image1)

            #Free memory
            del image1
            gc.collect()
            torch.cuda.empty_cache()

            # Calculate loss
            loss = criterion(pred1,label)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track running loss and accuracy
            running_loss += loss.item()
            total += label.size(0)
            correct += (torch.argmax(pred1,1) == label).sum().item()

            all_labels.extend(label.cpu().numpy())  # Move to CPU for confusion matrix
            all_preds.extend(torch.argmax(pred1, 1).cpu().numpy())  # Predictions
                        #Free memory
            del label, loss,pred1
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
            for image1, label in tqdm(val_dataloader, desc=f"Validating Epoch {epoch + 1}/{EPOCHS}"):
                
                #Data to GPU
                image1 = image1.to(device)
                label = label.long().to(device)

                # Forward pass
                pred1 = model(image1)

                #Free memory
                del image1
                gc.collect()
                torch.cuda.empty_cache()

                # Calculate loss
                loss = criterion(pred1,label)

                # Track running loss and accuracy
                running_loss += loss.item()

                val_total += label.size(0)
                val_correct += (torch.argmax(pred1,1) == label).sum().item()
                all_labels.extend(label.cpu().numpy())  # Move to CPU for confusion matrix
                all_preds.extend(torch.argmax(pred1, 1).cpu().numpy())  # Predictions

                #Free memory
                del pred1,label,loss
                gc.collect()
                torch.cuda.empty_cache()

        # Calculate validation loss and accuracy
        val_loss_value = running_loss / len(val_dataloader)
        val_accuracy_value = 100 * val_correct / val_total
        val_loss.append(val_loss_value)
        val_accuracy.append(val_accuracy_value)
        vcm = confusion_matrix(all_labels, all_preds)

        #If this model is better than the best
        if val_accuracy_value >= best:
            best=val_accuracy_value
            checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "model_type": model.type,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss":best
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