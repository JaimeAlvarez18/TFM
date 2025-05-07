

from glob import glob
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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from utils.data_acquisition import data_set_with_nature, images_Dataset,mix_dataset,create_and_save_embeddings,test_dataset, create_and_save_ALL_embeddings
from utils.models import Big_model
import numpy as np
import torch.amp as amp
import time


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
    
    BATCH_SIZE=96
    RESOLUTION=256
    MARGIN=1
    EMBEDDING_SIZE=128
    EFFICIENTNET_TYPE="efficientnet-b0"
    LOSS='EuclideanDistance1'
    NUMBER=8
    path_embeddings=f'Models/Embeddings/embeddings_{EMBEDDING_SIZE}_{BATCH_SIZE}_{NUMBER}_{LOSS}.npz' 
    PATH_TO_SAVE=f'Models/Classification_Models/Classification_embeddings_{EMBEDDING_SIZE}_{BATCH_SIZE}_{NUMBER}_{LOSS}.pth' 
    OUTPUT=f'Results/outputs_Classification_Embeddings_{EMBEDDING_SIZE}_{BATCH_SIZE}_{NUMBER}_{LOSS}.csv' 


    print("Getting data ...")
    loader_data = data_set_with_nature('Datasets/GenImage/')
    train,val,test,y_train,y_val,y_test = loader_data.get_data()
    
    train_dataset=test_dataset(train,y_train,device,RESOLUTION)
    train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
    
    print("Creating model...")
    if retrain:
        checkpoint=torch.load(route_classifier,weights_only=False)
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
    

    # embs1,classes =  create_and_save_ALL_embeddings(model.encoder,train_dataloader,path_embeddings)
    
    # embs = np.array([embs1[classes == cls].mean(axis=0) for cls in np.unique(classes)])
    
    data = np.load(path_embeddings)
    embs1 = data['embeddings']
    classes = data['labels']
    embs = np.array([embs1[classes == cls].mean(axis=0) for cls in np.unique(classes)])
    
    

    print('Preparing data ...')
    train1,train2,y_train=loader_data.make_pairs_embeddings(train,embs, y_train)
    val1,val2,y_val=loader_data.make_pairs_embeddings(val, embs,y_val)
    test1,test2,y_test=loader_data.make_pairs_embeddings(test,embs, y_test)
    
    embs=torch.from_numpy(embs).to(device) 


    print("Creating dataloaders ...")
    train_dataset=mix_dataset(train1,train2,y_train,device,RESOLUTION)
    train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=12,prefetch_factor=8)

    val_dataset=mix_dataset(val1,val2,y_val,device,RESOLUTION)
    val_dataloader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=12,prefetch_factor=8)

    test_data=mix_dataset(test1,test2,y_test,device,RESOLUTION)
    test_dataloader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=12,prefetch_factor=8)


    del train,val,test,y_train,y_test,y_val,train_dataset,val_dataset,test_data
    gc.collect()
    torch.cuda.empty_cache() 



    print("Training model...")
    EPOCHS=1
    train_loss=[]
    train_accuracy=[]
    best=0.0
    val_loss=[]
    val_accuracy=[]
    scaler = amp.GradScaler()

    for epoch in range(EPOCHS):
        # # Set model to training mode
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []  # List to store true labels
        all_preds = []   # List to store predictions

        # Training loop
        for image1, embedding, label in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}"):
            
            #Reset optimizer
            optimizer.zero_grad()

            #Data to GPU
            image1 = image1.to(device)
            embedding = embedding.to(device)
            label = label.long().to(device)

            # Forward pass
            with amp.autocast(device_type='cuda'):
                pred = model.encoder.predict_one_image(image1)
                pred=model.classify_embeddings(pred,embedding)
                # Calculate loss
                loss = criterion(pred,label)


                        #Free memory
            del image1,embedding
            gc.collect()
            torch.cuda.empty_cache()



            # Backward pass and optimization
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
            checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "encoder_path": model.path_to_encoder,
                    "optimizer_state_dict":optimizer.state_dict(),
                    "encoder_type" : model.encoder_type,
                }
            torch.save(checkpoint, PATH_TO_SAVE)

        # Calculate training loss and accuracy

        # train_loss_value = running_loss / len(train_dataloader)
        # train_accuracy_value = 100 * correct / total
        # train_loss.append(train_loss_value)
        # train_accuracy.append(train_accuracy_value)         
        # tcm = confusion_matrix(all_labels, all_preds)
        

        # # Set model to evaluation mode
        # model.eval()

        # # Initialize validation stats
        # running_loss = 0.0
        # val_correct = 0
        # val_total = 0
        # all_labels = []  # List to store true labels
        # all_preds = []   # List to store predictions

        # # Validation loop
        # with torch.no_grad():
        #     for image1, embedding, label in tqdm(test_dataloader, desc=f"Validating Epoch {epoch + 1}/{EPOCHS}"):
                
        #         #Data to GPU
        #         image1 = image1.to(device)
        #         embedding = embedding.to(device)
        #         label = label.long().to(device)

        #         with amp.autocast(device_type='cuda'):
        #             # Forward pass
        #             pred = model.encoder.predict_one_image(image1)
        #             pred=model.classify_embeddings(pred,embedding)
        #             # Calculate loss
        #             loss = criterion(pred,label)

        #         #Free memory
        #         del image1,embedding
        #         gc.collect()
        #         torch.cuda.empty_cache()



        #         # Track running loss and accuracy
        #         running_loss += loss.item()
        #         val_total += label.size(0)
        #         val_correct += (torch.argmax(pred,1) == label).sum().item()
        #         all_labels.extend(label.cpu().numpy())  # Move to CPU for confusion matrix
        #         all_preds.extend(torch.argmax(pred, 1).cpu().numpy())  # Predictions

            
                

        #         #Free memory
        #         del pred,label,loss
        #         gc.collect()
        #         torch.cuda.empty_cache()


        # # Calculate validation loss and accuracy
        # val_loss_value = running_loss / len(val_dataloader)
        # val_accuracy_value = 100 * val_correct / val_total
        # val_loss.append(val_loss_value)
        # val_accuracy.append(val_accuracy_value)
        # vcm = confusion_matrix(all_labels, all_preds) 
        # if val_accuracy_value >= best:
        #     best=val_loss_value
        #     checkpoint = {
        #             "model_state_dict": model.state_dict(),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             "encoder_path": model.path_to_encoder,
        #             "encoder_type" : model.encoder_type,
        #             "best_result":best

        #         }
        #     torch.save(checkpoint, PATH_TO_SAVE)


        

        # print()
        # print('-'*60)
        # print('-'*60)
        # # Print results for the epoch
        # print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        # print()
        # print('*'*60)
        # print("Train")
        # print(f"Loss: {train_loss_value:.4f}. Accuracy: {train_accuracy_value}.")
        # print(f"Training Confusion Matrix:")
        # print_confusion_matrix(tcm)
        # print()
        # print('*'*60)
        # print()
        # print("Validation")
        # print(f"Loss: {val_loss_value:.4f}. Accuracy: {val_accuracy_value}.")
        # print(f"Validation Confusion Matrix:")
        # print_confusion_matrix(vcm)
        # print('-'*60)
        # print('-'*60)
        # print()