import sys
from utils.data_acquisition import data_set,images_Dataset,test_dataset
import torch
from torch.utils.data import DataLoader
device = "cuda" if torch.cuda.is_available() else "cpu"
import gc
from utils.data_acquisition import create_and_save_ALL_embeddings
from sklearn.neighbors import KNeighborsClassifier


from tqdm import tqdm
from utils.models import Big_model,siamese_model
from torch import nn
import numpy as np
from sklearn.metrics import confusion_matrix
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
    if len(sys.argv) == 2:
        route_encoder=sys.argv[1]
        
    BATCH_SIZE=182
    RESOLUTION=256
    EMBEDDING_SIZE=128
    
    loader_data = data_set('Datasets/GenImage/')
    train,val,test,y_train,y_val,y_test = loader_data.get_data()
    
    print("Creating dataloaders")
    train_dataset=test_dataset(train,y_train,device,RESOLUTION)
    train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=8,prefetch_factor=8)
    test_data=test_dataset(test,y_test,device,RESOLUTION)
    test_dataloader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=8,prefetch_factor=8)
    
    checkpoint=torch.load(route_encoder,weights_only=False)
    model= siamese_model(checkpoint["model_type"],device,EMBEDDING_SIZE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    embs1,classes =  create_and_save_ALL_embeddings(model,train_dataloader)


    
    knn=KNeighborsClassifier(n_neighbors=1)
    knn.fit(embs1,classes)
    
    
    
    suma=0
    total=0
    index = 0
    all_labels = []  # List to store true labels
    all_preds = []

    for image1, label in tqdm(test_dataloader, desc=f"Classifying {index + 1}/{len(test_dataloader)}"):
        with torch.no_grad():
            index += 1
            image1 = image1.to(device)  # Move entire batch to GPU
            embeddings=model.predict_one_image(image1)
            embeddings=embeddings.cpu().numpy()
            # embs_batch = torch.stack(embs).to(device)  # Stack embeddings for parallel computation
            predictions=knn.predict(embeddings)
            # Convert label to numpy and store results
            label = label.cpu().numpy()
            all_labels.extend(label)
            all_preds.extend(predictions)
            
            suma += np.sum(label == predictions)
            total += len(label)
            
            # Free memory
            del image1, label, predictions, embeddings
            gc.collect()
            torch.cuda.empty_cache()
            accuracy=(suma/total)*100
            print(accuracy)


    accuracy=(suma/total)*100
    cm = confusion_matrix(all_labels, all_preds)
    print()
    print('-'*60)
    print('-'*60)
    # Print results for the epoch
    print("Test")
    print(f"Accuracy: {accuracy}.")
    print(f"Validation Confusion Matrix:")
    print_confusion_matrix(cm)
    print('-'*60)
    print('-'*60)
    print()
    
    