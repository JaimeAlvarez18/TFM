
import sys
from utils.data_acquisition import data_set,images_Dataset,test_dataset, create_and_save_embeddings,data_set_with_nature
import torch
from torch.utils.data import DataLoader
device = "cuda" if torch.cuda.is_available() else "cpu"
import gc
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

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
    if len(sys.argv) == 3:
        route_encoder=sys.argv[1]
        route_classifier=sys.argv[2]
        
    
    BATCH_SIZE=96
    RESOLUTION=256
    MARGIN=1
    EMBEDDING_SIZE=128
    EFFICIENTNET_TYPE="efficientnet-b0"
    LOSS="EuclideanDistance1"
    CLASSES=8
    path_embeddings=f'Models/Embeddings/embeddings_{EMBEDDING_SIZE}_{BATCH_SIZE}_{CLASSES}_{LOSS}.npz'
    OUTPUT=f'Results/outputs_Classification_Embeddings_{EMBEDDING_SIZE}_{BATCH_SIZE}_{CLASSES}_{LOSS}.csv'
        
    print("Getting data ...")
    loader_data = data_set_with_nature('Datasets/GenImage/')
    train,val,test,y_train,y_val,y_test = loader_data.get_data()


    print("Creating dataloaders")
    train_data=test_dataset(train,y_train,device,RESOLUTION)
    train_dataloader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
    
    test_data=test_dataset(test,y_test,device,RESOLUTION)
    test_dataloader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
    
    del train,val,test,y_train,y_test,y_val,test_data
    gc.collect()
    torch.cuda.empty_cache() 
    

    
    print("Loading model to classify ...")
    checkpoint=torch.load(route_classifier)
    model= Big_model(route_encoder,device,EMBEDDING_SIZE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    data = np.load(path_embeddings)
    embs = data['embeddings']
    labels = data['labels']
    
    
    embs = np.array([embs[labels == cls].mean(axis=0) for cls in np.unique(labels)])
    embs=torch.from_numpy(embs).to(device)        
    
    suma=0
    total=0
    index = 0
    all_labels = []  # List to store true labels
    all_preds = []
    for image1, label in tqdm(test_dataloader, desc=f"Classifying {index + 1}/{len(test_dataloader)}"):
        index += 1
        image1 = image1.to(device)  # Move entire batch to GPU
        embeddings=model.encoder.predict_one_image(image1)
        # embs_batch = torch.stack(embs).to(device)  # Stack embeddings for parallel computation

        batched_predictions = []
        
        with torch.no_grad():  # Disable gradient computation
            for i in range(embeddings.shape[0]):  # Iterate over each image in the batch
                im = embeddings[i].repeat(len(embs), 1)  
                # im shape: [num_embs, C, H, W] (repeat image for each embedding)
                

                with amp.autocast(device_type='cuda'):
                    output = model.classify_embeddings(im, embs)  # Compute output for all embeddings in parallel
                output = output.cpu().numpy()

                pred = np.argmax(output[:, 0])  # Find best match
                
                batched_predictions.append(pred)
        
        # Convert label to numpy and store results
        label = label.cpu().numpy()
        all_labels.extend(label)
        all_preds.extend(batched_predictions)
        
        suma += np.sum(label == batched_predictions)
        total += len(label)
        
        # Free memory
        del image1, label, batched_predictions, output
        gc.collect()
        torch.cuda.empty_cache()
        accuracy=(suma/total)*100


    accuracy=(suma/total)*100
    
    cm = confusion_matrix(all_labels, all_preds)
    np.savetxt(OUTPUT,cm,delimiter=",",fmt="%d")
    
    obj=[1,5,6,7]
    obj1=[0,2,3,4,8]
    precision1 =precision_score(all_labels,all_preds,labels=obj,average=None)
    recall1 =recall_score(all_labels,all_preds,labels=obj,average=None)
    f11=f1_score(all_labels,all_preds,labels=obj,average=None)
    
    precision2 =precision_score(all_labels,all_preds,labels=obj,average="weighted")
    recall2 =recall_score(all_labels,all_preds,labels=obj,average="weighted")
    f12=f1_score(all_labels,all_preds,labels=obj,average="weighted")
    
    precision3 =precision_score(all_labels,all_preds,labels=obj1,average="weighted")
    recall3 =recall_score(all_labels,all_preds,labels=obj1,average="weighted")
    f13=f1_score(all_labels,all_preds,labels=obj1,average="weighted")
    
    print("-"*100)
    print("-"*100)
    print(f"Precis zero-shot: BigGan{precision1[0]:.4f}; Real {precision1[1]:.4f}; SD 1.4 {precision1[2]:.4f}; SD 1.5 {precision1[3]:.4f}")
    print(f"Recall zero-shot: BigGan{recall1[0]:.4f}; Real {recall1[1]:.4f}; SD 1.4 {recall1[2]:.4f}; SD 1.5 {recall1[3]:.4f}")
    print(f"F1-sco zero-shot: BigGan{f11[0]:.4f}; Real {f11[1]:.4f}; SD 1.4 {f11[2]:.4f}; SD 1.5 {f11[3]:.4f}")
    print("-"*100)
    print(f"Total precis zero-shot: {precision2:.4f}")
    print(f"Total recall zero-shot: {recall2:.4f}")
    print(f"Total F1-sco zero-shot: {f12:.4f}")
    print("-"*100)
    print(f"Total precis NO zero-shot: {precision3:.4f}")
    print(f"Total recall NO zero-shot: {recall3:.4f}")
    print(f"Total F1-sco NO zero-shot: {f13:.4f}")
    print("-"*100)
    print("-"*100)
    
    print(cm)
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

