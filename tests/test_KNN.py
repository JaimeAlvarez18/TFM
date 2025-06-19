import sys
from utils.data_acquisition import data_set,images_Dataset,test_dataset,data_set_with_nature
import torch
from torch.utils.data import DataLoader
device = "cuda" if torch.cuda.is_available() else "cpu"
import gc
from utils.data_acquisition import create_and_save_ALL_embeddings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score,auc
from sklearn.model_selection import train_test_split


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
        print(f" ".join([f"{val:4}" for val in row]))




if __name__ == "__main__":
    if len(sys.argv) == 2:
        route_encoder=sys.argv[1]
        
    BATCH_SIZE=182
    RESOLUTION=256
    MARGIN=1
    EMBEDDING_SIZE=128
    EFFICIENTNET_TYPE="efficientnet-b0"
    CLASSES=5
    LOSS="SupConLoss"
    path_embeddings=f'Models/Embeddings/embeddings_128_182_5_SupConLoss.npz'
    OUTPUT=f'Results/outputs_Classification_KNN_{EMBEDDING_SIZE}_{BATCH_SIZE}_{CLASSES}_{LOSS}.csv'
    
    loader_data = data_set_with_nature('Datasets/GenImage/')
    train,val,test,y_train,y_val,y_test = loader_data.get_data()
    
    print(np.unique(np.array(y_test),return_counts=True))
    
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

    SEEDS=[39,38,37,36,35,41,42,34,33,32,31,30]
    
    data = np.load(path_embeddings)
    embs1 = data['embeddings']
    labels1 = data['labels']

    NEWS=[10,20,50]
    NEIGBOURS=[3,3,11]
    print(f'Antes de downsample {np.unique(labels1,return_counts=True)}')
    for seed in SEEDS:
        count=0
        print("*"*100)
        print("*"*100)
        print("*"*100)
        for new in NEWS:
            
            embs,_,labels,_= train_test_split(embs1,labels1,train_size=9*new,stratify=labels1,random_state=seed)
            print(f'Después de downsample {np.unique(labels,return_counts=True)}')
            
            knn=KNeighborsClassifier(n_neighbors=NEIGBOURS[count])
            count+=1
            knn.fit(embs,labels)
            
            
            
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
                    # print(predictions)
                    # print(predictions)
                    # print(label)
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
                    # print(accuracy)


            accuracy=(suma/total)*100
            
            # cm = confusion_matrix(all_labels, all_preds)
            # np.savetxt(OUTPUT,cm,delimiter=",",fmt="%d")
            
            # obj=[1,5,6,7]
            # obj1=[0,2,3,4,8]
            # precision1 =precision_score(all_labels,all_preds,labels=obj,average=None)
            # recall1 =recall_score(all_labels,all_preds,labels=obj,average=None)
            # f11=f1_score(all_labels,all_preds,labels=obj,average=None)
            
            # precision2 =precision_score(all_labels,all_preds,labels=obj,average="weighted")
            # recall2 =recall_score(all_labels,all_preds,labels=obj,average="weighted")
            # f12=f1_score(all_labels,all_preds,labels=obj,average="weighted")
            f12=f1_score(all_labels,all_preds,average="weighted")
            
            # precision3 =precision_score(all_labels,all_preds,labels=obj1,average="weighted")
            # recall3 =recall_score(all_labels,all_preds,labels=obj1,average="weighted")
            # f13=f1_score(all_labels,all_preds,labels=obj1,average="weighted")
        
            # print("-"*100)
            # print("-"*100)
            # print(f"Precis zero-shot: {precision1[0]:.4f},{precision1[1]:.4f},{precision1[2]:.4f},{precision1[3]:.4f}")
            # print(f"Recall zero-shot: {recall1[0]:.4f},{recall1[1]:.4f},{recall1[2]:.4f},{recall1[3]:.4f}")
            # print(f"F1-sco zero-shot: {f11[0]:.4f},{f11[1]:.4f},{f11[2]:.4f},{f11[3]:.4f}")
            # print("-"*100)
            # print(f"Total precis zero-shot: {precision2:.4f}")
            # print(f"Total recall zero-shot: {recall2:.4f}")
            # print(f"Total F1-sco zero-shot: {f12:.4f}")
            # print("-"*100)
            # print(f"Total precis NO zero-shot: {precision3:.4f}")
            # print(f"Total recall NO zero-shot: {recall3:.4f}")
            # print(f"Total F1-sco NO zero-shot: {f13:.4f}")
            # print("-"*100)
            # print("-"*100)  
            
            # print()
            # print('-'*60)
            # print('-'*60)
            # # Print results for the epoch
            # print("Test")
            print(f"Accuracy: {accuracy}.")
            print(f"F1: {f12}.")
            # print(f"Validation Confusion Matrix:")
            # print_confusion_matrix(cm)
            # print('-'*60)
            # print('-'*60)
            # print()
    
    