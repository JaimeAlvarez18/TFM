import sys
from utils.data_acquisition import data_set,images_Dataset,test_dataset,data_set_with_nature,data_set_binary_with_nature,data_set_binary_synth
import torch
from torch.utils.data import DataLoader
device = "cuda" if torch.cuda.is_available() else "cpu"
import gc
from utils.data_acquisition import create_and_save_ALL_embeddings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score,roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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
    MARGIN=1
    EMBEDDING_SIZE=128
    EFFICIENTNET_TYPE="efficientnet-b0"
    CLASSES=2
    LOSS="SupConLoss"
    path_embeddings=f'Models/Embeddings/embeddings_{EMBEDDING_SIZE}_{BATCH_SIZE}_SYNTH_{LOSS}_TODOS.npz'
    OUTPUT=f'Results/outputs_Binary_Classification_KNN_{EMBEDDING_SIZE}_{BATCH_SIZE}_{CLASSES}_{LOSS}.csv'
    
    loader_data = data_set_binary_with_nature('Datasets/GenImage/')
    testing_data=data_set_binary_synth()
    # test,y_test = testing_data.get_data_test()
    all_train,all_test,all_y_train,all_y_test=loader_data.get_data()
    
    checkpoint=torch.load(route_encoder,weights_only=False)
    model= siamese_model(checkpoint["model_type"],device,EMBEDDING_SIZE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    data = np.load(path_embeddings)
    embs = data['embeddings']
    labels = data['labels']
    # mask = (all_labels == i) | (all_labels == 5)
    embs=embs
    labels=labels



    
    knn=KNeighborsClassifier(n_neighbors=11)
    # knn = RandomForestClassifier(n_estimators=2)
    knn.fit(embs,labels)
    
    
    
    suma=0
    total=0
    index = 0
    #Convert indexes of generators in dataset with nature to generators with no nature

    
    for i in range(len(all_test)):
        # transformation=dictionary.get(i)
        all_roc=[]
        all_acc=[]
        temp_train=all_train[i]
        temp_test=all_test[i]
        temp_y_test=all_y_test[i]
        temp_y_train=all_y_train[i]
        temp_train = [item for sublist in temp_train for item in sublist]
        temp_test = [item for sublist in temp_test for item in sublist]
        temp_y_train = [item for sublist in temp_y_train for item in sublist]
        temp_y_test = [item for sublist in temp_y_test for item in sublist]
        
        # print(np.unique(np.array(temp_y_test),return_counts=True))



        temp_train,temp_val,temp_y_train,temp_y_val = train_test_split(temp_train,temp_y_train,train_size=0.9,stratify=temp_y_train,random_state=42)
        print("Creating dataloaders")
        train_dataset=test_dataset(temp_train,temp_y_train,device,RESOLUTION)
        train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=8,prefetch_factor=8)
        test_data=test_dataset(temp_test,temp_y_test,device,RESOLUTION)
        test_dataloader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=8,prefetch_factor=8)
        all_labels = []  # List to store true labels
        all_preds = []

        for image1, label in tqdm(test_dataloader, desc=f"Classifying {index + 1}/{len(test_dataloader)}"):
            with torch.no_grad():
                index += 1
                image1 = image1.to(device)  # Move entire batch to GPU
                embeddings=model.predict_one_image(image1)
                embeddings=embeddings.cpu().numpy()
                # embs_batch = torch.stack(embs).to(device)  # Stack embeddings for parallel computation
                predictions=knn.predict_proba(embeddings)
                # Convert label to numpy and store results
                label = label.cpu().numpy()
                all_labels.extend(label)
                all_preds.extend(predictions)
                
                # suma += np.sum(label == predictions)
                total += len(label)
                
                # Free memory
                del image1, label, predictions, embeddings
                gc.collect()
                torch.cuda.empty_cache()

        
        all_labels=np.array(all_labels)
        all_preds=np.array(all_preds)
        print(all_preds.shape)

        pr=[0 if pred[5]> pred[i] else 1 for pred in all_preds]
        labels=[0 if label ==5 else 1 for label in all_labels]
        roc1=roc_auc_score(labels, 1-all_preds[:,5])
        roc=roc_auc_score(labels, all_preds[:,5])
        if roc1>roc:
            roc=roc1
            
        print(f"ROC AUC for class {i} vs Real: {roc:.4f}")
        accuracy=accuracy_score(all_labels,pr)
        print(f"Accuracy for class {i} vs Real: {accuracy:.4f}")
        
        
        # preds=all_preds[:,5]
        # print(all_preds.shape)
        # rocs=[]
        # accuracies=[]
        
        # unique_labels = np.unique(all_labels)

        # for i in unique_labels:
        #     if i != 5:
        #         # Create a boolean mask for class i and class 5
        #         mask = (all_labels == i) | (all_labels == 5)

        #         # Filter labels and predictions
        #         labels = all_labels[mask]
        #         preds_subset = preds[mask]
        #         if i>5:
        #             preds_subset=1-preds_subset

        #         # Binarize labels: make class 5 the positive class (1), class i the negative (0)

        #         # Compute ROC AUC
        #         roc = roc_auc_score(labels, preds_subset)
        #         rocs.append(roc)
        #         print("-"*50)
        #         print(f"ROC AUC for class {i} vs Real: {roc:.4f}")
                
        #         pr=[5 if pred[5]> pred[int(i)] else i for pred in all_preds[mask]]

        #         accuracy=accuracy_score(labels,pr)
        #         print(f"Accuracy for class {i} vs Real: {accuracy:.4f}")
        
    # rocs=[]
    
    # for i in range(np.unique(np.array(all_labels))):
    #     if i!=5:
    #         preds1=all_labels[:,[i,5]]
    #         labels=[]
    #         preds=[]
            
    #         for index, i in enumerate(all_preds):
    #             partial = np.sum(i)-i[5]
    #             pr=[partial,i[5]]
    #             preds.append(pr)
    #             if i[5]>partial:
    #                 labels.append(1)
    #             else:
    #                 labels.append(0)
                    
    #         all_labels=[1 if labe==5 else 0 for labe in all_labels]
            
    #         roc=roc_auc_score(all_labels,np.array(preds)[:,1])
    #         print(roc)
    #         rocs.append(roc)
    
    # accuracy=(suma/total)*100
    
    # for index,i in enumerate(np.unique(np.array(all_labels))):
    #     if i !=5:
            
    
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
    
    # precision3 =precision_score(all_labels,all_preds,labels=obj1,average="weighted")
    # recall3 =recall_score(all_labels,all_preds,labels=obj1,average="weighted")
    # f13=f1_score(all_labels,all_preds,labels=obj1,average="weighted")
    
    # print("-"*100)
    # print("-"*100)
    # print(f"Precis zero-shot: BigGan{precision1[0]:.4f}; Real {precision1[1]:.4f}; SD 1.4 {precision1[2]:.4f}; SD 1.5 {precision1[3]:.4f}")
    # print(f"Recall zero-shot: BigGan{recall1[0]:.4f}; Real {recall1[1]:.4f}; SD 1.4 {recall1[2]:.4f}; SD 1.5 {recall1[3]:.4f}")
    # print(f"F1-sco zero-shot: BigGan{f11[0]:.4f}; Real {f11[1]:.4f}; SD 1.4 {f11[2]:.4f}; SD 1.5 {f11[3]:.4f}")
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
    # print(f"Accuracy: {accuracy}.")
    # print(f"Validation Confusion Matrix:")
    # print_confusion_matrix(cm)
    # print('-'*60)
    # print('-'*60)
    # print()
    
    