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
    path_embeddings=f'Models/Embeddings/MINIBATCH_128_2048_ForenSynths_Partial_Checkpoint.npz' #Embeddigs de entrenamiento de ForenSynths
    OUTPUT=f'Results/outputs_Binary_Classification_KNN_{EMBEDDING_SIZE}_{BATCH_SIZE}_{CLASSES}_{LOSS}.csv'
    
    loader_data = data_set_binary_with_nature('GenImage_resized/')

    _,all_test,_,all_y_test = loader_data.get_data()
    
    checkpoint=torch.load(route_encoder,weights_only=False)
    model= siamese_model(checkpoint["model_type"],device,EMBEDDING_SIZE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    data = np.load(path_embeddings)
    embs = data['embeddings']
    labels = data['labels']
    print(embs.shape)
    # mask = (all_labels == i) | (all_labels == 5)
    embs=embs
    labels=labels



    
    knn=KNeighborsClassifier(n_neighbors=1001,n_jobs=-1)
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

        temp_test=all_test[i]
        temp_y_test=all_y_test[i]



        
        # print(np.unique(np.array(temp_y_test),return_counts=True))



        print("Creating dataloaders")
        test_data=test_dataset(temp_test,temp_y_test,device,RESOLUTION)
        test_dataloader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,prefetch_factor=2)
        all_labels = []  # List to store true labels
        all_preds = []
            
        suma=0
        total=0
        index = 0

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

        pr=np.array([np.argmax(np.array(pred)) for pred in all_preds])
        # print("Predictions distribution:")
        # print(np.unique(pr,return_counts=True))

        
        labels=np.array([0 if label == 8 else 1 for label in all_labels])
        # print("True labels distribution:")
        # print(np.unique(np.array(labels),return_counts=True))
        roc1=roc_auc_score(labels, 1-pr)
        roc=roc_auc_score(labels, pr)
        if roc1>roc:
            roc=roc1
            
        print(f"ROC AUC for class {i} vs Real: {roc:.4f}")
        accuracy=accuracy_score(labels,pr)
        print(f"Accuracy for class {i} vs Real: {accuracy:.4f}")
        
    
    