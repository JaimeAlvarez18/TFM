
import sys
from utils.data_acquisition import data_set,images_Dataset,test_dataset
import torch
from torch.utils.data import DataLoader
device = "cuda" if torch.cuda.is_available() else "cpu"
import gc


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
        
    BATCH_SIZE=95
    RESOLUTION=256
    MARGIN=1
    EMBEDDING_SIZE=256
    EFFICIENTNET_TYPE="efficientnet-b0"
    
    print("Getting data ...")
    loader_data = data_set('Datasets/GenImage/')
    train,val,test,y_train,y_val,y_test = loader_data.get_data()
    
    print("Creating dataloaders")
    train_dataset=test_dataset(train,y_train,device,RESOLUTION)
    train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
    
    test_data=test_dataset(test,y_test,device,RESOLUTION)
    test_dataloader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
    
    print("Loading model to classify ...")
    checkpoint=torch.load(route_classifier)
    model= Big_model(route_encoder,device,EMBEDDING_SIZE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    data,y=next(iter(train_dataloader))
    data,y=next(iter(train_dataloader))
    embs=[]
    index=[0,1,2,3,4,5,6,7]
    for p in index:
        for id,i in enumerate(y):
            if i==p:
                im=data[id]
                embs.append(im)
                break
    embs=torch.from_numpy(np.array([model.encoder.predict_one_image(emb.unsqueeze(0).to(device)).cpu() for emb in embs])).squeeze(1).to(device)
    
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

