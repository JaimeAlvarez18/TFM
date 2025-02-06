
import sys
from data_acquisition import data_set,images_Dataset,test_dataset
import torch
from torch.utils.data import DataLoader
device = "cuda" if torch.cuda.is_available() else "cpu"
import gc


from tqdm import tqdm
from models import Big_model,siamese_model
from torch import nn
import numpy as np
from sklearn.metrics import confusion_matrix




import time

def print_confusion_matrix(cm):
    # Print confusion matrix in a nice format
    # Print header
    print(" " * 8, "Predicted")
    print(" " * 8, " ".join([f"{i:4}" for i in range(cm.shape[1])]))
    
    # Print each row
    for i, row in enumerate(cm):
        print(f"True {i:2} ", " ".join([f"{val:4}" for val in row]))

def create_and_save_embeddings(model,train_dataloader,path_to_save):
    #Get embeddings representing each data generator
    dictionary={
        0:[],
        1:[],
        2:[],
        3:[],
        4:[],
        5:[],
        6:[],
        7:[],   
    }
    index=0
    model.eval()
    for image1, label in tqdm(train_dataloader, desc=f"Getting embedding {index + 1}/{len(train_dataloader)}"):
        index += 1
        image1 = image1.to(device)
        embs=model.predict_one_image(image1)
        embs=embs.detach()
        label=label.numpy()
        for ind,emb in enumerate(embs):
            dictionary[label[ind]].append(emb.cpu().numpy())

        del embs,label
        
        gc.collect()
        torch.cuda.empty_cache()
    
    for key in dictionary:
        embs=np.array(dictionary[key]).mean(axis=0)
        dictionary[key]=embs

    checkpoint = {
                    "embeddings": dictionary
                }
    torch.save(checkpoint, path_to_save)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        route_encoder=sys.argv[1]
        route_classifier=sys.argv[2]
    print("Getting data ...")
    loader_data = data_set('Datasets/GenImage/')
    train,val,test,y_train,y_val,y_test = loader_data.get_data()
    
    print("Creating dataloaders")
    train_dataset=test_dataset(train[:500000],y_train[:500000],device,256)
    train_dataloader=DataLoader(train_dataset,batch_size=39,shuffle=True,num_workers=4)
    
    test_data=test_dataset(test,y_test,device,256)
    test_dataloader=DataLoader(test_data,batch_size=64,shuffle=True,num_workers=4)
    
    # del train,val,test,y_train,y_test,y_val,test_data,train_dataset,
    # gc.collect()
    # torch.cuda.empty_cache() 
    
    
    # print("Loading model to compute embeddings ...")
    # checkpoint=torch.load(route_encoder)
    # model=siamese_model(checkpoint["model_type"],device)
    # model.load_state_dict(checkpoint["model_state_dict"])
    # model.to(device)
    
    # print("Creating embeddings ...")
    # path_embeddings='Models/Embeddings/embeddings_b0_256_new.pth'
    # create_and_save_embeddings(model,train_dataloader,path_embeddings)
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
    # del model
    # gc.collect()
    # torch.cuda.empty_cache()
    # checkpoint=torch.load(path_embeddings,weights_only=False)    
    # embs=checkpoint['embeddings']
    
    print("Loading model to classify ...")
    checkpoint=torch.load(route_classifier)
    model= Big_model(route_encoder,device,512)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    
    suma=0
    total=0
    index = 0
    all_labels = []  # List to store true labels
    all_preds = []
    # for image1, label in tqdm(train_dataloader, desc=f"Classifying {index + 1}/{len(test_dataloader)}"):
    #     index += 1
    #     image1 = image1.to(device)
        


    #     batched_predictions=[]
    #     for im in image1:
    #         predictions=[]
    #         im=im.unsqueeze(0).to(device)
    #         for key in embs:
    #             emb=key.unsqueeze(0).to(device)

    #             output=model(im,emb).detach().cpu().squeeze(0).numpy()

    #             predictions.append(output)
    #         del im
    #         gc.collect()
    #         torch.cuda.empty_cache() 
    #         predictions=np.array(predictions)

    #         pred=np.argmax(predictions[:,0])
    #         batched_predictions.append(pred)
        
    #     torch.cuda.empty_cache()
    #     label = label.numpy()
    #     all_labels.extend(label)  # Move to CPU for confusion matrix
    #     all_preds.extend(batched_predictions)  # Predictions
    #     suma += np.sum(label==batched_predictions)
    #     total += len(label)
    #     del image1,label,batched_predictions,predictions
    #     gc.collect()
    #     torch.cuda.empty_cache() 
    for image1, label in tqdm(test_dataloader, desc=f"Classifying {index + 1}/{len(test_dataloader)}"):
        index += 1
        image1 = image1.to(device)  # Move entire batch to GPU
        embs_batch = torch.stack(embs).to(device)  # Stack embeddings for parallel computation

        batched_predictions = []
        
        with torch.no_grad():  # Disable gradient computation
            for i in range(image1.shape[0]):  # Iterate over each image in the batch
                im = image1[i].unsqueeze(0).repeat(len(embs_batch), 1, 1, 1)  
                # im shape: [num_embs, C, H, W] (repeat image for each embedding)
                
                output = model(im, embs_batch)  # Compute output for all embeddings in parallel
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

