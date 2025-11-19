# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: /home/jalvarez/work/TFM/tests/test_KNN_SOURCE_ATTRIBUTION_SYNTH_TRAINED_Few_Shot_Mamba.py
# Bytecode version: 3.11a7e (3495)
# Source timestamp: 2025-11-12 21:37:19 UTC (1762983439)

import sys
import warnings
warnings.filterwarnings('ignore')
from utils.data_acquisition import data_set, images_Dataset, test_dataset, data_set_with_nature, data_set_binary_with_nature, data_set_binary_synth
import torch
from torch.utils.data import DataLoader
print(torch.cuda.is_available())
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(device)
import gc
from utils.data_acquisition import create_and_save_ALL_embeddings_Mamba
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import transformers
from transformers import AutoModelForImageClassification
from utils.data_acquisition import compute_oscr_5classes
import joblib
from tqdm import tqdm
from utils.models import Big_model, siamese_model
from torch import nn
import numpy as np
from sklearn.metrics import confusion_matrix
import torch.amp as amp
import time

def print_confusion_matrix(cm):
    print('        ', 'Predicted')
    print('        ', ' '.join([f'{i:4}' for i in range(cm.shape[1])]))
    for i, row in enumerate(cm):
        print(f'True {i:2} ', ' '.join([f'{val:4}' for val in row]))
if __name__ == '__main__':
    if len(sys.argv) == 2:
        route_encoder = sys.argv[1]
    BATCH_SIZE = 32
    RESOLUTION = 256
    MARGIN = 1
    EMBEDDING_SIZE = 128
    EFFICIENTNET_TYPE = 'efficientnet-b0'
    CLASSES = 2
    LOSS = 'SupConLoss'
    path_embeddings = 'Models/Embeddings/embeddings_128_182_MD_Real_VQDM_SupConLoss.npz'
    loader_data = data_set_with_nature('GenImage_resized/')
    train, val, test, y_train, y_val, y_test = loader_data.get_data()
    print(np.unique(np.array(y_train), return_counts=True))
    print(np.unique(np.array(y_test), return_counts=True))




    n_generators = 9

    # test_indices = [0, 1, 3, 6]  # vals = ["glide","ADM","stable_diffusion_v_1_5","BigGan"] #New ES1
    # train_indices = [2, 4, 5, 7, 8]


    # test_indices = [0, 2, 6,8]  # vals = ["wukong","glide","ADM","stable_diffusion_v_1_5"] #New ES2
    # train_indices = [1, 3, 4,5, 7]    

    # test_indices = [2,3, 6,8]  # vals = ["wukong","glide","stable_diffusion_v_1_5","Midjourney"] #New ES3        
    # train_indices = [0,1, 4,5, 7]


    test_indices = [2, 5,7,8]  # vals = ["wukong","stable_diffusion_v_1_4","Midjourney","vqdm"] #NEw ES5      
    train_indices = [0,1, 3,4,6]   
        


#         ['ADM' 'BigGan' 'Midjourney' 'glide' 'real' 'stable_diffusion_v_1_4'
#  'stable_diffusion_v_1_5' 'vqdm' 'wukong']




    model = AutoModelForImageClassification.from_pretrained('nvidia/MambaVision-L3-256-21K', trust_remote_code=True, dtype='auto').to('cpu')
    checkpoint = torch.load(route_encoder, map_location='cpu', weights_only=False)
    print(f"Batch losses: {checkpoint['batch_losses']}")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    ks = [150]
    for k in ks:
        data_selected = []
        y_selected = []
        
        for i in range(n_generators):
            if i not in test_indices:
                # Regular generators: k samples
                indices = np.random.choice(np.argwhere(y_train == i).flatten(), size=k, replace=False)
            else:
                # Test generators: more samples (k * number of test indices)
                indices = np.random.choice(np.argwhere(y_train == i).flatten(), size=int(k / len(test_indices)), replace=False)
            
            data_selected.extend(np.array(train)[indices])
            y_selected.extend(np.array(y_train)[indices])
        y_selected = [10 if i in test_indices else i for i in y_selected]
        y_train = [10 if i in test_indices else i for i in y_train]
        y_test = [10 if i in test_indices else i for i in y_test]
        print(np.unique(np.array(y_selected), return_counts=True), np.unique(np.array(y_train), return_counts=True), np.unique(np.array(y_test), return_counts=True))
        train_dataset = test_dataset(data_selected, y_selected, device, RESOLUTION)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, prefetch_factor=1)

        # test,_,y_test,_=train_test_split(test,y_test,train_size=0.02,stratify=y_test)
        test_data = test_dataset(test, y_test, device, RESOLUTION)
        test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, prefetch_factor=1)
        train_embeddings, label_embeddings = create_and_save_ALL_embeddings_Mamba(model, train_dataloader=train_dataloader, path='Models/15000_NewES1.npz', save=False, device1=device)
        # print(train_embeddings.shape, label_embeddings.shape)
        # print(train_embeddings, label_embeddings)
        if k < 11:
            knn = KNeighborsClassifier(n_neighbors=3)
        else:  # inserted
            knn = KNeighborsClassifier(n_neighbors=11, n_jobs=24)
        knn.fit(train_embeddings, label_embeddings)
        gc.collect()
        torch.cuda.empty_cache()
        suma = 0
        total = 0
        index = 0
        all_labels = []
        all_probs = []
        for image1, label in tqdm(test_dataloader, desc=f'Classifying {index + 1}/{len(test_dataloader)}'):
            index += 1
            with torch.no_grad():
                image1 = image1.to(device)
                outputs = model(image1)['logits'].cpu().numpy()
                label = label.cpu().numpy()
                probs = knn.predict_proba(outputs)
                all_labels.extend(label)
                all_probs.extend(probs)
                del image1, label, probs, outputs
                gc.collect()
                torch.cuda.empty_cache()
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        valid_indices = np.unique(np.array(all_labels))
        valid_indices = np.delete(valid_indices, np.where(valid_indices == 10))
        open_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        print(f'Open AUC: {open_auc}')
        oscr, _, _, _ = compute_oscr_5classes(all_labels, all_probs)
        print(f'Open OSCR: {oscr}')
        mask = np.isin(all_labels, valid_indices)
        filtered_labels = all_labels[mask]
        filtered_probs = all_probs[mask][:, :(-1)]
        filtered_probs = filtered_probs / filtered_probs.sum(axis=1, keepdims=True)
        label_mapping = {old: new for new, old in enumerate(valid_indices)}
        remapped_labels = np.array([label_mapping[label] for label in filtered_labels])
        predicted_labels = np.argmax(filtered_probs, axis=1)
        accuracy = accuracy_score(remapped_labels, predicted_labels)
        print(f'Closed accuracy: {accuracy}')