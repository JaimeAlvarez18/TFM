import numpy as np
from glob import glob
import random
random.seed(42)
import os
from torch.utils.data import Dataset
import cv2
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import torch.amp as amp
device='cuda' if torch.cuda.is_available() else 'cpu'
import json
import gc       


class images_Dataset(Dataset):

    def __init__(self,route_images,classes,device,size):
        self.route_images=route_images
        self.classes=classes
        self.device=device
        self.size=size


    def __len__(self):
        return len(self.route_images)
    
    def __getitem__(self, index):
        #Get images and label
        im1,im2=self.route_images[index]
        y=self.classes[index]

        #load images
        image1 = cv2.imread(im1)

        image2 = cv2.imread(im2)
            

        #If any image is corrupted
        if type(image1) == type(None):
            print(f"Warning: Image at {im1}")
            os.remove(im1)
        
        if  type(image2) == type(None):
            print(f"Warning: Image at {im2}")
            os.remove(im2)

        #to RGB     
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        
        
        #scale images to [0,1]
        image1=image1/255.0
        image2=image2/255.0
        

        #Rescale resolution to self.size,self.size
        image1 = cv2.resize(image1, (self.size, self.size))
        image2 = cv2.resize(image2, (self.size, self.size))
        

        #Permute dimensions (Channels, Width, Height)
        image1=np.transpose(image1,[2,0,1])
        image2=np.transpose(image2,[2,0,1])
        

        #Data to tensor
        image1=torch.from_numpy(image1).to(torch.float16)
        image2=torch.from_numpy(image2).to(torch.float16)
        y=torch.tensor(y).to(torch.float16)

        return image1,image2, y
    
class test_dataset(Dataset):
    def __init__(self,X,y,device,size):
        self.device=device
        self.routes=X
        self.labels=y
        self.size=size
    def __getitem__(self, index):
        im1=self.routes[index]
        y=self.labels[index]

        #load images
        image1 = cv2.imread(im1)
 

        #If any image is corrupted
        if type(image1) == type(None):
            print(f"Warning: Image at {im1}")
            os.remove(im1)

        #to RGB     
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        
        #scale images to [0,1]
        image1=image1/255.0

        #Rescale resolution to self.size,self.size
        image1 = cv2.resize(image1, (self.size, self.size))


        #Permute dimensions (Channels, Width, Height)
        image1=np.transpose(image1,[2,0,1])

        #Data to tensor
        image1=torch.from_numpy(image1).to(torch.float32)
        y=torch.tensor(y).to(torch.float32)

        return image1,y
    def __len__(self):
        return len(self.routes)
    
class mix_dataset(Dataset):
    def __init__(self,X,embs,y,device,size):
        self.device=device
        self.routes=X
        self.labels=y
        self.size=size
        self.embs=embs
    def __getitem__(self, index):
        im1=self.routes[index]
        emb=self.embs[index]
        y=self.labels[index]

        #load images
        image1 = cv2.imread(im1)
 

        #If any image is corrupted
        if type(image1) == type(None):
            print(f"Warning: Image at {im1}")
            os.remove(im1)

        #to RGB     
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        
        #scale images to [0,1]
        image1=image1/255.0

        #Rescale resolution to self.size,self.size
        image1 = cv2.resize(image1, (self.size, self.size))


        #Permute dimensions (Channels, Width, Height)
        image1=np.transpose(image1,[2,0,1])

        #Data to tensor
        image1=torch.from_numpy(image1).to(torch.float32)
        emb=torch.from_numpy(emb).to(torch.float32)
        y=torch.tensor(y).to(torch.float32)

        return image1,emb,y
    def __len__(self):
        return len(self.routes)
    
class data_set():
    def __init__(self,route):
        self.route=route

    def get_data(self,train_size=0.9,random_state=5):
        train=[]
        test=[]
        y_train=[]
        y_test=[]

        itera=os.walk(self.route)
        datasets=next(iter(itera))[1]
        for index,dataset in enumerate(datasets):
            direct=self.route+dataset+'/'
            train.append(glob(direct+'train/ai/*.PNG')+glob(direct+'train/ai/*.png'))
            test.append(glob(direct+'val/ai/*.PNG')+glob(direct+'val/ai/*.png'))
            y_train.append([dataset]*len(train[index]))
            y_test.append([dataset]*len(test[index]))

        train = [item for sublist in train for item in sublist]
        test = [item for sublist in test for item in sublist]
        y_train = [item for sublist in y_train for item in sublist]
        y_test = [item for sublist in y_test for item in sublist]

        label_encoder=LabelEncoder().fit(y_train)
        y_train=label_encoder.transform(y_train)
        y_test=label_encoder.transform(y_test)
        print(label_encoder.classes_)


        train,val,y_train,y_val = train_test_split(train,y_train,train_size=train_size,stratify=y_train,random_state=random_state)

        return train,val,test,y_train,y_val,y_test
    def make_pairs(self,x, y):
        """Creates a tuple containing image pairs with corresponding label.

        Arguments:
            x: List containing images, each index in this list corresponds to one image.
            y: List containing labels, each label with datatype of `int`.

        Returns:
            Tuple containing two numpy arrays as (pairs_of_samples, labels),
            where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
            labels are a binary array of shape (2len(x)).
        """

        num_classes = max(y) + 1
        digit_indices = [np.where(y == i)[0] for i in range(num_classes)] # 10 vectores con los índices de cada número

        pairs = []
        labels = []
        
        #For each sample in the set.
        for idx1 in range(len(x)):

            # add a matching example

            #Get a sample
            x1 = x[idx1]

            #Label of the sample   
            label1 = y[idx1]

            #Get randomly a sample with the same label
            idx2 = random.choice(digit_indices[label1])
            x2 = x[idx2]

            #Create the pair and the label
            pairs += [[x1, x2]]  
            labels += [0]

            #Add a non-matching sample
            label2 = random.randint(0, num_classes - 1)

            #Get a label different to the initial one
            while label2 == label1:
                label2 = random.randint(0, num_classes - 1)

            #Get a sample of the previous label calculated (different to the initial one)
            idx2 = random.choice(digit_indices[label2])
            x2 = x[idx2]

            #Create the pair and the label
            pairs += [[x1, x2]] 
            labels += [1] 

        return np.array(pairs), np.array(labels).astype("float32")
    
    
    def make_pairs_embeddings(self,x,embs, y):
        """Creates a tuple containing image pairs with corresponding label.

        Arguments:
            x: List containing images, each index in this list corresponds to one image.
            y: List containing labels, each label with datatype of `int`.

        Returns:
            Tuple containing two numpy arrays as (pairs_of_samples, labels),
            where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
            labels are a binary array of shape (2len(x)).
        """

        num_classes = max(y) + 1

        pairs1 = []
        pairs2=[]
        labels = []
        
        #For each sample in the set.
        for idx1 in range(len(x)):

            # add a matching example

            #Get a sample
            x1 = x[idx1]

            #Label of the sample   
            label1 = y[idx1]

            #Get randomly a sample with the same label
            x2 = embs[label1]

            #Create the pair and the label
            pairs1 += [x1]
            pairs2 += [x2]    
            labels += [0]

            #Add a non-matching sample
            label2 = random.randint(0, num_classes - 1)

            #Get a label different to the initial one
            while label2 == label1:
                label2 = random.randint(0, num_classes - 1)

            #Get a sample of the previous label calculated (different to the initial one)

            x2 = embs[label2]

            #Create the pair and the label
            pairs1 += [x1]
            pairs2 += [x2]  
            labels += [1] 

        return np.array(pairs1),np.array(pairs2), np.array(labels).astype("float32")
    
class data_set_with_nature():
    def __init__(self,route):
        self.route=route
        self.route_nature="Datasets/GenImageOther/BigGan"

    def get_data(self,train_size=0.9,random_state=5):
        train=[]
        test=[]
        y_train=[]
        y_test=[]

        itera=os.walk(self.route)
        datasets=next(iter(itera))[1]
        for index,dataset in enumerate(datasets):
            direct=self.route+dataset+'/'
            train.append(glob(direct+'train/ai/*.PNG')+glob(direct+'train/ai/*.png'))
            test.append(glob(direct+'val/ai/*.PNG')+glob(direct+'val/ai/*.png'))
            y_train.append([dataset]*len(train[index]))
            y_test.append([dataset]*len(test[index]))
        
        nature1=glob(self.route+"BigGan/val/nature/*.JPEG")

        test.append(nature1)
        y_test.append(["real"]*len(nature1))
        print(len(nature1))
        
        nature1=glob(self.route_nature+"/train/nature/*.JPEG")
        train.append(nature1)
        print(len(nature1))
        y_train.append(["real"]*len(nature1))
            

        train = [item for sublist in train for item in sublist]
        test = [item for sublist in test for item in sublist]
        y_train = [item for sublist in y_train for item in sublist]
        y_test = [item for sublist in y_test for item in sublist]

        label_encoder=LabelEncoder().fit(y_train)
        y_train=label_encoder.transform(y_train)
        y_test=label_encoder.transform(y_test)
        print(label_encoder.classes_)



        train,val,y_train,y_val = train_test_split(train,y_train,train_size=train_size,stratify=y_train,random_state=random_state)

        return train,val,test,y_train,y_val,y_test
    def make_pairs_embeddings(self,x,embs, y):
        """Creates a tuple containing image pairs with corresponding label.

        Arguments:
            x: List containing images, each index in this list corresponds to one image.
            y: List containing labels, each label with datatype of `int`.

        Returns:
            Tuple containing two numpy arrays as (pairs_of_samples, labels),
            where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
            labels are a binary array of shape (2len(x)).
        """

        num_classes = max(y) + 1

        pairs1 = []
        pairs2=[]
        labels = []
        
        #For each sample in the set.
        for idx1 in range(len(x)):

            # add a matching example

            #Get a sample
            x1 = x[idx1]

            #Label of the sample   
            label1 = y[idx1]

            #Get randomly a sample with the same label
            x2 = embs[label1]

            #Create the pair and the label
            pairs1 += [x1]
            pairs2 += [x2]    
            labels += [0]

            #Add a non-matching sample
            label2 = random.randint(0, num_classes - 1)

            #Get a label different to the initial one
            while label2 == label1:
                label2 = random.randint(0, num_classes - 1)

            #Get a sample of the previous label calculated (different to the initial one)

            x2 = embs[label2]

            #Create the pair and the label
            pairs1 += [x1]
            pairs2 += [x2]  
            labels += [1] 

        return np.array(pairs1),np.array(pairs2), np.array(labels).astype("float32")
    

    
    
class data_set_N_with_nature():
    def __init__(self,route):
        self.route=route
        self.route_nature="Datasets/GenImageOther/BigGan"

    def get_data(self,train_size=0.9,random_state=5):
        train=[]
        test=[]
        y_train=[]
        y_test=[]
        
        # vals=['stable_diffusion_v_1_4','stable_diffusion_v_1_5','BigGan']
        
        
        # vals=['Midjourney',"VQDM","BigGan"]#entrenar solo con modelos de difussion (midjourney no se sabe que es)
            #order ['ADM' 'GLIDE' 'stable_diffusion_v_1_4' 'stable_diffusion_v_1_5' 'wukong']
            
        # vals=['GLIDE',"VQDM","stable_diffusion_v_1_5"]#entrenar con SD 1.4 y test con 1.5 (ver si son muy parecidos)
            #order ['ADM' 'BigGan' 'Midjourney' 'stable_diffusion_v_1_4' 'wukong']
            
        # vals=['ADM',"wukong","GLIDE"] #
            #order ['BigGan' 'Midjourney' 'VQDM' 'stable_diffusion_v_1_4' 'stable_diffusion_v_1_5']
        
        # vals=['wukong',"stable_diffusion_v_1_4","stable_diffusion_v_1_5"] #se dejan fuera todos los modelos que hacen uso de CLIP
            #order ['ADM' 'BigGan' 'GLIDE' 'Midjourney' 'VQDM']
        
        vals=[] #all
            #order ['ADM' 'BigGan' 'GLIDE' 'Midjourney' 'VQDM' 'real' 'stable_diffusion_v_1_4' 'stable_diffusion_v_1_5' 'wukong']



        itera=os.walk(self.route)
        datasets=next(iter(itera))[1]
        for index,dataset in enumerate(datasets):
            direct=self.route+dataset+'/'
            if dataset in vals:
                test.append(glob(direct+'val/ai/*.PNG')+glob(direct+'val/ai/*.png'))
                y_test.append([dataset]*len(test[len(test)-1]))
            else:
                
                train.append(glob(direct+'train/ai/*.PNG')+glob(direct+'train/ai/*.png'))
                y_train.append([dataset]*len(train[len(train)-1]))
        nature1=glob(self.route_nature+"/val/nature/*.JPEG")

        test.append(nature1)
        y_test.append(["real"]*len(nature1))
        
        nature1=glob(self.route_nature+"/train/nature/*.JPEG")
        train.append(nature1)
        y_train.append(["real"]*len(nature1))
            

        train = [item for sublist in train for item in sublist]
        test = [item for sublist in test for item in sublist]
        y_train = [item for sublist in y_train for item in sublist]
        y_test = [item for sublist in y_test for item in sublist]

        encoder=LabelEncoder().fit(y_train)
        y_train=encoder.transform(y_train)
        # y_test=encoder.transform(y_test)
        
        print(encoder.classes_)

        train,val,y_train,y_val = train_test_split(train,y_train,train_size=train_size,stratify=y_train,random_state=random_state)

        return train,val,test,y_train,y_val,y_test

    def make_pairs(self,x, y):
        """Creates a tuple containing image pairs with corresponding label.

        Arguments:
            x: List containing images, each index in this list corresponds to one image.
            y: List containing labels, each label with datatype of `int`.

        Returns:
            Tuple containing two numpy arrays as (pairs_of_samples, labels),
            where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
            labels are a binary array of shape (2len(x)).
        """

        num_classes = max(y) + 1
        digit_indices = [np.where(y == i)[0] for i in range(num_classes)] # 10 vectores con los índices de cada número

        pairs = []
        labels = []
        
        #For each sample in the set.
        for idx1 in range(len(x)):

            # add a matching example

            #Get a sample
            x1 = x[idx1]

            #Label of the sample   
            label1 = y[idx1]

            #Get randomly a sample with the same label
            idx2 = random.choice(digit_indices[label1])
            x2 = x[idx2]

            #Create the pair and the label
            pairs += [[x1, x2]]  
            labels += [0]

            #Add a non-matching sample
            label2 = random.randint(0, num_classes - 1)

            #Get a label different to the initial one
            while label2 == label1:
                label2 = random.randint(0, num_classes - 1)

            #Get a sample of the previous label calculated (different to the initial one)
            idx2 = random.choice(digit_indices[label2])
            x2 = x[idx2]

            #Create the pair and the label
            pairs += [[x1, x2]] 
            labels += [1] 

        return np.array(pairs), np.array(labels).astype("float32")
    
    
    def make_pairs_embeddings(self,x,embs, y):
        """Creates a tuple containing image pairs with corresponding label.

        Arguments:
            x: List containing images, each index in this list corresponds to one image.
            y: List containing labels, each label with datatype of `int`.

        Returns:
            Tuple containing two numpy arrays as (pairs_of_samples, labels),
            where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
            labels are a binary array of shape (2len(x)).
        """

        num_classes = max(y) + 1

        pairs1 = []
        pairs2=[]
        labels = []
        
        #For each sample in the set.
        for idx1 in range(len(x)):

            # add a matching example

            #Get a sample
            x1 = x[idx1]

            #Label of the sample   
            label1 = y[idx1]

            #Get randomly a sample with the same label
            x2 = embs[label1]

            #Create the pair and the label
            pairs1 += [x1]
            pairs2 += [x2]    
            labels += [0]

            #Add a non-matching sample
            label2 = random.randint(0, num_classes - 1)

            #Get a label different to the initial one
            while label2 == label1:
                label2 = random.randint(0, num_classes - 1)

            #Get a sample of the previous label calculated (different to the initial one)

            x2 = embs[label2]

            #Create the pair and the label
            pairs1 += [x1]
            pairs2 += [x2]  
            labels += [1] 

        return np.array(pairs1),np.array(pairs2), np.array(labels).astype("float32")
class data_set_binary_with_nature():
    def __init__(self,route):
        self.route=route
        self.route_nature="Datasets/GenImageOther/BigGan"

    def get_data(self,train_size=0.9,random_state=5):
        all_train=[]
        all_test=[]
        all_y_train=[]
        all_y_test=[]

        itera=os.walk(self.route)
        datasets=next(iter(itera))[1]
        print(datasets)
        for index,dataset in enumerate(datasets): 
            # 0 Real 1 sintetica
            test=[]
            train=[]
            y_train=[]
            y_test=[]
            direct=self.route+dataset+'/'

            test.append(glob(direct+'val/ai/*.PNG')+glob(direct+'val/ai/*.png'))
            y_test.append([1]*len(test[len(test)-1]))
            lista=glob(direct+'val/nature/*.JPEG')
            # print(len(lista))
            test.append(lista)
            y_test.append([0]*len(lista))

            
            train.append(glob(direct+'train/ai/*.PNG')+glob(direct+'train/ai/*.png'))
            y_train.append([1]*len(train[len(train)-1]))
            
                    
            nature1=glob(self.route_nature+"/train/nature/*.JPEG")
            train.append(nature1)
            y_train.append([0]*len(nature1))
        
        
            all_train.append(train)
            all_test.append(test)
            all_y_train.append(y_train)
            all_y_test.append(y_test)
            



        return all_train,all_test,all_y_train,all_y_test

    def make_pairs(self,x, y):
        """Creates a tuple containing image pairs with corresponding label.

        Arguments:
            x: List containing images, each index in this list corresponds to one image.
            y: List containing labels, each label with datatype of `int`.

        Returns:
            Tuple containing two numpy arrays as (pairs_of_samples, labels),
            where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
            labels are a binary array of shape (2len(x)).
        """

        num_classes = max(y) + 1
        digit_indices = [np.where(y == i)[0] for i in range(num_classes)] # 10 vectores con los índices de cada número

        pairs = []
        labels = []
        
        #For each sample in the set.
        for idx1 in range(len(x)):

            # add a matching example

            #Get a sample
            x1 = x[idx1]

            #Label of the sample   
            label1 = y[idx1]

            #Get randomly a sample with the same label
            idx2 = random.choice(digit_indices[label1])
            x2 = x[idx2]

            #Create the pair and the label
            pairs += [[x1, x2]]  
            labels += [0]

            #Add a non-matching sample
            label2 = random.randint(0, num_classes - 1)

            #Get a label different to the initial one
            while label2 == label1:
                label2 = random.randint(0, num_classes - 1)

            #Get a sample of the previous label calculated (different to the initial one)
            idx2 = random.choice(digit_indices[label2])
            x2 = x[idx2]

            #Create the pair and the label
            pairs += [[x1, x2]] 
            labels += [1] 

        return np.array(pairs), np.array(labels).astype("float32")
    
    
    def make_pairs_embeddings(self,x,embs, y):
        """Creates a tuple containing image pairs with corresponding label.

        Arguments:
            x: List containing images, each index in this list corresponds to one image.
            y: List containing labels, each label with datatype of `int`.

        Returns:
            Tuple containing two numpy arrays as (pairs_of_samples, labels),
            where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
            labels are a binary array of shape (2len(x)).
        """

        num_classes = max(y) + 1

        pairs1 = []
        pairs2=[]
        labels = []
        
        #For each sample in the set.
        for idx1 in range(len(x)):

            # add a matching example

            #Get a sample
            x1 = x[idx1]

            #Label of the sample   
            label1 = y[idx1]

            #Get randomly a sample with the same label
            x2 = embs[label1]

            #Create the pair and the label
            pairs1 += [x1]
            pairs2 += [x2]    
            labels += [0]

            #Add a non-matching sample
            label2 = random.randint(0, num_classes - 1)

            #Get a label different to the initial one
            while label2 == label1:
                label2 = random.randint(0, num_classes - 1)

            #Get a sample of the previous label calculated (different to the initial one)

            x2 = embs[label2]

            #Create the pair and the label
            pairs1 += [x1]
            pairs2 += [x2]  
            labels += [1] 

        return np.array(pairs1),np.array(pairs2), np.array(labels).astype("float32")
class data_set_binary_synth():
    def __init__(self,route="/home/jaime/Desktop/TFM/TFM/Datasets/Synth/"):
        self.route=route
        
        
    def get_data_test(self):
        all_test=[]

        all_y_test=[]

        itera=os.walk(self.route+"CNN_synth_testset/")
        datasets=next(iter(itera))[1]

        for index,dataset in enumerate(datasets): 
            # 0 Real 1 sintetica
            test=[]
            y_test=[]
            direct=self.route+dataset+'/'

            test.append(glob(direct+'1_fake/*.PNG')+glob(direct+'1_fake/*.png'))
            y_test.append([1]*len(test[len(test)-1]))
            lista=glob(direct+'0_real*.png')+glob(direct+'0_real/*.PNG')
            # print(len(lista))
            test.append(lista)
            y_test.append([0]*len(lista))

            
        
        

            all_test.append(test)
            all_y_test.append(y_test)
        return all_test,all_y_test

    def get_data(self,train_size=0.9,random_state=5):

        ruta=self.route+"Train_ProGan/"
        itera=os.walk(self.route+"Train_ProGan/")
        datasets=next(iter(itera))[1]
        print(datasets)
        test=[]
        train=[]
        y_train=[]
        y_test=[]
        for index,dataset in enumerate(datasets): 
            # 0 Real 1 sintetica

            direct=ruta+dataset+'/'
            # print(direct)

            lista=glob(direct+'1_fake/*.PNG')+glob(direct+'1_fake/*.png')
            # print(lista)
            train.append(lista)
            y_train.append([1]*len(lista))

            lista=glob(direct+'0_real/*.PNG')+glob(direct+'0_real/*.png')
            train.append(lista)
            y_train.append([0]*len(lista))
            
        train = [item for sublist in train for item in sublist]
        test = [item for sublist in test for item in sublist]
        y_train = [item for sublist in y_train for item in sublist]
        y_test = [item for sublist in y_test for item in sublist]
        
        encoder=LabelEncoder().fit(y_train)
        y_train=encoder.transform(y_train)
        y_test=encoder.transform(y_test)
        
        print(encoder.classes_)

        train,val,y_train,y_val = train_test_split(train,y_train,train_size=train_size,stratify=y_train,random_state=random_state)


        return train,val,y_train,y_val

    def make_pairs(self,x, y):
        """Creates a tuple containing image pairs with corresponding label.

        Arguments:
            x: List containing images, each index in this list corresponds to one image.
            y: List containing labels, each label with datatype of `int`.

        Returns:
            Tuple containing two numpy arrays as (pairs_of_samples, labels),
            where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
            labels are a binary array of shape (2len(x)).
        """

        num_classes = max(y) + 1
        digit_indices = [np.where(y == i)[0] for i in range(num_classes)] # 10 vectores con los índices de cada número

        pairs = []
        labels = []
        
        #For each sample in the set.
        for idx1 in range(len(x)):

            # add a matching example

            #Get a sample
            x1 = x[idx1]

            #Label of the sample   
            label1 = y[idx1]

            #Get randomly a sample with the same label
            idx2 = random.choice(digit_indices[label1])
            x2 = x[idx2]

            #Create the pair and the label
            pairs += [[x1, x2]]  
            labels += [0]

            #Add a non-matching sample
            label2 = random.randint(0, num_classes - 1)

            #Get a label different to the initial one
            while label2 == label1:
                label2 = random.randint(0, num_classes - 1)

            #Get a sample of the previous label calculated (different to the initial one)
            idx2 = random.choice(digit_indices[label2])
            x2 = x[idx2]

            #Create the pair and the label
            pairs += [[x1, x2]] 
            labels += [1] 

        return np.array(pairs), np.array(labels).astype("float32")
    
    
    def make_pairs_embeddings(self,x,embs, y):
        """Creates a tuple containing image pairs with corresponding label.

        Arguments:
            x: List containing images, each index in this list corresponds to one image.
            y: List containing labels, each label with datatype of `int`.

        Returns:
            Tuple containing two numpy arrays as (pairs_of_samples, labels),
            where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
            labels are a binary array of shape (2len(x)).
        """

        num_classes = max(y) + 1

        pairs1 = []
        pairs2=[]
        labels = []
        
        #For each sample in the set.
        for idx1 in range(len(x)):

            # add a matching example

            #Get a sample
            x1 = x[idx1]

            #Label of the sample   
            label1 = y[idx1]

            #Get randomly a sample with the same label
            x2 = embs[label1]

            #Create the pair and the label
            pairs1 += [x1]
            pairs2 += [x2]    
            labels += [0]

            #Add a non-matching sample
            label2 = random.randint(0, num_classes - 1)

            #Get a label different to the initial one
            while label2 == label1:
                label2 = random.randint(0, num_classes - 1)

            #Get a sample of the previous label calculated (different to the initial one)

            x2 = embs[label2]

            #Create the pair and the label
            pairs1 += [x1]
            pairs2 += [x2]  
            labels += [1] 

        return np.array(pairs1),np.array(pairs2), np.array(labels).astype("float32")
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
        with amp.autocast(device_type='cuda'):
            embs=model.encoder.predict_one_image(image1)
        embs=embs.detach()
        label=label.numpy()
        for ind,emb in enumerate(embs):
            dictionary[label[ind]].append(emb.cpu().numpy())

        del embs,label
        
        gc.collect()
        torch.cuda.empty_cache()
    
    last=[]
    for key in dictionary:
        embs=np.array(dictionary[key]).mean(axis=0)
        last.append(embs)
    last=np.array(last)
    np.save(path_to_save, last)
def create_and_save_ALL_embeddings(model,train_dataloader,path):
    #Get embeddings representing each data generator
    labels=[]
    embeddings=[]
    index=0
    model.eval()
    for image1, label in tqdm(train_dataloader, desc=f"Getting embedding {index + 1}/{len(train_dataloader)}"):
        index += 1
        image1 = image1.to(device)
        with amp.autocast(device_type='cuda'):
            embs=model.predict_one_image(image1)
        embs=embs.detach().cpu()
        label=label.numpy()
        embeddings+=embs
        labels.extend(label)

        del embs,label
        
        gc.collect()
        torch.cuda.empty_cache()
    labels=np.array(labels)
    embeddings=np.array(embeddings)
    np.savez(path, embeddings=embeddings, labels=labels)

    
    return embeddings,labels

def get_N_embeddings(model,train_dataloader,N):
    #Get embeddings representing each data generator
    labels=[]
    embeddings=[]
    index=0
    totals=[N for i in(range(8))]
    model.eval()
    for image1, label in tqdm(train_dataloader, desc=f"Getting embedding {index + 1}/{len(train_dataloader)}"):
        if np.sum(np.array(totals))>0:
            index += 1
            image1 = image1.to(device)
            with amp.autocast(device_type='cuda'):
                embs=model.encoder.predict_one_image(image1)
            embs=embs.detach().cpu()
            label=label.numpy()
            for i,number in enumerate(label):
                if totals[int(number)] > 0:
                    totals[int(number)]-=1
                    labels.append(int(number))
                    embeddings.append(embs[i])
        else:
            break

        del embs,label
        gc.collect()
        torch.cuda.empty_cache()

    
    return np.array(embeddings),np.array(labels)
