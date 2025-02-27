from torch import nn
import torch
import gc
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class siamese_model(nn.Module):
    def __init__(self,type,device,embedding_size):
        super().__init__()
        self.device=device
        self.type=type
        self.embedding_size=embedding_size
        self.efficientNet = EfficientNet.from_pretrained(type,3)  # Initialize from scratch
        # Modify the classifier (output layer) of EfficientNet

        self.efficientNet._fc = nn.Linear(1280,self.embedding_size)

    def forward(self,im1,im2):

        emb1=self.efficientNet(im1)
        emb2=self.efficientNet(im2)

        del im2, im1
        gc.collect()
        torch.cuda.empty_cache()

        return emb1,emb2
    
    def predict_one_image(self,im1):

        emb1=self.efficientNet(im1)
        del im1
        gc.collect()
        torch.cuda.empty_cache()
        return emb1
        

class Big_model(nn.Module):
    def __init__(self,path_to_encoder,device,embedding_size):
        super().__init__()
        self.path_to_encoder=path_to_encoder
        self.device=device
        self.embedding_size=embedding_size

        checkpoint=torch.load(self.path_to_encoder)
        self.encoder_type=checkpoint["model_type"]
        self.encoder=siamese_model(self.encoder_type,self.device,embedding_size=self.embedding_size)
        self.encoder.load_state_dict(checkpoint["model_state_dict"])

        self.classification=nn.Linear(self.embedding_size*2,self.embedding_size)
        self.act=nn.GELU()
        self.last=nn.Linear(self.embedding_size,2)
        for param in self.encoder.parameters():
            param.requires_grad = False
    def forward(self,im1,im2):
        emb1,im2=self.encoder(im1,im2)
        del im1
        gc.collect()
        torch.cuda.empty_cache()
        embs=torch.concat([emb1,im2],dim=1)
        out=self.classification(embs)
        out=self.act(out)
        return (self.last(out))
    
    def classify_embeddings(self,emb1,emb2):
        embs=torch.concat([emb1,emb2],dim=1)
        out=self.classification(embs)
        out=self.act(out)
        out=self.last(out)
        return out

        