from torch import nn
import torch
import gc

from efficientnet_pytorch import EfficientNet

class siamese_model(nn.Module):
    def __init__(self,type,device):
        super().__init__()
        self.device=device
        self.type=type
        self.efficientNet = EfficientNet.from_name(type,3)  # Initialize from scratch
        # Modify the classifier (output layer) of EfficientNet

        self.efficientNet._fc = nn.Linear(1280,512)
        self.norm=nn.BatchNorm1d(512)

        # self.linear=nn.Linear(1280*2,2)
        # self.softmax=nn.Softmax(dim=1)
        

    def euclidean_distance(self,vect1,vect2):
        """Find the Euclidean distance between two vectors.

        Arguments:
            vects: List containing two tensors of same length.

        Returns:
            Tensor containing euclidean distance
            (as floating point value) between vectors.
        """

        sum_square = torch.sum(torch.square(vect1 - vect2), dim=1, keepdim=True)
        
        # Return the square root of the sum of squares with numerical stability
        epsilon = torch.tensor(torch.finfo(vect1.dtype).eps)  # Small value to avoid sqrt(0)
        return torch.sqrt(torch.maximum(sum_square, epsilon))

    def forward(self,im1,im2):

        emb1=self.efficientNet(im1)
        emb2=self.efficientNet(im2)
        del im2, im1
        gc.collect()
        torch.cuda.empty_cache()

        emb1=self.norm(emb1)
        emb2=self.norm(emb2)

        return emb1,emb2

class Big_model(nn.Module):
    def __init__(self,path_to_encoder,device,embedding_size):
        super().__init__()
        self.path_to_encoder=path_to_encoder
        self.device=device
        self.embedding_size=embedding_size

        checkpoint=torch.load(self.path_to_encoder)
        self.encoder_type=checkpoint["model_type"]
        self.encoder=siamese_model(self.encoder_type,self.device)
        self.encoder.load_state_dict(checkpoint["model_state_dict"])

        self.classification=nn.Linear(self.embedding_size*2,self.embedding_size)
        self.act=nn.GELU()
        self.last=nn.Linear(self.embedding_size,2)
    def forward(self,im1,im2):
        emb1,emb2=self.encoder(im1,im2)
        del im2, im1
        gc.collect()
        torch.cuda.empty_cache()
        embs=torch.concat([emb1,emb2],dim=1)
        out=self.classification(embs)
        out=self.act(out)
        return (self.last(out))