import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self,out1, out2, labels):
        """
        Contrastive loss function.
        
        Args:
        - out1: Batch of the the embedding of the first images in the pairs (Batch_size, Embedding_size).
        - out2: Batch of the embeddings of the second images in the pairs (Batch_size, Embedding_size).
        - labels: A tensor of labels (0 for related, 1 for unrelated) (Batch_size,).
        - margin: The margin that separates positive and negative pairs.
        
        Returns:
        - The contrastive loss.
        """

        # Compute the Euclidean distance between the embeddings
        euclidean_distance = F.pairwise_distance(out1, out2, p=2)
        
        # Calculate the contrastive loss
        loss = torch.mean((1 - labels) * torch.pow(euclidean_distance, 2) + 
                        (labels) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        return loss
    
class Contrastive_loss(nn.Module):
    """Provides 'contrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'contrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(1-prediction, 0) ))
    def __init__(self,margin=1):
        super(Contrastive_loss,self).__init__()
        self.margin=margin

        
    def forward(self,y_true, y_pred):
        """Calculates the contrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing contrastive loss as floating point value.
        """

        # square_pred = torch.square(y_pred)

        # # Calculate the margin squared term
        # margin_square = torch.square(torch.maximum(self.margin - y_pred, torch.tensor(0.0)))

        # # Contrastive loss formula
        # loss = torch.mean((1 - y_true) * square_pred + y_true * margin_square)

        # return loss
        loss = torch.mean((1 - y_true) * torch.square(y_pred) + y_true * torch.square(torch.maximum(self.margin - y_pred, torch.zeros_like(y_pred))))
        return loss
class CLIP_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(emb_im1, emb_im2,temperature=1.0):
        # logits[i][j] is the dot_similarity(caption_i, image_j).
        logits = torch.matmul(emb_im1, emb_im2.T) / temperature

        # images_similarity[i][j] is the dot_similarity(image_i, image_j).
        images_similarity = torch.matmul(emb_im2, emb_im2.T)

        # captions_similarity[i][j] is the dot_similarity(caption_i, caption_j).
        captions_similarity = torch.matmul(emb_im1, emb_im1.T)

        # targets[i][j] = average dot_similarity(caption_i, caption_j) and dot_similarity(image_i, image_j).
        targets = F.softmax((captions_similarity + images_similarity) / (2 * temperature), dim=-1)

        # Compute the loss for the captions using crossentropy
        captions_loss = F.cross_entropy(logits, targets.argmax(dim=-1), reduction='mean')

        # Compute the loss for the images using crossentropy
        images_loss = F.cross_entropy(logits.T, targets.argmax(dim=-1), reduction='mean')

        # Return the mean of the loss over the batch.
        return (captions_loss + images_loss) / 2
    
class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, t_0=0.07, eps=1e-8):
        super(SupConLoss, self).__init__()
        self.temperature = torch.nn.Parameter(torch.tensor([t_0]))
        self.epsilon = eps


    def forward(self, features, labels):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        batch_size = features.shape[0]

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
            'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(features.device)

        views = features.shape[1] # = n_views
        full_features = torch.cat(torch.unbind(features, dim=1), dim=0).to(features.device) # = [bsz*views, ...]

        # compute logits (cosine sim)
        anchor_dot_contrast = torch.matmul(F.normalize(full_features),
                                           F.normalize(full_features.T)) * torch.exp(self.temperature.to(features.device)).clamp(100) # = [bsz*views, bsz*views]

        loss = self._loss_from_dot(anchor_dot_contrast, mask, views, batch_size)

        return loss
    def _loss_from_dot(self, anchor_dot_contrast, mask, views, batch_size): #(anchor, contrast)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(views, views)
        # mask-out self-contrast cases
        logits_mask = 1 - torch.eye(views*batch_size, device=mask.device)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + self.epsilon)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos.view(views, batch_size).mean()

        return loss