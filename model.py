# --- START OF MODIFIED model.py ---
import torch
import torch.nn as nn
import torchvision.models as models

class CaltechNetwork(nn.Module):
  def __init__(self, embedding_size=128):
    super().__init__()
    # Use the 'weights' parameter for pretrained models
    self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    self.backbone.fc = nn.Identity()                           

    self.fc = nn.Sequential(                                     
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Linear(512, embedding_size)
    )

  def forward(self, x1, x2):
    emb1 = self.backbone(x1)
    emb2 = self.backbone(x2)
    emb1 = self.fc(emb1)
    emb2 = self.fc(emb2)
    return emb1, emb2

def contrastive_loss(emb1, emb2, label, margin=1.0):
  distance = torch.nn.functional.pairwise_distance(emb1, emb2)
  # Ensure label is float for multiplication if it isn't already
  label = label.float() 
  loss = torch.mean(
      (label) * torch.pow(distance, 2) +
      (1 - label) * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
  )
  return loss

def compute_metrics(tp, tn, fp, fn):
    # Ensure inputs are floats for division to avoid integer division if they are tensors
    tp, tn, fp, fn = float(tp), float(tn), float(fp), float(fn)
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8) 
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return accuracy, precision, recall, f1

def custom_confusion_matrix(y_true, y_pred):
    # y_true, y_pred are expected to be 1D tensors on CPU
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum() 
    fn = ((y_pred == 0) & (y_true == 1)).sum() 
    return tp, tn, fp, fn
# --- END OF MODIFIED model.py ---