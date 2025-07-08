# # --- START OF MODIFIED train.py ---
# from google.colab import drive
# drive.mount ('/content/drive') # Mount GDrive if you need to copy from it

import sys
import os
import shutil # For copying data
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Ensure the path to your custom modules is correct
# Adjust this path if your notebook structure is different
module_path = '/content/drive/MyDrive/Colab Notebooks/Deep Learning Assignment/Assignment 02/'
if module_path not in sys.path:
    sys.path.append(module_path)

from model import CaltechNetwork, contrastive_loss, custom_confusion_matrix, compute_metrics
from data_utils import get_dataloaders

def train_model(dataset_path_source, epochs=20, lr=0.001, batch_size=32, threshold=0.5, use_local_copy=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_path_to_use = dataset_path_source
    if use_local_copy and dataset_path_source.startswith('/content/drive'):
        local_dataset_path = '/content/caltech-101-local'
        print(f"Attempting to use a local copy of the dataset from {dataset_path_source} at {local_dataset_path}")
        if not os.path.exists(local_dataset_path) or not os.listdir(local_dataset_path): # Check if empty
            if os.path.exists(local_dataset_path): # remove if empty or partially copied
                print(f"Removing existing incomplete local copy at {local_dataset_path}")
                shutil.rmtree(local_dataset_path)
            print(f"Copying dataset from {dataset_path_source} to {local_dataset_path}...")
            shutil.copytree(dataset_path_source, local_dataset_path)
            print("Dataset copied locally.")
        else:
            print(f"Local dataset already exists at {local_dataset_path}. Using it.")
        dataset_path_to_use = local_dataset_path
    
    if not os.path.exists('weights'):
        os.makedirs('weights')
        print("Created 'weights' directory.")

    print("Initializing Dataloaders...")
    try:
        # Consider num_workers=0 for initial debugging if issues persist
        train_loader, val_loader, _ = get_dataloaders(dataset_path_to_use, batch_size, num_workers_val=2) # you can adjust num_workers
        print("Dataloaders initialized.")
        
        print("Attempting to get one batch from train_loader to verify...")
        img1_sample, _, labels_sample = next(iter(train_loader))
        print(f"Successfully got one batch. Image1 shape: {img1_sample.shape}, Labels shape: {labels_sample.shape}")
    except Exception as e:
        print(f"Error during Dataloader initialization or first batch fetch: {e}")
        print("This could be due to issues in CaltechDataset (e.g., not enough classes for pairing) or file access problems.")
        return

    model = CaltechNetwork().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)
    
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss_sum = 0.0
        train_tp, train_tn, train_fp, train_fn = 0, 0, 0, 0
        
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        print("Training Phase:")
        for batch_idx, (img1, img2, labels) in enumerate(train_loader):
            if batch_idx % 20 == 0: # Print progress
                 print(f"  Train Batch {batch_idx+1}/{len(train_loader)}")

            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            emb1, emb2 = model(img1, img2)
            loss = contrastive_loss(emb1, emb2, labels)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item() * img1.size(0)

            with torch.no_grad():
                distance = torch.nn.functional.pairwise_distance(emb1, emb2)
                # Label 1 = similar (small distance desired), 0 = dissimilar (large distance desired)
                # Prediction = 1 (similar) if distance < threshold
                predictions = (distance < threshold).float()
                
                tp, tn, fp, fn = custom_confusion_matrix(labels.cpu(), predictions.cpu())
                train_tp += tp.item(); train_tn += tn.item(); train_fp += fp.item(); train_fn += fn.item()

        avg_train_loss = train_loss_sum / len(train_loader.dataset)
        train_accuracy, train_precision, train_recall, train_f1 = compute_metrics(train_tp, train_tn, train_fp, train_fn)

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_tp, val_tn, val_fp, val_fn = 0, 0, 0, 0
        
        print("Validation Phase:")
        with torch.no_grad():
            for batch_idx, (img1, img2, labels) in enumerate(val_loader):
                if batch_idx % 20 == 0: # Print progress
                    print(f"  Validation Batch {batch_idx+1}/{len(val_loader)}")

                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                emb1, emb2 = model(img1, img2)
                
                loss = contrastive_loss(emb1, emb2, labels)
                val_loss_sum += loss.item() * img1.size(0)
                
                distance = torch.nn.functional.pairwise_distance(emb1, emb2)
                predictions = (distance < threshold).float() # Corrected prediction logic
                tp, tn, fp, fn = custom_confusion_matrix(labels.cpu(), predictions.cpu())
                val_tp += tp.item(); val_tn += tn.item(); val_fp += fp.item(); val_fn += fn.item()

        avg_val_loss = val_loss_sum / len(val_loader.dataset)
        val_accuracy, val_precision, val_recall, val_f1 = compute_metrics(val_tp, val_tn, val_fp, val_fn)
        
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Acc: {train_accuracy:.4f} | Prec: {train_precision:.4f} | Rec: {train_recall:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   Loss: {avg_val_loss:.4f} | Acc: {val_accuracy:.4f} | Prec: {val_precision:.4f} | Rec: {val_recall:.4f} | F1: {val_f1:.4f}")

        if avg_val_loss < best_val_loss:
            print(f"  Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model to 'weights/best_model.pth'")
            torch.save(model.state_dict(), 'weights/best_model.pth')
            best_val_loss = avg_val_loss
    
    print("\nTraining finished.")

if __name__ == '__main__':
    gdrive_root_path = '/content/drive/MyDrive/Colab Notebooks/Deep Learning Assignment/Assignment 02/'
    # Original dataset path on GDrive
    original_dataset_path = os.path.join(gdrive_root_path, 'caltech-101') 

    # If you are on Kaggle, the path would be different, e.g., '/kaggle/input/your-caltech101-dataset/'
    # For Kaggle, you might set use_local_copy=False if data is already fast in /kaggle/input/
    
    # Check if the dataset path exists before training
    if not os.path.exists(original_dataset_path):
        print(f"ERROR: Dataset not found at {original_dataset_path}. Please check the path.")
    else:
        train_model(original_dataset_path, epochs=20, lr=0.001, batch_size=32, use_local_copy=True)
# --- END OF MODIFIED train.py ---