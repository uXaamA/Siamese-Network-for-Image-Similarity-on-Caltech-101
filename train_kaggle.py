# --- START OF MODIFIED train.py (for Kaggle) ---
# from google.colab import drive # Not needed for Kaggle
# drive.mount ('/content/drive') # Not needed for Kaggle

import sys
import os
import shutil # For copying data (though use_local_copy will be False)
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Ensure the path to your custom modules is correct for Kaggle
module_path = '/kaggle/input/dl-assignment-again/pytorch/default/1/'
if module_path not in sys.path:
    sys.path.append(module_path)

# Import from your custom modules in the Kaggle input directory
from model import CaltechNetwork, contrastive_loss, custom_confusion_matrix, compute_metrics
from data_util import get_dataloaders # Adjusted to data_util.py

def train_model(dataset_path_source, epochs=20, lr=0.001, batch_size=32, threshold=0.5, use_local_copy=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_path_to_use = dataset_path_source
    if use_local_copy and not dataset_path_source.startswith('/kaggle/input/'): # Only copy if not already in /kaggle/input
        # This block is less likely to be used on Kaggle if dataset_path_source is from /kaggle/input/
        local_dataset_path = '/kaggle/working/caltech-101-local' # Kaggle's writable directory
        print(f"Attempting to use a local copy of the dataset from {dataset_path_source} at {local_dataset_path}")
        if not os.path.exists(local_dataset_path) or not os.listdir(local_dataset_path): # Check if empty
            if os.path.exists(local_dataset_path):
                print(f"Removing existing incomplete local copy at {local_dataset_path}")
                shutil.rmtree(local_dataset_path)
            print(f"Copying dataset from {dataset_path_source} to {local_dataset_path}...")
            shutil.copytree(dataset_path_source, local_dataset_path)
            print("Dataset copied locally.")
        else:
            print(f"Local dataset already exists at {local_dataset_path}. Using it.")
        dataset_path_to_use = local_dataset_path
    elif dataset_path_source.startswith('/kaggle/input/'):
        print(f"Using dataset directly from Kaggle input: {dataset_path_source}")
        dataset_path_to_use = dataset_path_source # Ensure this is set correctly
    
    # Save weights to Kaggle's writable directory
    weights_dir = '/kaggle/working/weights'
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        print(f"Created '{weights_dir}' directory.")

    print("Initializing Dataloaders...")
    try:
        # Using num_workers=2 as a default, adjust if needed for Kaggle's environment
        train_loader, val_loader, _ = get_dataloaders(dataset_path_to_use, batch_size, num_workers_val=2)
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
        model.train()
        train_loss_sum = 0.0
        train_tp, train_tn, train_fp, train_fn = 0, 0, 0, 0
        
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        print("Training Phase:")
        for batch_idx, (img1, img2, labels) in enumerate(train_loader):
            if batch_idx % 20 == 0:
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
                predictions = (distance < threshold).float()
                tp, tn, fp, fn = custom_confusion_matrix(labels.cpu(), predictions.cpu())
                train_tp += tp.item(); train_tn += tn.item(); train_fp += fp.item(); train_fn += fn.item()

        avg_train_loss = train_loss_sum / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0
        train_accuracy, train_precision, train_recall, train_f1 = compute_metrics(train_tp, train_tn, train_fp, train_fn)

        model.eval()
        val_loss_sum = 0.0
        val_tp, val_tn, val_fp, val_fn = 0, 0, 0, 0
        
        print("Validation Phase:")
        with torch.no_grad():
            for batch_idx, (img1, img2, labels) in enumerate(val_loader):
                if batch_idx % 20 == 0:
                    print(f"  Validation Batch {batch_idx+1}/{len(val_loader)}")

                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                emb1, emb2 = model(img1, img2)
                
                loss = contrastive_loss(emb1, emb2, labels)
                val_loss_sum += loss.item() * img1.size(0)
                
                distance = torch.nn.functional.pairwise_distance(emb1, emb2)
                predictions = (distance < threshold).float()
                tp, tn, fp, fn = custom_confusion_matrix(labels.cpu(), predictions.cpu())
                val_tp += tp.item(); val_tn += tn.item(); val_fp += fp.item(); val_fn += fn.item()

        avg_val_loss = val_loss_sum / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
        val_accuracy, val_precision, val_recall, val_f1 = compute_metrics(val_tp, val_tn, val_fp, val_fn)
        
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Acc: {train_accuracy:.4f} | Prec: {train_precision:.4f} | Rec: {train_recall:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   Loss: {avg_val_loss:.4f} | Acc: {val_accuracy:.4f} | Prec: {val_precision:.4f} | Rec: {val_recall:.4f} | F1: {val_f1:.4f}")

        if avg_val_loss < best_val_loss:
            model_save_path = os.path.join(weights_dir, 'best_model.pth')
            print(f"  Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model to '{model_save_path}'")
            torch.save(model.state_dict(), model_save_path)
            best_val_loss = avg_val_loss
    
    print("\nTraining finished.")

if __name__ == '__main__':
    # Kaggle input path for the dataset
    original_dataset_path = '/kaggle/input/dl-assignment-again/pytorch/default/1/caltech-101/caltech-101'

    # Check if the dataset path exists before training
    if not os.path.exists(original_dataset_path):
        print(f"ERROR: Dataset not found at {original_dataset_path}. Please check the path.")
    else:
        # For Kaggle, data in /kaggle/input/ is already fast, so use_local_copy=False is recommended.
        train_model(original_dataset_path, epochs=20, lr=0.001, batch_size=32, use_local_copy=False)
# --- END OF MODIFIED train.py (for Kaggle) ---