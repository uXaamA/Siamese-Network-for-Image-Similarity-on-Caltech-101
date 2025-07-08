import sys
import os
import shutil
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb # Import wandb

# Ensure the path to your custom modules is correct for Kaggle
# This should point to the directory containing model.py and data_util.py
MODULE_BASE_PATH = '/kaggle/input/dl-assignment-again/pytorch/default/1/'
if MODULE_BASE_PATH not in sys.path:
    sys.path.append(MODULE_BASE_PATH)

try:
    from model import CaltechNetwork, contrastive_loss, custom_confusion_matrix, compute_metrics
    from data_util import get_dataloaders
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print(f"Please ensure model.py and data_util.py are in {MODULE_BASE_PATH} or sys.path is correctly set.")
    sys.exit(1)

def train_model(dataset_path_source,
                project_name="caltech-siamese", # wandb project name
                run_name=None,                 # wandb run name (optional, can be auto-generated)
                epochs=20,
                lr=0.001,
                batch_size=32,
                threshold=0.5,
                embedding_size=128, # Added for logging
                num_workers_loader=2, # Added for logging
                use_local_copy=False): # Usually False for Kaggle /kaggle/input

    # --- wandb Initialization ---
    # Set WANDB_API_KEY as a secret in Kaggle if running non-interactively
    try:
        wandb.login() # Will prompt if not logged in, or use WANDB_API_KEY
    except Exception as e:
        print(f"wandb login failed, attempting to proceed without interactive login: {e}")
        pass # Can still work if API key is set via environment variable

    config_defaults = {
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "threshold": threshold,
        "embedding_size": embedding_size,
        "optimizer": "Adam",
        "lr_scheduler": "ReduceLROnPlateau",
        "dataset": os.path.basename(dataset_path_source) if dataset_path_source else "caltech-101",
        "num_workers_loader": num_workers_loader,
    }

    wandb.init(
        project=project_name,
        name=run_name, # If None, wandb will generate a name
        config=config_defaults
    )
    # Log the exact config used by wandb
    config = wandb.config
    # --- End wandb Initialization ---

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    wandb.log({"device": str(device)})


    dataset_path_to_use = dataset_path_source
    if use_local_copy and not dataset_path_source.startswith('/kaggle/input/'):
        local_dataset_path = '/kaggle/working/caltech-101-local'
        print(f"Attempting to use a local copy of the dataset from {dataset_path_source} at {local_dataset_path}")
        if not os.path.exists(local_dataset_path) or not os.listdir(local_dataset_path):
            if os.path.exists(local_dataset_path):
                shutil.rmtree(local_dataset_path)
            print(f"Copying dataset from {dataset_path_source} to {local_dataset_path}...")
            shutil.copytree(dataset_path_source, local_dataset_path)
            print("Dataset copied locally.")
        else:
            print(f"Local dataset already exists at {local_dataset_path}. Using it.")
        dataset_path_to_use = local_dataset_path
    elif dataset_path_source.startswith('/kaggle/input/'):
        print(f"Using dataset directly from Kaggle input: {dataset_path_source}")

    weights_dir = '/kaggle/working/weights' # Save weights to Kaggle's writable directory
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        print(f"Created '{weights_dir}' directory.")

    print("Initializing Dataloaders...")
    try:
        train_loader, val_loader, test_loader_unused = get_dataloaders(
            dataset_path_to_use,
            config.batch_size, # Use batch_size from wandb.config
            num_workers_val=config.num_workers_loader # Use num_workers from wandb.config
        )
        print("Dataloaders initialized.")
        print(f"Train loader: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
        print(f"Val loader: {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    except Exception as e:
        print(f"Error during Dataloader initialization: {e}")
        wandb.log({"error_dataloader_init": str(e)})
        wandb.finish(exit_code=1)
        return

    model = CaltechNetwork(embedding_size=config.embedding_size).to(device) # Use embedding_size from wandb.config
    optimizer = Adam(model.parameters(), lr=config.learning_rate) # Use lr from wandb.config
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)

    # --- wandb: Watch model gradients and parameters (optional) ---
    wandb.watch(model, log="gradients", log_freq=100) # log gradients every 100 batches

    best_val_loss = float('inf')
    best_val_accuracy = 0.0 # For saving based on accuracy too, if desired

    print(f"\nStarting training for {config.epochs} epochs...")
    for epoch in range(config.epochs):
        model.train()
        train_loss_sum = 0.0
        train_tp, train_tn, train_fp, train_fn = 0, 0, 0, 0

        print(f"\n--- Epoch {epoch+1}/{config.epochs} ---")
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

            train_loss_sum += loss.item() * img1.size(0) # Accumulate sum of losses for correct averaging

            with torch.no_grad(): # Calculate metrics without affecting gradients
                distance = torch.nn.functional.pairwise_distance(emb1, emb2)
                predictions = (distance < config.threshold).float() # Use threshold from wandb.config
                tp, tn, fp, fn = custom_confusion_matrix(labels.cpu(), predictions.cpu())
                train_tp += tp.item(); train_tn += tn.item(); train_fp += fp.item(); train_fn += fn.item()

        # Calculate average train metrics for the epoch
        num_train_samples = len(train_loader.dataset)
        avg_train_loss = train_loss_sum / num_train_samples if num_train_samples > 0 else 0
        train_accuracy, train_precision, train_recall, train_f1 = compute_metrics(train_tp, train_tn, train_fp, train_fn)

        # Validation phase
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

                v_loss = contrastive_loss(emb1, emb2, labels)
                val_loss_sum += v_loss.item() * img1.size(0)

                distance = torch.nn.functional.pairwise_distance(emb1, emb2)
                predictions = (distance < config.threshold).float()
                tp, tn, fp, fn = custom_confusion_matrix(labels.cpu(), predictions.cpu())
                val_tp += tp.item(); val_tn += tn.item(); val_fp += fp.item(); val_fn += fn.item()

        # Calculate average validation metrics for the epoch
        num_val_samples = len(val_loader.dataset)
        avg_val_loss = val_loss_sum / num_val_samples if num_val_samples > 0 else 0
        val_accuracy, val_precision, val_recall, val_f1 = compute_metrics(val_tp, val_tn, val_fp, val_fn)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Acc: {train_accuracy:.4f} | Prec: {train_precision:.4f} | Rec: {train_recall:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   Loss: {avg_val_loss:.4f} | Acc: {val_accuracy:.4f} | Prec: {val_precision:.4f} | Rec: {val_recall:.4f} | F1: {val_f1:.4f}")
        print(f"  Current LR: {current_lr:.6f}")

        # --- wandb Logging ---
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_f1": train_f1,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "learning_rate": current_lr
        })
        # --- End wandb Logging ---

        if avg_val_loss < best_val_loss:
            print(f"  Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}.")
            best_val_loss = avg_val_loss
            wandb.run.summary["best_val_loss"] = best_val_loss # Update summary for best
            
            model_save_path = os.path.join(weights_dir, 'best_model_val_loss.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"  Saved best model (by val_loss) to '{model_save_path}'")
            # --- wandb: Save best model artifact based on loss ---
            # You can also save with wandb.save(model_save_path) for simple file saving
            # Using Artifacts is more robust for versioning
            best_model_artifact_loss = wandb.Artifact(
                f"{wandb.run.name}-best-loss-model", type="model",
                description="Siamese model with the best validation loss.",
                metadata={"epoch": epoch + 1, "val_loss": avg_val_loss, "val_accuracy": val_accuracy}
            )
            best_model_artifact_loss.add_file(model_save_path)
            wandb.log_artifact(best_model_artifact_loss, aliases=['best_loss', f'epoch_{epoch+1}_loss'])
            # --- End wandb Model Artifact ---

        if val_accuracy > best_val_accuracy:
            print(f"  Validation accuracy improved from {best_val_accuracy:.4f} to {val_accuracy:.4f}.")
            best_val_accuracy = val_accuracy
            wandb.run.summary["best_val_accuracy"] = best_val_accuracy

            model_save_path_acc = os.path.join(weights_dir, 'best_model_val_acc.pth')
            torch.save(model.state_dict(), model_save_path_acc)
            print(f"  Saved best model (by val_accuracy) to '{model_save_path_acc}'")
            best_model_artifact_acc = wandb.Artifact(
                f"{wandb.run.name}-best-acc-model", type="model",
                description="Siamese model with the best validation accuracy.",
                metadata={"epoch": epoch + 1, "val_loss": avg_val_loss, "val_accuracy": val_accuracy}
            )
            best_model_artifact_acc.add_file(model_save_path_acc)
            wandb.log_artifact(best_model_artifact_acc, aliases=['best_accuracy', f'epoch_{epoch+1}_acc'])


    print("\nTraining finished.")
    # --- wandb Finish ---
    wandb.finish()
    # --- End wandb Finish ---

if __name__ == '__main__':
    # Kaggle input path for the dataset
    # Make sure this path points to the 'caltech-101' FOLDER, not a file inside it.
    original_dataset_path = os.path.join(MODULE_BASE_PATH, 'caltech-101/caltech-101')

    if not os.path.exists(original_dataset_path) or not os.path.isdir(original_dataset_path):
        print(f"ERROR: Dataset directory not found or is not a directory: {original_dataset_path}")
        print("Please ensure the path points to the root 'caltech-101' folder containing class subdirectories.")
    else:
        print(f"Found dataset directory: {original_dataset_path}")
        train_model(
            dataset_path_source=original_dataset_path,
            project_name="caltech101-siamese-contrastive", # Customize your W&B project name
            # run_name="resnet50-128emb-run1", # Optional: give a specific name to this run
            epochs=20,          # Number of epochs
            lr=0.0005,          # Learning rate
            batch_size=32,      # Batch size
            threshold=0.7,      # Distance threshold for accuracy calculation
            embedding_size=128, # Dimension of the output embedding
            num_workers_loader=2, # Number of workers for DataLoader
            use_local_copy=False # Set to False for Kaggle /kaggle/input
        )