# --- START OF MODIFIED data_utils.py ---
import os
import pandas as pd
import random
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class CaltechDataset(Dataset):
    def __init__(self, root_dir, dataframe, transform=None):
        self.root_dir = root_dir
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

        self.class_to_indices = {}
        valid_classes_for_pairing = 0
        for class_name in self.dataframe['label'].unique():
            class_indices = self.dataframe[self.dataframe['label'] == class_name].index.tolist()
            if len(class_indices) >= 2:
                self.class_to_indices[class_name] = class_indices
                valid_classes_for_pairing += 1
        
        if valid_classes_for_pairing < 2 and len(self.dataframe) > 0 : # Need at least 2 distinct classes for negative pairs
            print(f"WARNING: CaltechDataset initialized with only {valid_classes_for_pairing} class(es) having >= 2 samples.")
            print(f"  This will cause issues with negative pair generation if {valid_classes_for_pairing} < 2.")
            # print(f"  Dataframe labels: {self.dataframe['label'].unique()}") # For more detailed debug

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        anchor_row = self.dataframe.iloc[idx]
        try:
            anchor_image = Image.open(os.path.join(self.root_dir, anchor_row['path'])).convert('RGB')
        except FileNotFoundError:
            print(f"ERROR: Anchor image not found at {os.path.join(self.root_dir, anchor_row['path'])}")
            # Fallback: get another random item. Could lead to issues if many files are missing.
            return self.__getitem__(random.randint(0, len(self) - 1))
            
        anchor_label = anchor_row['label']

        if anchor_label not in self.class_to_indices:
            # This means the anchor's class had < 2 images, shouldn't happen if dataframe is pre-filtered correctly by split_data
            # or if class_to_indices was built correctly.
            # print(f"Warning: Anchor label {anchor_label} (idx {idx}) not in class_to_indices. Picking random item.")
            return self.__getitem__(random.randint(0, len(self) - 1))

        is_positive_pair = random.random() < 0.5

        if is_positive_pair:
            pos_indices = [i for i in self.class_to_indices[anchor_label] if i != idx]
            if not pos_indices: # Should only happen if class has 1 image, but class_to_indices requires >=2
                                # Or if this is the only image of its class in the current dataframe slice
                # Fallback: try to make it a negative pair, if possible, or recurse
                # print(f"Warning: No positive pair for {anchor_label} (idx {idx}). Trying negative or recursing.")
                is_positive_pair = False # Force attempt at negative
            else:
                other_idx = random.choice(pos_indices)
                label = 1.0
        
        if not is_positive_pair: # Try to make a negative pair
            neg_classes = [cls for cls in self.class_to_indices if cls != anchor_label]
            if not neg_classes:
                # CRITICAL: No other classes to form a negative pair. This causes infinite recursion if not handled.
                # This happens if the dataset (e.g., train_df) has only one unique class in self.class_to_indices.
                # Fallback to positive pair if possible, otherwise recurse on a different item.
                # print(f"CRITICAL WARNING: No negative classes for {anchor_label} (idx {idx}). Trying positive fallback.")
                pos_indices = [i for i in self.class_to_indices[anchor_label] if i != idx]
                if not pos_indices : # Still can't make a positive pair (e.g. anchor is only img of its class)
                    # This is a deep issue with data integrity for this item. Recurse on a different item.
                    # print(f"  Cannot make positive fallback either for {anchor_label}. Recursing on entirely different item.")
                    return self.__getitem__(random.randint(0, len(self) - 1))
                other_idx = random.choice(pos_indices)
                label = 1.0 # Forced positive
            else:
                neg_cls = random.choice(neg_classes)
                other_idx = random.choice(self.class_to_indices[neg_cls])
                label = 0.0
        
        try:
            other_image = Image.open(os.path.join(self.root_dir, self.dataframe.iloc[other_idx]['path'])).convert('RGB')
        except FileNotFoundError:
            print(f"ERROR: Other image not found at {os.path.join(self.root_dir, self.dataframe.iloc[other_idx]['path'])}")
            return self.__getitem__(random.randint(0, len(self) - 1))

        if self.transform:
            anchor_image = self.transform(anchor_image)
            other_image = self.transform(other_image)

        return anchor_image, other_image, torch.tensor(label, dtype=torch.float32)


def split_data (root_dir, test_size = 0.2, val_size = 0.1):
    data = []
    print(f"Scanning dataset at: {root_dir}")
    if not os.path.exists(root_dir) or not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory {root_dir} not found or is not a directory.")

    class_counts = {}
    for cls in os.listdir(root_dir):
        cls_dir = os.path.join(root_dir, cls)
        if os.path.isdir(cls_dir):
            images = [img for img in os.listdir(cls_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))] # Basic image filter
            class_counts[cls] = len(images)
            if len(images) >= 10: # Filter: class must have at least 10 images
                # print(f"  Class '{cls}' has {len(images)} images (>=10). Including.")
                for img_name in images:
                    data.append({'path': os.path.join(cls, img_name), 'label': cls})
            # else:
            #     print(f"  Class '{cls}' has {len(images)} images (<10). Skipping.")
    
    if not data:
        print("Available class counts:", class_counts)
        raise ValueError("No classes with >= 10 images found. Check dataset structure, image extensions, or filter criteria.")
    
    df = pd.DataFrame(data)
    print(f"Total images from eligible classes (>=10 images per class): {len(df)}")
    num_eligible_classes = df['label'].nunique()
    print(f"Number of unique eligible classes: {num_eligible_classes}")

    if num_eligible_classes < 2:
        raise ValueError(f"Dataset has only {num_eligible_classes} unique class(es) with >=10 images. Need at least 2 for contrastive learning with negative pairs.")

    # Stratified splitting (original logic was fine)
    # train_df is (1-test_size) of original. val_df is val_size of that train_df.
    train_val_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)
    # The val_size here is a fraction of train_val_df, which is correct.
    train_df, val_df = train_test_split(train_val_df, test_size=val_size/(1-test_size) if (1-test_size)>0 else val_size, stratify=train_val_df['label'], random_state=42)


    print(f"Train set: {len(train_df)} images, {train_df['label'].nunique()} classes.")
    if train_df['label'].nunique() < 2 :
        print("WARNING: Training set has < 2 unique classes after split. This will cause issues for negative sampling.")
    print(f"Validation set: {len(val_df)} images, {val_df['label'].nunique()} classes.")
    if val_df['label'].nunique() < 2 :
        print("WARNING: Validation set has < 2 unique classes after split. This will cause issues for negative sampling.")
    print(f"Test set: {len(test_df)} images, {test_df['label'].nunique()} classes.")
          
    return train_df, val_df, test_df

def get_dataloaders (root_dir, batch_size = 32, num_workers_val=4): # Added num_workers_val
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    train_df, val_df, test_df = split_data(root_dir)

    train_dataset = CaltechDataset(root_dir, train_df, transform = transform)
    val_dataset = CaltechDataset(root_dir, val_df, transform = transform)
    test_dataset = CaltechDataset(root_dir, test_df, transform = transform)

    # For debugging, you might want to set num_workers=0
    # num_workers=0 means data loading happens in the main process
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=num_workers_val, pin_memory=True, persistent_workers= (num_workers_val > 0) )
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=num_workers_val, pin_memory=True, persistent_workers= (num_workers_val > 0) )
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=num_workers_val, pin_memory=True, persistent_workers= (num_workers_val > 0) )

    return train_loader, val_loader, test_loader
# --- END OF MODIFIED data_utils.py ---