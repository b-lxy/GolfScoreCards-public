from pathlib import Path
import os, sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import confusion_matrix
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np


class CNN(nn.Module):
    def __init__(self, num_classes, lr=0.001):
        super(CNN, self).__init__()

        # expect 48x48 rescaled gray images
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), 
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(), nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((1, 1)), # Keep width resolution by collapsing height
        )
        cnn_out_channels = 256

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # Flatten to (Batch, 256)
        logits = self.fc(x)
        probs = F.softmax(logits, dim=1)
        return probs, logits
    
    def train_loop(self, train_loader, val_loader, epochs=5, scheduler=None):
        """
        Main training loop that iterates through epochs.
        """
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0

            # --- TRAINING PHASE ---
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                loss_val, correct_val = self._train_step(batch)
                
                total_loss += loss_val
                total_correct += correct_val
                total_samples += len(batch[0])
                pbar.set_postfix(loss=loss_val)

            avg_train_loss = total_loss / len(train_loader)
            train_acc = total_correct / total_samples

            # --- VALIDATION PHASE ---
            avg_val_loss, val_acc = self.evaluate(val_loader)

            if scheduler is not None: scheduler.step()
            
            # Save history for plotting later
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)

            print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%}\n")

        return history

    def _train_step(self, batch):
        """
        Performs a single training step on a batch of data.
        """
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()
        probs, logits = self.forward(images)
        loss = self.criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        _, preds = torch.max(logits, 1)
        correct = (preds == labels).sum().item()

        return loss.item(), correct
    
    def evaluate(self, dataloader, return_pred=False, detailed=False):
        """
        Evaluates the model on a given dataloader.
        Returns: Average Loss (if criterion provided) and Accuracy.
        """
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        pred = []
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                probs, logits = self.forward(images)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                    
                conf, predicted = torch.max(logits, 1)
                if return_pred:
                    for p, c in zip(predicted, conf):
                        pred.append((int(p.item() + 1), float(c.item())))
                if detailed:
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        outputs = [avg_loss, accuracy]
        
        if return_pred: outputs.append(pred)
        if detailed:
            cm = confusion_matrix(all_labels, all_preds)
            outputs.append(cm)
        return tuple(outputs)
    

class ScorecardDataset(Dataset):
    def __init__(self, data_dir, transform=None, syndata_dir=None, include_blank=False):
        """
        Args:
            data_dir (str): Path to the folder containing images and labels.
            transform (callable, optional): Optional transform to be applied on a REAL sample.
        """
        csv_path = os.path.join(data_dir, "labels.csv")
        self.df = pd.read_csv(csv_path)
        self.df['is_synthetic'] = False
        self.real_df = self.df.copy()
        self.img_dir = Path(os.path.join(data_dir, "images"))
        self.transform = transform
        self.syn_transform = T.Compose([
                                    T.Resize((48, 48)),              # Very slight rotation (upright cells)
                                    T.ColorJitter(brightness=0.1, contrast=0.1), # Simulate different pencil/ink/light
                                    T.ToTensor(), 
                                    T.Normalize((0.5,), (0.5,)) # Centers data around 0 for faster convergence
                                ])
        self.syn_df = None

        if syndata_dir:
            syn_csv_path = os.path.join(syndata_dir, "labels.csv")
            self.syn_df = pd.read_csv(syn_csv_path)
            self.syn_df['is_synthetic'] = True
            self.syn_img_dir = Path(os.path.join(syndata_dir, "images"))
            self.df = pd.concat([self.df, self.syn_df], ignore_index=True)
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()

        # Get row data
        row = self.df.iloc[idx]
        img_filename = row["image"]
        label = 0 if row['label'] == '-' else int(row["label"]) 
        assert 0 <= label <= 9
        is_syn = row['is_synthetic']

        # Select the correct directory based on the flag
        current_root = self.syn_img_dir if is_syn else self.img_dir
        img_path = os.path.join(current_root, img_filename)
        image = Image.open(img_path).convert('L')
        
        if self.transform and not is_syn: image = self.transform(image)
        if is_syn: image = self.syn_transform(image)
        return image, label