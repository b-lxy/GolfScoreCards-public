import os
import cv2
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as T
import torch

#########################################################################
"""
This dataset class uses a memory efficient approach by loading a batch of
.jpeg images and .txt labels only when the batch is called. The dataset only
stores image and label paths.
"""
#########################################################################

class ScorecardDataset(Dataset):
    def __init__(self, img_paths, label_paths, char2idx):
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.char2idx = char2idx
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(32),     # Standardize height ONLY
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_path = self.label_paths[idx]

        # load image
        strip = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if strip is None: raise ValueError(f"Failed to load {img_path}")
        # load label
        with open(label_path, 'r') as f:
            label = f.read().strip()

        img_tensor = self.transform(strip)
        label_tensor = torch.LongTensor([self.char2idx[c] for c in label])
        return img_tensor, label_tensor
    
def scorecard_collate_fn(batch, width_reduction=1):
    """
    To set width_reduction parameter for DataLoader, do:

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda batch: scorecard_collate_fn(batch, width_reduction=4)
    )
    
    """
    images, labels = zip(*batch)
    widths = [img.shape[2] for img in images]
    max_w = max(widths)
    padded_imgs = torch.stack([F.pad(img, (0, max_w - img.shape[2], 0, 0)) for img in images])
    input_lengths = torch.tensor([w // width_reduction for w in widths], dtype=torch.long) 
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    targets = torch.cat(labels)

    return padded_imgs, targets, input_lengths, target_lengths

class Tokenizer:
    def __init__(self):
        self.alphabet = "-0123456789@^+_|ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        # 0: [BLANK] (Required for CTC)
        # 1-10: '0'-'9'
        # 11: '@' (Circle)
        # 12: '^' (Triangle)
        # 13: '_' (Superscript)
        # 14: '|' (Delimiter for next cell)
        # 15: '+' (Square)
        # 16-42: 'A'-'Z'
        self.char2idx = {char: i for i, char in enumerate(self.alphabet)}
        self.idx2char = {i: char for i, char in enumerate(self.alphabet)}

    def encode(self, label):
        return torch.LongTensor([self.char2idx[c] for c in label])
    
def load_dataset(img_dir, label_dir, n):
    img_paths, label_paths = [], []
    img_files = sorted([f for f in os.listdir(img_dir)])
    if n:
        if n > len(img_files):
            raise ValueError(f"Cannot use {n} samples when only {len(img_files)} are available.")
        else:
            img_files = img_files[:n]
    for f_name in img_files:
        img_path = os.path.join(img_dir, f_name)
        lbl_name = os.path.splitext(f_name)[0] + '.txt'
        lbl_path = os.path.join(label_dir, lbl_name)
        if not os.path.exists(lbl_path):
            print(f"Warning: Missing label for {f_name}")
            continue
        img_paths.append(img_path)
        label_paths.append(lbl_path)

    return img_paths, label_paths