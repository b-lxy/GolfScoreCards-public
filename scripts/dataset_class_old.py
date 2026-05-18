import os
import cv2
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as T
import torch

class ScorecardDataset(Dataset):
    """
    This class assumes that strips is a list of already-cropped images
    (numpy arrays or PIL images).
    """
    def __init__(self, strips, labels, char2idx):
        self.strips = strips # List of [H, W] numpy arrays
        self.labels = labels # List of strings (one per strip)
        self.char2idx = char2idx
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((32, 512)), # Standardize height for the CNN
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.strips)

    def __getitem__(self, idx):
        strip = self.strips[idx]
        label = self.labels[idx]
        img_tensor = self.transform(strip)
        
        # Convert string label to tensor of indices (e.g., "@4" -> [11,5])
        label_tensor = torch.LongTensor([self.char2idx[c] for c in label])
        return img_tensor, label_tensor
    
def scorecard_collate_fn(batch):
    target_height = 32
    max_pool_red = 1    ## change according to pooling configuration
    images, labels = zip(*batch)
    rescaled_imgs = []
    input_lengths = []
    for img in images:
        c, h, w = img.shape
        new_width = int(target_height * (w/h))
        rescaled_img = T.functional.resize(img, (target_height, new_width), antialias=True)
        rescaled_imgs.append(rescaled_img)
        input_lengths.append(new_width // max_pool_red)
    input_lengths = torch.tensor([w for w in input_lengths], dtype=torch.long)

    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    targets = torch.cat(labels)
    # pad width
    max_w = max(img.shape[2] for img in rescaled_imgs)
    padded_imgs = torch.stack([F.pad(img, (0, max_w - img.shape[2], 0, 0)) for img in rescaled_imgs])
    
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

    def encode(self, labels):
        # Input: ["@4_1", "5", "^3"]
        # Output: Flattened tensor and lengths for CTCLoss
        encoded = []
        for label in labels:
            encoded.extend([self.char2idx[c] for c in label])
        return torch.LongTensor(encoded)
    
def load_dataset(img_dir, label_dir):
    strips, labels = [], []
    img_files = sorted([f for f in os.listdir(img_dir)])
    for f_name in img_files:
        img_path = os.path.join(img_dir, f_name)
        lbl_name = os.path.splitext(f_name)[0] + '.txt'
        lbl_path = os.path.join(label_dir, lbl_name)
        if os.path.exists(lbl_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            with open(lbl_path, 'r') as f:
                lbl_str = f.read().strip()
            strips.append(img)
            labels.append(lbl_str)
        else:
            print(f"Warning: Missing label for {f_name}")

    return strips, labels