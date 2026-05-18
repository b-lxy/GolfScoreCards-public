import pandas   as pd
import torch
from tqdm       import tqdm

from scripts.metrics import cer
from scripts.utils import best_path_decode

def train_step(model, trainloader, optimizer, criterion, device):
    """
    Single epoch training function.
    """
    model.train()
    running_loss = 0
    for images, targets, input_lengths, target_lengths in tqdm(trainloader):
        images, targets = images.to(device), targets.to(device)
    
        # Forward pass: [TimeSteps, Batch, Classes]
        log_probs = model(images) 
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
    
        optimizer.zero_grad()
        loss.backward()
    
        # Gradient clipping is highly recommended for RNNs/CTC
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    
        optimizer.step()
        running_loss += loss.item()

    # return average loss for this epoch
    return running_loss / len(trainloader)   

def evaluate(model, dataloader, criterion, device, tokenizer):
    model.eval()
    total_loss = 0
    correct_seq = 0
    total_seq = 0
    gt_targets = []
    pred_targets = []
    total_cer = 0.0

    with torch.no_grad():
        for images, targets, input_lengths, target_lengths in dataloader:
            images, targets = images.to(device), targets.to(device)
            log_probs = model(images) 
            # batch_size = images.size(0)
            # input_lengths = torch.full((batch_size,), log_probs.size(0), dtype=torch.long).to(device)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()
            preds = best_path_decode(log_probs, tokenizer)

            start = 0
            for i, length in enumerate(target_lengths):
                target_indices = targets[start:start+length].tolist()
                target_str = [tokenizer.idx2char[idx] for idx in target_indices]
                start += length
                gt = ''.join(target_str)
                pred = ''.join(preds[i])
                total_cer += cer(gt, pred)
                total_seq += 1
                gt_targets.append(gt)
                pred_targets.append(pred)
    avg_loss = total_loss / len(dataloader)
    avg_cer = (total_cer / total_seq) if total_seq > 0 else 0
    examples = pd.DataFrame({
        'Ground Truth': gt_targets, 
        'Predicted': pred_targets
    })
    return avg_loss, avg_cer, examples