import os
import torch
from   torch.utils.data  import DataLoader
import matplotlib.pyplot as plt
import matplotlib        as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

from scripts.constants    import PROJECT_ROOT, ALPHABET
from scripts.metrics      import cer
from scripts.utils        import best_path_decode
from scripts.dataset_class import ScorecardDataset, load_dataset, scorecard_collate_fn

def real_data_eval(model, criterion, device, tokenizer, save_fig, show=True, print_text='', n=None, real_datadir=os.path.join(PROJECT_ROOT, 'real_dataset'), width_reduction=1):
    """
    Evaluate the model on real-world data.
    """
    real_img_path = os.path.join(real_datadir, "images")
    real_lbl_path = os.path.join(real_datadir, "labels")
    strips, labels = load_dataset(real_img_path, real_lbl_path, n)
    real_dataset = ScorecardDataset(strips, labels, tokenizer.char2idx)
    dataloader = DataLoader(real_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: scorecard_collate_fn(x, width_reduction=width_reduction))

    N = len(real_dataset)
    fig, ax = plt.subplots(N+1, 1, figsize=(8, 6*N), constrained_layout=True)
    ax = ax.ravel()
    plot_idx = 1

    model.eval()
    total_loss = 0
    total_seq = 0
    total_cer = 0.0
    min_cer = None
    max_cer = None
    with torch.no_grad():
        for images, targets, input_lengths, target_lengths in dataloader:
            images, targets = images.to(device), targets.to(device)
            log_probs = model(images) 
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
                cer_score = cer(gt, pred)
                if min_cer == None or cer_score < min_cer: min_cer = cer_score
                if max_cer == None or cer_score > max_cer: max_cer = cer_score
                total_cer += cer_score
                total_seq += 1
                display_text = (
                    f"{'Ground truth':<15}: {gt}\n"
                    f"{'Prediction':<15}: {pred}\n"
                    f"{'CER':<15}: {cer_score:.3f}"
                )
                h, w = images[i].shape[-2:]
                ax[plot_idx].set_box_aspect(h/w)
                ax[plot_idx].imshow(images[i].cpu().squeeze(0).numpy(), cmap='gray', aspect='auto')
                ax[plot_idx].text(0.0, -0.1, display_text,transform=ax[plot_idx].transAxes, ha='left', va='top')
                ax[plot_idx].axis('off')
                plot_idx += 1

    avg_loss = total_loss / len(dataloader)
    avg_cer = (total_cer / total_seq) if total_seq > 0 else 0
    title_text = (
        f"Model evaluation on real data - {N} samples\n"
        f"Average loss: {avg_loss:.4f}\n"
        f"Average CER: {avg_cer:.3f}\n\n"
        f"Alphabet: {ALPHABET}\n\n"
        f"Description: {print_text}"
    )
    ax[0].text(0.0, 0.7, title_text, transform=ax[0].transAxes, ha='left', va='center', fontsize=14)
    ax[0].axis('off')
    plt.tight_layout(rect=[0, 0.8, 1, 1])

    if save_fig:
        save_fig_dir = os.path.join(PROJECT_ROOT, 'results')
        os.makedirs(save_fig_dir, exist_ok=True)
        plt.savefig(fname=f'{save_fig_dir}/{save_fig}.pdf', format='pdf', bbox_inches='tight')
    if show:
        plt.show()

    plt.close()

    return {
        'N': N,
        'average_loss': avg_loss,
        'average_cer': avg_cer,
        'min_cer': min_cer,
        'max_cer': max_cer
    }
