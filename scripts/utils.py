import os
import torch
import matplotlib.pyplot as plt
import matplotlib        as mpl
import importlib
import torch.optim       as optim
import cv2

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

from constants import PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_REGISTRY = {
    "crnn": ("models.crnn.model_class", "CRNN"),
    "crnn2": ("models.crnn.model_class", "CRNN2")
}

def save_images_as_jpeg(images, output_dir, prefix="image"):
    os.makedirs(output_dir, exist_ok=True)

    for i, img in enumerate(images):
        filename = os.path.join(output_dir, f"{prefix}_{i}.jpeg")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, img)

    print(f"Saved {len(images)} images to {output_dir}")

def get_last_index(dir_path, level="sample"):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        return None

    indices = []
    for name in os.listdir(dir_path):
        if level == "dataset": idx = int(name.replace("set", ""))
        else: idx = int(name.split("_")[1].split(".")[0])
        indices.append(idx)

    return max(indices) if indices else None

def get_model_class(model_type: str):
    module_path, class_name = MODEL_REGISTRY[model_type]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def see_losses(model_name, remove_first_n=0):
    dir = os.path.join(PROJECT_ROOT, "weights", model_name)
    checkpoints = [os.path.join(dir, f) for f in os.listdir(dir)]
    tloss, vloss = [], []
    for cp_path in checkpoints:
        cp = torch.load(cp_path, map_location=device, weights_only=False)
        tloss += cp['train_loss']
        vloss += cp['validation_loss']

    x = range(remove_first_n, remove_first_n + len(tloss[remove_first_n:]))

    plt.plot(x, tloss[remove_first_n:], label='Train Loss', marker='o')
    plt.plot(x, vloss[remove_first_n:], label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.show()

def load_checkpoint(model_class, model_name, checkpoint_name, num_classes):
    dir = os.path.join(PROJECT_ROOT, "weights", model_name)
    cp_path = os.path.join(dir, f"{checkpoint_name}.pt")
    checkpoint = torch.load(cp_path, map_location=device, weights_only=False)
    MC = get_model_class(model_class)
    model = MC(num_classes).to(device)
    model.load_state_dict(checkpoint.get('model_state_dict'))
    optimizer = optim.Adadelta(model.parameters())
    optimizer.load_state_dict(checkpoint.get("optimizer_state_dict"))

    return model, optimizer, checkpoint

def best_path_decode(model_out, tokenizer):
    """
    Uses the CTC Collapse logic to obtain the final score.
    
    :param model_out: Description
    :param tokenizer: Description
    """
    arg_maxes = torch.argmax(model_out, dim=2)  # [TimeSteps, Batch]
    arg_maxes = arg_maxes.permute(1,0)          # [Batch, TimeSteps]
    final_preds = []
    for batch_idx in range(arg_maxes.size(0)):
        seq = arg_maxes[batch_idx].tolist()
        decoded = []
        prev_char = -1
        for char_idx in seq:
            if char_idx != prev_char and char_idx != 0:  # check for BLANK character
                decoded.append(tokenizer.idx2char[char_idx])
            prev_char = char_idx
        final_preds.append("".join(decoded))
    return final_preds

