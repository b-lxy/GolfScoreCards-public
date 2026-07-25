import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle as pkl
import cv2
import os, sys
from PIL import Image
import io
from pathlib import Path
import base64

from .image_registration import filter, invariance
from models.resnet18 import RESNET18

with open('APIs/templates.pkl', 'rb') as f:
    templates = pkl.load(f)

infer_transform = T.Compose([
    T.Resize((64, 64)),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

weights_path = "weights"
resnet_version = "resnet18_v1"
model = RESNET18(num_classes=10)
state_dict = torch.load(os.path.join(weights_path, resnet_version, "resnet_real_train_v1.pth"), map_location=model.device, weights_only=True)
model.load_state_dict(state_dict['model_state'])
model.eval()

available_templates = {1: "The Lakes", 2: "Saujana"}

def run_ocr(image_bytes, use_t, show_vis=False):

    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    h_grid, s_grids = extract_cells(image, use_t)
    merged_sg = []
    for rows in zip(*s_grids):
        merged_sg.append([cell for row in rows for cell in row])
    s_preds = recognize_scorecard(merged_sg, model, infer_transform)
    results_view = None
    if show_vis:
        results_view = result_vis(header_grid=h_grid, score_grid=merged_sg, score_preds=s_preds)
    return {"scores": s_preds, "header": None, "visual": results_view}


def extract_cells(image, use_t):

    template = templates.get(f"Data{str(use_t)}")
    reg_img = register_image(image, template)
    score_cells = template.get('score')
    header_cells = template.get('header')

    score_grids = crop_blocked_grid(reg_img, score_cells, target_h=48, pad=4)
    header_grid = crop_cells_grid(reg_img, *get_grid_from_block(header_cells), target_h=48, pad=4)
    norm_score_grids = [normalize_grids(grid, len(header_grid)) for grid in score_grids]

    return header_grid, norm_score_grids

def recognize_scorecard(grid, model, transform):

    rows, cols = len(grid), len(grid[0])  # fixed in your case
    flat_cells = [cell for row in grid for cell in row]
    tensors = [transform(Image.fromarray(cell)) for cell in flat_cells]
    batch = torch.stack(tensors).to(model.device)

    with torch.no_grad():
        probs, logits = model(batch)
        preds = torch.argmax(logits, dim=1)
        confs = torch.max(probs, dim=1).values
        confs = confs.cpu().numpy()
        preds = preds.cpu().numpy()

    # Reshape back to grid
    preds_grid = preds.reshape(rows, cols)
    confs_grid = confs.reshape(rows, cols)
    result_grid = [[{"row": i, "col": j, "digit": int(preds_grid[i][j]), "confidence": float(confs_grid[i][j])}
                    for j in range(cols)]
                    for i in range(rows)]
    
    return result_grid

def register_image(samp, template, method='default'):

    temp = template.get('image')
    crease_x = template.get('crease_x')

    if method == 'default':
        points_temp, points_samp = filter(samp, temp)
    elif method == 'invariance':
        points_temp, points_samp = invariance(samp, temp)

    height, width = temp.shape[:2]
    left_mask = points_temp[:, 0] < crease_x
    right_mask = points_temp[:, 0] >= crease_x
    reg_img = np.zeros_like(temp)

    left_inliers, right_inliers = 0, 0

    for mask, x_range, side in [(left_mask, (0, crease_x), "L"), (right_mask, (crease_x, width), "R")]:
        if np.sum(mask) < 4: continue
        
        # Rigidly align just this half
        H, inlier_mask = cv2.findHomography(points_samp[mask], points_temp[mask], cv2.RANSAC, 5.0)
        warped = cv2.warpPerspective(samp, H, (width, height))
        
        # Stitch into final image
        x1, x2 = x_range
        reg_img[:, x1:x2] = warped[:, x1:x2]

        if side == "L":   left_inliers = int(inlier_mask.sum())
        else:             right_inliers = int(inlier_mask.sum())
    
    return reg_img

def get_grid_from_block(block):
    """
    Get a grid from a block of coordinates.
    """
    # block shape: (10, 5, 2)
    block = np.asarray(block)

    x_coords = np.unique(block[:, 0]).astype(int)
    y_coords = np.unique(block[:, 1]).astype(int)

    return np.sort(x_coords), np.sort(y_coords)

def crop_cells_grid(img, x_coords, y_coords, target_h=28, pad=0):
    grid = []

    for i in range(len(y_coords) - 1):
        row = []
        for j in range(len(x_coords) - 1):
            x1, x2 = x_coords[j] - pad, x_coords[j+1] + pad
            y1, y2 = y_coords[i] - pad, y_coords[i+1] + pad

            cell = img[y1:y2, x1:x2]
            # resize to target height
            h, w = cell.shape[:2]
            scale = target_h / h
            new_w = max(1, int(w * scale))

            cell_resized = cv2.resize(cell, (new_w, target_h), interpolation=cv2.INTER_AREA)
            row.append(cell_resized)
        grid.append(row)

    return grid

def crop_blocked_grid(img, score_cells, target_h=28, pad=0):
    """
    Crop each blocks of score cells.
    """
    grids = []
    for block in score_cells:
        x_coords, y_coords = get_grid_from_block(block)
        grids.append(crop_cells_grid(img, x_coords, y_coords, target_h, pad))
    return grids

def normalize_grids(grid, target_rows):
    if len(grid) == target_rows:      return grid
    if len(grid[0]) == target_rows:   return [list(row) for row in zip(*grid)]

def result_vis(header_grid, score_grid, show_header=True, header_preds=None, score_preds=None):
    # 1. Prepare data
    full_display_grid = []
    if show_header:
        for h_row, s_row in zip(header_grid, score_grid):
            full_display_grid.append([h_row[0]] + s_row)
    else:
        full_display_grid = score_grid
        header_preds = None

    # 2. Determine grid dimensions
    num_rows = len(full_display_grid)
    num_cols = len(full_display_grid[0]) if num_rows > 0 else 0

    # 3. Create the subplot figure
    _, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 1))

    # 4. Fill the subplots
    for r in range(num_rows):
        for c in range(num_cols):
            ax = axes[r, c] if num_rows > 1 else axes[c]
            cell_img = full_display_grid[r][c]
            ax.imshow(cell_img, cmap='gray')

            if c == 0 and show_header:            # header cells titles
                if header_preds:
                    cell = header_preds[r][0]
                    p, c = cell['digit'], cell['confidence']
                    ax.set_title(f"{p} ({c:.3f})", fontsize=14)
            else:
                if score_preds:   # score cells titles
                    pred_col = c - 1 if show_header else c
                    cell = score_preds[r][pred_col]
                    pred, conf = cell['digit'], cell['confidence']
                    if conf <= 0.95:
                        for spine in ax.spines.values():
                            if conf < 0.9: spine.set_edgecolor('red')
                            else: spine.set_edgecolor('blue')
                            spine.set_linewidth(2)
                    ax.set_title(f"{pred} ({conf:.3f})", fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.subplots_adjust(wspace=0.02, hspace=0.5)
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("ascii")