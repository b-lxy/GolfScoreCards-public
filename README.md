# OCR on Golf Score Cards

Author: Xin Ying Leong (https://www.linkedin.com/in/blxy)

> This project is still in progress.

This project builds a specialized OCR pipeline to extract handwritten scores from a single snapshot for subsequent score digitalization.
Golf scores often include messy handwritting, superscripts and shapes, making it a challenge for the recognition of the main digit.
The pipeline includes the following stages:
- Image registration on a defined template with cropping points, which are used to crop score cells.
- Score cells are passed through a Convolutional Neural Network for classification, classes include digits 1-9 and the blank cell (`-`). Confidence scores are returned with the prediction for threshold-based decision making.


#### Setup Steps

1. Create a venv.
2. Activate the venv and run the following commands ONE AT A TIME:
    ```
    pip install --upgrade pip
    pip install torch torchvision torchaudio   ## cpu version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121   ## gpu version (specify cuda version)
    pip install -r requirements.txt
    ```

# Preliminary Scripts

#### `scripts/rename_samples.py`

Usage examples when running from terminal:
 1. `python rename_samples.py ..\data\Data1 --replace`
 2. `python rename_samples.py ..\data\Data1 --new_ext jpg`
 3. `python rename_samples.py ..\data\Data1 --base_name train --output_dir ..\data\train_images`
