```markdown
# Self-Calibrated AutoEncoder for Image Colorization

An advanced deep learning model for **automatic image colorization** using a **Self-Calibrated AutoEncoder (SCA-Net)** architecture.  
Inspired by [lukemelas/Automatic-Image-Colorization](https://github.com/lukemelas/Automatic-Image-Colorization),
redesigned with self-calibrated convolutional layers for more accurate and stable color restoration.

---

## Features

- Self-Calibrated Convolutional AutoEncoder  
- Evaluation with PSNR, SSIM, and MSE metrics  
- Grayscale to color inference on images of any resolution  
- Supports high-resolution datasets (e.g., COCO, CelebA-HQ, Places365)  
- Clean, modular, and extensible codebase  

---

## Project Structure

Automatic-Image-Colorization/
├── main.py           # Training and validation workflow
├── model.py          # SCA-Net architecture with self-calibrated blocks
├── utils.py          # Dataset loader, metrics, and utility functions
├── inference.py      # Single-image inference script
├── checkpoints/      # Saved model weights
├── outputs/          # Output colorized images and visualizations
├── requirements.txt  # Python dependencies
└── README.md         # This documentation file

---

## Dataset Preparation

Prepare your dataset directory with separate training and validation splits:

dataset/
├── train/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── val/
    ├── imgA.jpg
    ├── imgB.jpg
    └── ...

- Store images as RGB (no need to manually convert to grayscale).

**Recommended Datasets:**

- [Places365-Standard](http://places2.csail.mit.edu/download.html)  
- [CelebA-HQ](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
- [COCO 2017 Images](https://cocodataset.org/#download)  

---

## Training

Run the following command to start training:

```
python main.py dataset/ --epochs 50 --batch-size 16
```

- Training prints MSE loss per batch.  
- After each epoch, PSNR and SSIM evaluation metrics are reported.  
- Visual outputs (grayscale, original, and colorized) are saved to `outputs/`.

---

## Model Overview

SCA-Net is an encoder-decoder convolutional network enhanced with self-calibrated blocks that enable:

- Context-aware feature recalibration  
- Improved texture and color fidelity  
- Natural look color restoration  

### Architecture Flow:

Input (L channel)  
↓  
Encoder (Self-Calibrated Blocks)  
↓  
Bottleneck (Self-Calibrated)  
↓  
Decoder (Upsampling + Skip Connections)  
↓  
Output (AB channels)

---

## Acknowledgements

- Original implementation by [lukemelas/Automatic-Image-Colorization](https://github.com/lukemelas/Automatic-Image-Colorization)  
- Self-Calibrated Conv design inspired by SCNet (Liu et al., CVPR 2020)  

---

## How to Use

1. Place `requirements.txt` and this `README.md` in your project root folder.  
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Prepare your dataset and start training or inference using provided scripts.

---
```

