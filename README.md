```markdown
# Self-Calibrated AutoEncoder for Image Colorization

This project implements an advanced deep learning model for **automatic image colorization**, built with a **Self-Calibrated AutoEncoder (SCA-Net)** architecture.
It is based on the original [lukemelas/Automatic-Image-Colorization](https://github.com/lukemelas/Automatic-Image-Colorization) but redesigned with self-calibrated
convolutional layers to achieve more accurate and stable color restoration.

---

## Features

- Self-Calibrated Convolutional AutoEncoder  
- Evaluation metrics: PSNR, SSIM, MSE  
- Grayscale to color inference at any image resolution  
- Support for high-resolution datasets (e.g., COCO, CelebA-HQ, Places365)  
- Clean, modular, and maintainable codebase  

---

## Project Structure

```
Automatic-Image-Colorization/
├── main.py           # Training and validation pipeline
├── model.py          # Self-Calibrated AutoEncoder (SCA-Net) architecture
├── utils.py          # Dataset loader, metrics, and visualizations
├── inference.py      # Script for inference on single images
├── checkpoints/      # Directory for saved model checkpoints
├── outputs/          # Directory for saving output visualizations
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```

---

## Dataset Preparation

Organize your dataset with train and validation splits as follows:

```
dataset/
├── train/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── val/
    ├── imgA.jpg
    ├── imgB.jpg
    └── ...
```

Each folder should contain RGB images (no need to pre-convert to grayscale).

**Recommended Datasets:**

- [Places365-Standard](http://places2.csail.mit.edu/download.html)  
- [CelebA-HQ](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
- [COCO 2017 Images](https://cocodataset.org/#download)  

---

## Training

To train the model, run:

```
python main.py dataset/ --epochs 50 --batch-size 16
```

**During Training:**

- MSE loss is displayed per batch.  
- PSNR and SSIM metrics are computed after each epoch.  
- Visual outputs (grayscale, original color, colorized) are saved under the `outputs/` directory.

---

## Model Overview

The Self-Calibrated AutoEncoder (SCA-Net) is a convolutional encoder-decoder network enhanced with self-calibrated convolutional blocks. This design allows for:

- Context-aware feature recalibration  
- Improved texture consistency in colorization  
- Balanced and natural color restoration  

**Architecture flow:**

```
Input (L channel)
    ↓
Encoder (Self-Calibrated Blocks)
    ↓
Bottleneck (Self-Calibrated)
    ↓
Decoder (Upsampling + Skip Connections)
    ↓
Output (AB channels)
```

---

## Acknowledgements

- Original framework inspired by [lukemelas/Automatic-Image-Colorization](https://github.com/lukemelas/Automatic-Image-Colorization)  
- Self-Calibrated convolution concept from SCNet (Liu et al., CVPR 2020)  

---

## Next Steps

1. Save `requirements.txt` and `README.md` files in your project root.  
2. Install dependencies with:  
    ```
    pip install -r requirements.txt
    ```  

3. Begin training or inference on your prepared dataset.

---
```

