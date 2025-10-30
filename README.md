# 🎨 Self-Calibrated AutoEncoder for Image Colorization

This project is an advanced deep learning model for **automatic image colorization**, built with a **Self-Calibrated AutoEncoder (SCA-Net)** architecture.  
It is inspired by the original [lukemelas/Automatic-Image-Colorization](https://github.com/lukemelas/Automatic-Image-Colorization) project, but redesigned with self-calibrated convolutional layers for more accurate and stable color restoration.

---

## 🧩 Features
✅ Self-Calibrated Convolutional AutoEncoder  
✅ PSNR / SSIM / MSE evaluation metrics  
✅ Grayscale → Color inference on any resolution image  
✅ High-resolution dataset support (e.g., COCO, CelebA-HQ, Places365)  
✅ Clean, modular codebase  

---

## 🗂 Project Structure

Automatic-Image-Colorization/
├── main.py # Training + validation
├── model.py # Self-Calibrated AutoEncoder (SCA-Net)
├── utils.py # Dataset loader, metrics, visualizations
├── inference.py # Single-image inference script
├── checkpoints/ # Saved models
├── outputs/ # Visual results
├── requirements.txt
└── README.md


---

## 📚 Dataset Preparation

Prepare your dataset folder with **train/val** splits:

dataset/
├── train/
│ ├── img1.jpg
│ ├── img2.jpg
│ └── ...
└── val/
├── imgA.jpg
├── imgB.jpg
└── ...


Each folder should contain RGB images (no need to pre-convert to grayscale).

Good datasets:
- [Places365-Standard](http://places2.csail.mit.edu/download.html)
- [CelebA-HQ](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [COCO 2017 Images](https://cocodataset.org/#download)

---

## 🏋️‍♂️ Training

```bash
python main.py dataset/ --epochs 50 --batch-size 16

During training:

MSE loss is printed per batch.

PSNR and SSIM are computed after each epoch.

Output visualizations (grayscale, original, colorized) are saved under outputs/.

🧠 Model Overview
Self-Calibrated AutoEncoder (SCA-Net)

A convolutional encoder–decoder with self-calibrated blocks, enabling:

Context-aware feature recalibration

Better handling of texture consistency

Improved color balance for natural scenes

Input (L channel)
 ↓
Encoder (SC Blocks)
 ↓
Bottleneck (Self-Calibrated)
 ↓
Decoder (Upsampling + Skip Connections)
 ↓
Output (AB channels)

🙌 Acknowledgements

lukemelas/Automatic-Image-Colorization
 for the original framework

SCNet (Liu et al., CVPR 2020) for the Self-Calibrated Convolution idea


---

## ✅ Next Steps

Once you paste these:
1. Save `requirements.txt` and `README.md` in your root project folder.  
2. Run:
   ```bash
   pip install -r requirements.txt
