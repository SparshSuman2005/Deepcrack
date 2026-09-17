
## DeepCrack — U-Net Crack Segmentation

-----------------------------------------------------------
By Computer Vision and Remote Sensing Lab, Wuhan University.

A deep learning pipeline for **pixel-level crack detection** in surface images using a **U-Net** encoder-decoder architecture built with PyTorch.

---

## Project Structure

```
DeepCrack/
├── dataset/                    # Image data (excluded from git)
│   ├── train_img/              # Original training images
│   ├── train_lab/              # Original training masks
│   ├── train_img_resized/      # Resized 256×256 training images
│   ├── resized_lab_mask/       # Resized 256×256 training masks
│   ├── test_img/               # Test images
│   └── test_lab/               # Test masks
├── preprocessing/              # Data preprocessing scripts
│   ├── know_about.py           # Dataset exploration / visualization
│   ├── resizeimage.py          # Resize training images to 256×256
│   └── resize_labels.py        # Resize training masks to 256×256
├── src/                        # Model source code
│   ├── model.py                # U-Net architecture
│   ├── dataset.py              # PyTorch Dataset for crack images
│   ├── transforms.py           # Joint image-mask augmentations
│   ├── losses.py               # DiceLoss, BCEDiceLoss, FocalLoss
│   ├── metrics.py              # IoU, Dice, Precision, Recall, F1
│   ├── train.py                # Training script
│   ├── evaluate.py             # Test-set evaluation
│   ├── predict.py              # Single image / batch inference
│   └── visualize.py            # Training curves & prediction plots
├── checkpoints/                # Saved model weights (gitignored)
├── outputs/                    # Logs, predictions, figures (gitignored)
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Model Architecture

Standard **U-Net** with 4 encoder levels, a bottleneck, and 4 decoder levels:

```
Input (3×256×256)
  ├── Encoder 1:  3  → 64   (+ skip)
  ├── Encoder 2:  64 → 128  (+ skip)
  ├── Encoder 3:  128 → 256 (+ skip)
  ├── Encoder 4:  256 → 512 (+ skip)
  ├── Bottleneck: 512 → 1024
  ├── Decoder 4:  1024 → 512 (+ skip from Enc4)
  ├── Decoder 3:  512 → 256  (+ skip from Enc3)
  ├── Decoder 2:  256 → 128  (+ skip from Enc2)
  ├── Decoder 1:  128 → 64   (+ skip from Enc1)
  └── Output:     64 → 1 (sigmoid)
Output (1×256×256)
```

**~31M trainable parameters**

---

## Setup

```bash
# Clone the repository
git clone https://github.com/SparshSuman2005/Deepcrack.git
cd Deepcrack

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Train the model

```bash
python src/train.py --epochs 50 --batch_size 8 --lr 1e-3
```

**Key arguments:**
| Argument | Default | Description |
|---|---|---|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 8 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--val_split` | 0.2 | Validation split fraction |
| `--img_size` | 256 | Image size (square) |

### Evaluate on test set

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pth
```

### Run inference on images

```bash
# Single image
python src/predict.py --input path/to/image.jpg

# Entire directory (with ground truth comparison)
python src/predict.py --input dataset/test_img --mask_dir dataset/test_lab
```

### Visualize training history

```bash
python src/visualize.py --mode both
```

---

## Dataset

The DeepCrack dataset from Wuhan University:
- **300 training** image-mask pairs (resized to 256×256)
- **237 test** image-mask pairs

### Citation

```bibtex
@article{liu2019deepcrack,
  title={DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack Segmentation},
  author={Liu, Yahui and Yao, Jian and Lu, Xiaohu and Xie, Renping and Li, Li},
  journal={Neurocomputing},
  volume={338},
  pages={139--153},
  year={2019},
  doi={10.1016/j.neucom.2019.01.036}
}
```

---

## Contact

Please contact Yahui Liu (yahui.liu AT unitn.it) for questions about the dataset.
