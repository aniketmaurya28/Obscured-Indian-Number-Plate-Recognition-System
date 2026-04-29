"""
=================================================================
CNN CHARACTER CLASSIFIER — Train & Test
For ANPR v10.0  |  Python 3.11.4
=================================================================

FOLDER STRUCTURE — put your images here:
─────────────────────────────────────────
dataset/
├── train/
│   ├── A/   ← put all 'A' character images here
│   ├── B/
│   ├── C/
│   │   ... (one folder per character)
│   ├── Z/
│   ├── 0/
│   ├── 1/
│   │   ...
│   └── 9/
│
├── val/     ← same structure (20% of your images)
│   ├── A/
│   ├── B/
│   │   ...
│   └── 9/
│
└── test/    ← same structure (10% of your images)
    ├── A/
    ├── B/
    │   ...
    └── 9/

IMAGE REQUIREMENTS:
  • Format : JPG, PNG, BMP
  • Size   : any size (auto-resized to 32×32)
  • Color  : color or grayscale (auto-converted)
  • Content: one character per image, cropped tightly

HOW TO SPLIT YOUR IMAGES:
  If you have 1000 images of 'A':
    → 700 go to  train/A/
    → 200 go to  val/A/
    → 100 go to  test/A/

USAGE:
  # Train the model
  python cnn_train.py --mode train

  # Test the trained model
  python cnn_train.py --mode test

  # Train then immediately test
  python cnn_train.py --mode both

  # Predict a single image
  python cnn_train.py --mode predict --image path/to/char.jpg
=================================================================
"""

import os
import argparse
import string
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (safe for all OS)
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# CONFIG  ← change these if needed
# ─────────────────────────────────────────────────────────────
DATASET_DIR   = r"D:\dataset"          # root folder with train/val/test
MODEL_SAVE    = "26char_classifier.pth"  # output model (used by main.py)
IMAGE_SIZE    = 32                 # must stay 32 — matches CharCNN
BATCH_SIZE    = 128
EPOCHS        = 30
LEARNING_RATE = 0.001
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 36 classes: A-Z then 0-9  (same order as main.py)
CLASSES = list(string.ascii_uppercase) + list(string.digits)
NUM_CLASSES = len(CLASSES)         # 36


# ══════════════════════════════════════════════════════════════
# 1. MODEL  (exact same architecture as main.py CharCNN)
# ══════════════════════════════════════════════════════════════
class CharCNN(nn.Module):
    def __init__(self, num_classes=36):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.block3(self.block2(self.block1(x))))


# ══════════════════════════════════════════════════════════════
# 2. DATASET  (reads the folder structure described above)
# ══════════════════════════════════════════════════════════════
class CharDataset(Dataset):
    """
    Loads character images from:
        dataset/train/A/img1.jpg
        dataset/train/B/img2.png  ...etc
    Preprocesses each image the same way main.py does at inference.
    """

    def __init__(self, root_dir, augment=False):
        self.samples  = []   # list of (image_path, class_index)
        self.augment  = augment

        # Build augmentation pipeline for training
        self.aug = transforms.Compose([
            transforms.RandomAffine(
                degrees=8,          # slight rotation
                translate=(0.08, 0.08),
                scale=(0.90, 1.10),
                shear=5,
            ),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.4),
        ]) if augment else None

        # Scan folders
        for cls in CLASSES:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                # try lowercase folder name too
                cls_dir = os.path.join(root_dir, cls.lower())
            if not os.path.isdir(cls_dir):
                continue
            idx = CLASSES.index(cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), idx)
                    )

        print(f"    Found {len(self.samples)} images in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # ── Read & preprocess  (mirrors CharClassifier.predict_top5) ──
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((32, 32), dtype=np.uint8)

        # Pad to square
        h, w = img.shape
        if h != w:
            t  = max(h, w)
            ph = (t - h) // 2
            pw = (t - w) // 2
            bg = int(np.median(img))
            img = cv2.copyMakeBorder(
                img, ph, t - h - ph, pw, t - w - pw,
                cv2.BORDER_CONSTANT, value=bg
            )

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE),
                         interpolation=cv2.INTER_AREA)

        # CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        img   = clahe.apply(img)

        # Binarize
        _, img = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Ensure dark text on white background
        if img.mean() > 127:
            img = cv2.bitwise_not(img)

        # Convert to tensor [0, 1]
        tensor = torch.FloatTensor(
            img.astype(np.float32) / 255.0
        ).unsqueeze(0)   # shape: (1, 32, 32)

        # Apply augmentation to the tensor (training only)
        if self.aug is not None:
            tensor = self.aug(tensor)

        return tensor, label


# ══════════════════════════════════════════════════════════════
# 3. TRAIN
# ══════════════════════════════════════════════════════════════
def train():
    print("\n" + "═"*55)
    print("  TRAINING  |  device:", DEVICE)
    print("═"*55)

    # ── Load datasets ──────────────────────────────────────────
    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir   = os.path.join(DATASET_DIR, "val")

    if not os.path.isdir(train_dir):
        print(f"\n  ERROR: Cannot find  {train_dir}")
        print("  Please create the folder structure shown at the top.")
        return

    train_ds = CharDataset(train_dir, augment=True)
    val_ds   = CharDataset(val_dir,   augment=False)

    if len(train_ds) == 0:
        print("\n  ERROR: No images found in train/")
        print("  Check that each character has its own subfolder.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # ── Model, loss, optimiser ─────────────────────────────────
    model     = CharCNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                           weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4, verbose=True
    )

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   []}

    print(f"\n  Classes  : {NUM_CLASSES}  (A-Z + 0-9)")
    print(f"  Train    : {len(train_ds)} images")
    print(f"  Val      : {len(val_ds)} images")
    print(f"  Epochs   : {EPOCHS}")
    print(f"  Batch    : {BATCH_SIZE}\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ── Training pass ──────────────────────────────────────
        model.train()
        train_loss, train_correct = 0.0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * imgs.size(0)
            train_correct += (out.argmax(1) == labels).sum().item()

        train_loss /= len(train_ds)
        train_acc   = train_correct / len(train_ds)

        # ── Validation pass ────────────────────────────────────
        model.eval()
        val_loss, val_correct = 0.0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out  = model(imgs)
                loss = criterion(out, labels)
                val_loss    += loss.item() * imgs.size(0)
                val_correct += (out.argmax(1) == labels).sum().item()

        val_loss /= len(val_ds) if len(val_ds) > 0 else 1
        val_acc   = val_correct / len(val_ds) if len(val_ds) > 0 else 0

        scheduler.step(val_acc)
        elapsed = time.time() - t0

        # ── Log ────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"loss {train_loss:.4f}  acc {train_acc:.1%}  |  "
              f"val_loss {val_loss:.4f}  val_acc {val_acc:.1%}  "
              f"({elapsed:.1f}s)")

        # ── Save best model ────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"  ✓ Saved best model  →  {MODEL_SAVE}  "
                  f"(val_acc {val_acc:.1%})")

    print(f"\n  Training complete.  Best val accuracy: {best_val_acc:.1%}")
    _plot_history(history)


def _plot_history(history):
    """Save training curves to training_curves.png"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["val_loss"],   label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot([a * 100 for a in history["train_acc"]], label="train")
    ax2.plot([a * 100 for a in history["val_acc"]],   label="val")
    ax2.set_title("Accuracy (%)"); ax2.set_xlabel("Epoch")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120)
    print("  Plot saved → training_curves.png")


# ══════════════════════════════════════════════════════════════
# 4. TEST
# ══════════════════════════════════════════════════════════════
def test():
    print("\n" + "═"*55)
    print("  TESTING")
    print("═"*55)

    if not os.path.isfile(MODEL_SAVE):
        print(f"\n  ERROR: Model not found at {MODEL_SAVE}")
        print("  Run training first:  python cnn_train.py --mode train")
        return

    test_dir = os.path.join(DATASET_DIR, "test")
    if not os.path.isdir(test_dir):
        print(f"\n  ERROR: Cannot find {test_dir}")
        return

    test_ds = CharDataset(test_dir, augment=False)
    if len(test_ds) == 0:
        print("\n  ERROR: No images found in test/")
        return

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    # Load model
    model = CharCNN(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_SAVE, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    print(f"\n  Loaded model: {MODEL_SAVE}")
    print(f"  Test images : {len(test_ds)}\n")

    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # ── Per-class accuracy ─────────────────────────────────────
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    overall = correct / len(all_labels)
    print(f"  Overall accuracy : {overall:.1%}  "
          f"({correct}/{len(all_labels)})\n")

    # ── Detailed report ────────────────────────────────────────
    print("  Per-class report:")
    print("  " + "-"*52)
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASSES,
        zero_division=0
    )
    for line in report.splitlines():
        print("  " + line)

    # ── Find worst-performing characters ──────────────────────
    print("\n  5 hardest characters (lowest per-class accuracy):")
    per_class_correct = {c: [0, 0] for c in CLASSES}
    for pred, label in zip(all_preds, all_labels):
        cls = CLASSES[label]
        per_class_correct[cls][1] += 1
        if pred == label:
            per_class_correct[cls][0] += 1

    worst = sorted(
        [(c, v[0]/v[1] if v[1] > 0 else 0)
         for c, v in per_class_correct.items() if v[1] > 0],
        key=lambda x: x[1]
    )[:5]
    for cls, acc in worst:
        print(f"    '{cls}' → {acc:.1%}")

    # ── Confusion matrix plot ──────────────────────────────────
    _plot_confusion(all_labels, all_preds)


def _plot_confusion(labels, preds):
    """Save confusion matrix to confusion_matrix.png"""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASSES, fontsize=8)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — CharCNN")

    # Annotate cells
    thresh = cm.max() / 2
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center', fontsize=6,
                        color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)
    print("\n  Confusion matrix saved → confusion_matrix.png")


# ══════════════════════════════════════════════════════════════
# 5. PREDICT A SINGLE IMAGE
# ══════════════════════════════════════════════════════════════
def predict_single(image_path):
    print(f"\n  Predicting: {image_path}")

    if not os.path.isfile(MODEL_SAVE):
        print(f"  ERROR: Model not found at {MODEL_SAVE}")
        return

    if not os.path.isfile(image_path):
        print(f"  ERROR: Image not found at {image_path}")
        return

    model = CharCNN(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_SAVE, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    # Build a tiny 1-image dataset
    dummy_ds = CharDataset.__new__(CharDataset)
    dummy_ds.samples = [(image_path, 0)]
    dummy_ds.augment = False
    dummy_ds.aug     = None
    dummy_ds.__class__ = CharDataset

    tensor, _ = dummy_ds[0]
    tensor    = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    top5_vals, top5_idx = torch.topk(probs, 5)

    print("\n  Top-5 predictions:")
    print("  " + "-"*28)
    for rank, (val, idx) in enumerate(
            zip(top5_vals.tolist(), top5_idx.tolist()), 1):
        marker = " ← best" if rank == 1 else ""
        print(f"  {rank}. '{CLASSES[idx]}'  {val:.1%}{marker}")


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="CharCNN — train / test / predict"
    )
    ap.add_argument(
        "--mode", choices=["train", "test", "both", "predict"],
        default="both",
        help="train | test | both | predict"
    )
    ap.add_argument(
        "--image", default=None,
        help="path to a single character image (used with --mode predict)"
    )
    args = ap.parse_args()

    print(f"\n  Device : {DEVICE}")
    print(f"  Classes: {NUM_CLASSES}  ({CLASSES[0]}..{CLASSES[-1]})")

    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    elif args.mode == "both":
        train()
        test()
    elif args.mode == "predict":
        if not args.image:
            print("  ERROR: --image required with --mode predict")
            print("  Example: python cnn_train.py --mode predict --image A.jpg")
        else:
            predict_single(args.image)