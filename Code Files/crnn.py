"""
=================================================================
CRNN TRAINING SCRIPT — Indian Obscured Plate Recognition
=================================================================
GPU     : NVIDIA RTX 3050 6GB
Python  : 3.11.4
Dataset : Synthetic Indian Plates (25,000 images)

Run:
    python crnn_train.py

Output:
    models/crnn_best.pth     ← best model weights
    models/crnn_final.pth    ← final epoch weights
    logs/training_log.csv    ← per epoch metrics
    logs/training_plot.png   ← loss/accuracy curves
=================================================================
"""

import os
import csv
import time
import random
import string
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path


# ══════════════════════════════════════════
# CONFIG — edit paths here
# ══════════════════════════════════════════
DATASET_DIR  = r"C:\Users\anike\Downloads\Synthetic_Plate\Synthetic_Plate\dataset"
OUTPUT_DIR   = r"D:\ANPR YOLO\CRNN"
MODEL_DIR    = f'{OUTPUT_DIR}/models'
LOG_DIR      = f'{OUTPUT_DIR}/logs'

# Image dimensions — must match generator
IMG_W        = 128    # CRNN input width
IMG_H        = 32     # CRNN input height

# Training
BATCH_SIZE   = 64
NUM_EPOCHS   = 50
LEARNING_RATE= 0.001
NUM_WORKERS  = 4

# Early stopping
PATIENCE     = 7

# Characters
CHARACTERS   = string.ascii_uppercase + string.digits  # A-Z + 0-9
BLANK_IDX    = 0   # CTC blank token index
NUM_CLASSES  = len(CHARACTERS) + 1  # 37 (36 + blank)

# Create output folders
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════
# LABEL ENCODER / DECODER
# ══════════════════════════════════════════
char_to_idx = {c: i+1 for i, c in enumerate(CHARACTERS)}
idx_to_char = {i+1: c for i, c in enumerate(CHARACTERS)}
# idx 0 = CTC blank


def encode_label(text):
    """Convert plate text string to list of indices."""
    return [char_to_idx[c] for c in text.upper() if c in char_to_idx]


def decode_prediction(preds):
    """
    Greedy CTC decoding.
    preds: (seq_len, num_classes) tensor
    """
    _, indices = preds.max(1)
    indices    = indices.cpu().numpy()

    chars = []
    prev  = BLANK_IDX
    for idx in indices:
        if idx != BLANK_IDX and idx != prev:
            if idx in idx_to_char:
                chars.append(idx_to_char[idx])
        prev = idx

    return ''.join(chars)


# ══════════════════════════════════════════
# DATASET CLASS
# ══════════════════════════════════════════
class PlateDataset(Dataset):
    def __init__(self, split='train'):
        self.split    = split
        self.img_dir  = f'{DATASET_DIR}/{split}/images'
        self.csv_path = f'{DATASET_DIR}/{split}/labels.csv'

        # Load CSV
        self.samples = []
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    'filename': row['filename'],
                    'label':    row['label'],
                    'level':    row['level'],
                })

        print(f"[{split}] Loaded {len(self.samples):,} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        img_path = f"{self.img_dir}/{sample['filename']}"
        img      = cv2.imread(img_path)

        if img is None:
            # Return blank if image missing
            img = np.zeros((IMG_H, IMG_W), dtype=np.float32)
            img = torch.FloatTensor(img).unsqueeze(0)
            return img, torch.IntTensor([]), 0

        # Preprocess
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_W, IMG_H),
                        interpolation=cv2.INTER_AREA)

        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        img   = clahe.apply(img)

        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0

        # To tensor (1, H, W)
        img = torch.FloatTensor(img).unsqueeze(0)

        # Encode label
        label         = encode_label(sample['label'])
        label_tensor  = torch.IntTensor(label)
        label_length  = len(label)

        return img, label_tensor, label_length


def collate_fn(batch):
    """
    Custom collate to handle variable length labels.
    Pads labels to same length for batching.
    """
    images, labels, label_lengths = zip(*batch)

    # Stack images
    images = torch.stack(images, 0)

    # Concatenate labels
    label_lengths = torch.IntTensor(label_lengths)
    labels        = torch.cat(labels, 0)

    return images, labels, label_lengths


# ══════════════════════════════════════════
# CRNN MODEL
# ══════════════════════════════════════════
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            bidirectional=True,
            batch_first=False
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        output, _ = self.rnn(x)
        return self.fc(output)


class CRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # CNN Backbone
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),          # 32→16 height, 128→64 width
            nn.Dropout2d(0.2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),          # 16→8 height, 64→32 width
            nn.Dropout2d(0.2),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout2d(0.2),

            # Block 4 — no MaxPool, preserve width as time steps
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout2d(0.2),
        )

        # Projection — reduces 256*8=2048 → 256
        # prevents BiLSTM from having 21M params
        self.projection = nn.Linear(256 * 8, 256)

        # RNN — reads CNN features as sequence
        self.rnn = nn.Sequential(
            BidirectionalLSTM(256, 256, 256),
            BidirectionalLSTM(256, 256, num_classes),
        )

    def forward(self, x):
        # CNN feature extraction
        # x: (batch, 1, 32, 128)
        features = self.cnn(x)
        # features: (batch, 256, 8, 32)

        b, c, h, w = features.size()

        # Reshape: width becomes time steps
        features = features.permute(3, 0, 1, 2)
        # (32, batch, 256, 8)

        features = features.reshape(w, b, c * h)
        # (32, batch, 2048)

        # Project down before LSTM
        features = self.projection(features)
        # (32, batch, 256)

        # RNN sequence reading
        output = self.rnn(features)
        # (32, batch, num_classes)

        return output


# ══════════════════════════════════════════
# ACCURACY METRICS
# ══════════════════════════════════════════
def compute_accuracy(model, dataloader, device):
    """
    Compute plate-level and character-level accuracy.
    Plate accuracy  = exact match (full plate correct)
    Char accuracy   = per character correct
    """
    model.eval()
    plate_correct = 0
    char_correct  = 0
    char_total    = 0
    total         = 0

    with torch.no_grad():
        for images, labels, label_lengths in dataloader:
            images = images.to(device)
            output = model(images)
            # output: (seq_len, batch, num_classes)

            # Log softmax for CTC
            log_probs = torch.nn.functional.log_softmax(
                output, dim=2
            )

            batch_size = images.size(0)
            label_idx  = 0

            for i in range(batch_size):
                # Decode prediction
                pred = decode_prediction(log_probs[:, i, :])

                # Get ground truth
                l_len = label_lengths[i].item()
                gt_indices = labels[label_idx:label_idx + l_len]
                gt = ''.join(
                    idx_to_char.get(idx.item(), '')
                    for idx in gt_indices
                )
                label_idx += l_len

                # Plate accuracy
                if pred == gt:
                    plate_correct += 1

                # Character accuracy
                for p, g in zip(pred, gt):
                    if p == g:
                        char_correct += 1
                char_total += max(len(pred), len(gt))

                total += 1

    plate_acc = plate_correct / max(total, 1)
    char_acc  = char_correct  / max(char_total, 1)
    return plate_acc, char_acc


# ══════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════
def train():
    print("=" * 60)
    print("CRNN TRAINING — Indian Obscured Plate Recognition")
    print("=" * 60)

    # Device
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device  : {device}")
    if device.type == 'cuda':
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
        print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Datasets
    print("\nLoading datasets...")
    train_ds = PlateDataset('train')
    val_ds   = PlateDataset('val')

    train_ld = DataLoader(
        train_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = NUM_WORKERS,
        collate_fn  = collate_fn,
        pin_memory  = True if device.type == 'cuda' else False,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        collate_fn  = collate_fn,
        pin_memory  = True if device.type == 'cuda' else False,
    )

    # Model
    model = CRNN(num_classes=NUM_CLASSES).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters : {params:,}")

    # Loss — CTC
    criterion = nn.CTCLoss(
        blank        = BLANK_IDX,
        reduction    = 'mean',
        zero_infinity= True    # prevents inf loss on bad batches
    )

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr           = LEARNING_RATE,
        weight_decay = 1e-4
    )

    # Scheduler — reduce LR when val loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode    = 'min',
        patience= 3,
        factor  = 0.5,
        min_lr  = 1e-6,
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss':   [],
        'plate_acc':  [],
        'char_acc':   [],
    }

    best_val_loss  = float('inf')
    best_plate_acc = 0.0
    wait           = 0
    start_time     = time.time()

    # CSV log
    log_path = f'{LOG_DIR}/training_log.csv'
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'val_loss',
            'plate_acc', 'char_acc', 'lr', 'time'
        ])

    print(f"\n{'='*75}")
    print(f"{'Ep':>4} | {'TrLoss':>8} | {'VaLoss':>8} | "
          f"{'PlateAcc':>9} | {'CharAcc':>8} | "
          f"{'LR':>8} | {'Time':>6} | Status")
    print(f"{'='*75}")

    for epoch in range(NUM_EPOCHS):
        ep_start = time.time()

        # ── Train ────────────────────────────
        model.train()
        train_loss = 0.0
        num_batches = 0

        for images, labels, label_lengths in tqdm(
            train_ld,
            desc=f'Epoch {epoch+1}/{NUM_EPOCHS}',
            leave=False
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(images)
            # output: (seq_len, batch, num_classes)

            # Log softmax required by CTCLoss
            log_probs = torch.nn.functional.log_softmax(
                output, dim=2
            )

            # Input lengths — all same (seq_len = 32)
            seq_len      = output.size(0)
            batch_size   = images.size(0)
            input_lengths = torch.full(
                (batch_size,), seq_len,
                dtype=torch.long
            )

            loss = criterion(
                log_probs,
                labels,
                input_lengths,
                label_lengths
            )

            loss.backward()

            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )

            optimizer.step()

            train_loss  += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / max(num_batches, 1)

        # ── Validate ─────────────────────────
        model.eval()
        val_loss    = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, labels, label_lengths in val_ld:
                images = images.to(device)
                labels = labels.to(device)

                output    = model(images)
                log_probs = torch.nn.functional.log_softmax(
                    output, dim=2
                )

                seq_len      = output.size(0)
                batch_size   = images.size(0)
                input_lengths = torch.full(
                    (batch_size,), seq_len,
                    dtype=torch.long
                )

                loss = criterion(
                    log_probs,
                    labels,
                    input_lengths,
                    label_lengths
                )

                val_loss    += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        # ── Accuracy (every 5 epochs) ─────────
        if (epoch + 1) % 5 == 0 or epoch == 0:
            plate_acc, char_acc = compute_accuracy(
                model, val_ld, device
            )
        else:
            plate_acc = history['plate_acc'][-1] if history['plate_acc'] else 0.0
            char_acc  = history['char_acc'][-1]  if history['char_acc']  else 0.0

        # ── Scheduler step ────────────────────
        scheduler.step(avg_val_loss)
        lr = optimizer.param_groups[0]['lr']

        # ── Save best model ───────────────────
        status = ''
        if avg_val_loss < best_val_loss:
            best_val_loss  = avg_val_loss
            best_plate_acc = plate_acc
            wait           = 0
            torch.save(
                model.state_dict(),
                f'{MODEL_DIR}/crnn_best.pth'
            )
            status = '★ SAVED'
        else:
            wait += 1

        # ── History ───────────────────────────
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['plate_acc'].append(plate_acc)
        history['char_acc'].append(char_acc)

        # ── Log to CSV ────────────────────────
        ep_time = time.time() - ep_start
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1,
                f'{avg_train_loss:.4f}',
                f'{avg_val_loss:.4f}',
                f'{plate_acc:.4f}',
                f'{char_acc:.4f}',
                f'{lr:.2e}',
                f'{ep_time:.1f}s',
            ])

        # ── Print ─────────────────────────────
        print(
            f"{epoch+1:4d} | "
            f"{avg_train_loss:8.4f} | "
            f"{avg_val_loss:8.4f} | "
            f"{plate_acc:8.1%} | "
            f"{char_acc:7.1%} | "
            f"{lr:8.2e} | "
            f"{ep_time:5.1f}s | "
            f"{status}"
        )

        # ── Early stopping ────────────────────
        if wait >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            print(f"No improvement for {PATIENCE} epochs")
            break

    # ── Save final model ──────────────────────
    torch.save(
        model.state_dict(),
        f'{MODEL_DIR}/crnn26.pth'
    )

    # ── Training summary ──────────────────────
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time      : {total_time/60:.1f} min")
    print(f"Best val loss   : {best_val_loss:.4f}")
    print(f"Best plate acc  : {best_plate_acc:.1%}")
    print(f"Model saved     : {MODEL_DIR}/crnn26.pth")
    print(f"Log saved       : {log_path}")

    # ── Plot training curves ──────────────────
    plot_training(history)

    return model, history


# ══════════════════════════════════════════
# PLOT TRAINING CURVES
# ══════════════════════════════════════════
def plot_training(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('CRNN Training — Indian Plate Recognition', fontsize=13)

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['val_loss'],   label='Val Loss')
    axes[0].set_title('CTC Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plate accuracy
    axes[1].plot(epochs, history['plate_acc'], color='green')
    axes[1].set_title('Plate Accuracy (Exact Match)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    # Character accuracy
    axes[2].plot(epochs, history['char_acc'], color='orange')
    axes[2].set_title('Character Accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = f'{LOG_DIR}/training_plot.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {out}")


# ══════════════════════════════════════════
# QUICK TEST — run after training
# ══════════════════════════════════════════
def test_model(model_path=None):
    """Test trained model on a few sample images."""
    if model_path is None:
        model_path = f'{MODEL_DIR}/crnn26.pth'

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    model = CRNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    # Load test dataset
    test_ds = PlateDataset('test')
    test_ld = DataLoader(
        test_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        collate_fn  = collate_fn,
    )

    print("\nTESTING MODEL")
    print("=" * 50)

    plate_acc, char_acc = compute_accuracy(model, test_ld, device)

    print(f"Test Plate Accuracy : {plate_acc:.1%}")
    print(f"Test Char Accuracy  : {char_acc:.1%}")

    # Show 10 sample predictions
    print("\nSample Predictions:")
    print(f"{'Ground Truth':<15} {'Predicted':<15} {'Match'}")
    print("-" * 40)

    sample_ds = PlateDataset('test')
    shown     = 0

    with torch.no_grad():
        for i in range(min(50, len(sample_ds))):
            img, label_t, label_len = sample_ds[i]
            img_input = img.unsqueeze(0).to(device)

            output    = model(img_input)
            log_probs = torch.nn.functional.log_softmax(
                output, dim=2
            )
            pred = decode_prediction(log_probs[:, 0, :])

            gt = ''.join(
                idx_to_char.get(idx.item(), '')
                for idx in label_t
            )

            match = '✅' if pred == gt else '❌'
            print(f"{gt:<15} {pred:<15} {match}")
            shown += 1
            if shown >= 10:
                break

    return plate_acc, char_acc


# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════
if __name__ == '__main__':
    # Train
    model, history = train()

    # Test
    print("\n" + "="*60)
    print("RUNNING TEST EVALUATION")
    print("="*60)
    test_model()