"""
=================================================================
ANPR SYSTEM v11.9 — FIXED EDITION
=================================================================
Pipeline (exact flowchart order):

  1. Low Light Check  - auto-detect dark images
  2. YOLOv8           - detect plate (enhanced input if dark)
  3. Perspective      - correct skew / tilt
  4. Multi-Preprocess - Plain/Grey/Binary/CLAHE/Sharp/Night/Bloom
  5. PaddleOCR + EasyOCR - dual engine reading
  6. Voting System    - weighted confidence merge
  7. 65-Pair Resolver - confusion correction (RESTORED v11.3)
  8. Regex Validation - Indian plate format check
  9. Final Output     - display + CSV save

FIXES IN v11.9:
  - RESTORED ConfusionResolver from v11.3
  - PROTECTED_FROM_DIGIT: Removed C, H, S
  - _get_protected_positions: Restored Pattern 2 (LLDDLDDDD)
  - _fix_format: Penalty restored to -1 (from -5)
  - _fix_weak: Window ±2, threshold 3 (from ±3, 4)
  - _fix_isolated: Run-length ≥2 (from ≥3)
  - DISABLED: DegradedImageRestorer (false positives)
  - DISABLED: TwoLinePlateHandler (false triggers)
  - REMOVED: Red text variant (noise in voting)

INSTALL:
  pip install paddlepaddle==2.6.2 paddleocr==2.8.1
  pip install easyocr ultralytics
  pip install opencv-python pillow numpy==1.26.4 torch

FILES NEEDED:
  best.pt  <- your YOLO model (place in same folder)
=================================================================
"""

import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button, Frame, StringVar
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import csv
import os
import re
import time
import warnings
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── CONFIG ───────────────────────────────
MODEL_PATH        = 'best.pt'
EXCEL_FILENAME    = 'Plate_Logs.csv'
FAILURE_LOG_FILE  = 'Failure_Log.csv'
EASYOCR_MODEL_DIR = './ocr_models'
PLATE_W           = 400
PLATE_H           = 120
MIN_CHARS         = 4
MAX_CHARS         = 12
CAM_WIDTH         = 1280
CAM_HEIGHT        = 720


# ══════════════════════════════════════════
# STEP 1 — LOW LIGHT ENHANCER
# ══════════════════════════════════════════
class LowLightEnhancer:

    @staticmethod
    def is_dark(bgr, threshold=80):
        gray       = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        avg        = np.mean(gray)
        dark_ratio = np.sum(gray < 60) / gray.size
        is_low     = (avg < threshold or dark_ratio > 0.6)
        if is_low:
            print(f"  [LIGHT] Dark image detected: avg={avg:.0f}, "
                  f"dark_ratio={dark_ratio:.0%}")
        return is_low, avg

    @staticmethod
    def enhance(bgr):
        is_low, brightness = LowLightEnhancer.is_dark(bgr)
        if not is_low:
            return bgr

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clip = 4.0 if brightness < 50 else 3.0
        l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        if brightness < 30:
            gamma = 3.0
        elif brightness < 50:
            gamma = 2.5
        elif brightness < 70:
            gamma = 2.0
        else:
            gamma = 1.5

        table = np.array([
            np.clip(((i / 255.0) ** (1.0 / gamma)) * 255, 0, 255)
            for i in range(256)
        ], dtype=np.uint8)
        enhanced = cv2.LUT(enhanced, table)

        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        enhanced = np.clip(cv2.filter2D(enhanced, -1, kernel), 0, 255).astype(np.uint8)

        return enhanced

    @staticmethod
    def enhance_for_yolo(bgr):
        is_low, brightness = LowLightEnhancer.is_dark(bgr)
        if not is_low:
            return bgr

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        gamma = 1.8 if brightness < 50 else 1.5
        table = np.array([
            np.clip(((i / 255.0) ** (1.0 / gamma)) * 255, 0, 255)
            for i in range(256)
        ], dtype=np.uint8)
        return cv2.LUT(enhanced, table)


# ══════════════════════════════════════════
# STEP 3 — PERSPECTIVE CORRECTION
# ══════════════════════════════════════════
class PerspectiveCorrector:

    @staticmethod
    def correct(bgr):
        corrected = PerspectiveCorrector._four_point(bgr)
        if corrected is not None:
            return corrected
        return PerspectiveCorrector._deskew(bgr)

    @staticmethod
    def _deskew(bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        pts = np.column_stack(np.where(th > 0))
        if len(pts) < 10:
            return bgr
        angle = cv2.minAreaRect(pts)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 0.5 or abs(angle) > 20:
            return bgr
        h, w = bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def _four_point(bgr):
        try:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return None
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts[:5]:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype(np.float32)
                    return PerspectiveCorrector._warp(bgr, pts)
        except Exception:
            pass
        return None

    @staticmethod
    def _warp(bgr, pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        wA = np.linalg.norm(br - bl)
        wB = np.linalg.norm(tr - tl)
        hA = np.linalg.norm(tr - br)
        hB = np.linalg.norm(tl - bl)
        W = max(int(wA), int(wB), 1)
        H = max(int(hA), int(hB), 1)
        dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(bgr, M, (W, H))


# ══════════════════════════════════════════
# STEP 4 — MULTI-PREPROCESSING (FIXED)
# Removed red variant to reduce noise
# ══════════════════════════════════════════
class MultiPreprocessor:

    TARGET_W = PLATE_W
    TARGET_H = PLATE_H

    @staticmethod
    def reduce_bloom(bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l_c = clahe.apply(l)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        eroded = cv2.erode(l_c, kernel, iterations=1)
        return cv2.cvtColor(cv2.merge([eroded, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def standardise(bgr):
        h, w = bgr.shape[:2]
        scale = MultiPreprocessor.TARGET_W / max(w, 1)
        nh = max(int(h * scale), MultiPreprocessor.TARGET_H)
        return cv2.resize(bgr, (MultiPreprocessor.TARGET_W, nh),
                          interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def all_variants(bgr):
        std = MultiPreprocessor.standardise(bgr)
        grey = cv2.cvtColor(std, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gc = clahe.apply(grey)
        blur = cv2.GaussianBlur(gc, (3, 3), 0)

        plain = std.copy()
        grey_bgr = cv2.cvtColor(gc, cv2.COLOR_GRAY2BGR)

        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)

        adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 31, 8)
        adaptive = cv2.cvtColor(adapt, cv2.COLOR_GRAY2BGR)

        k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharp = np.clip(cv2.filter2D(std, -1, k), 0, 255).astype(np.uint8)

        gamma = 1.8
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                          for i in range(256)], dtype=np.uint8)
        night = cv2.LUT(std, table)

        bloom = MultiPreprocessor.reduce_bloom(std)

        # REMOVED: red variant — was causing noise in voting
        variants = {
            'plain': plain,
            'binary': binary,
            'sharp': sharp,
            'grey': grey_bgr,
            'adaptive': adaptive,
            'night': night,
            'bloom': bloom,
        }

        avg_bright = np.mean(cv2.cvtColor(std, cv2.COLOR_BGR2GRAY))
        if avg_bright < 100:
            gamma2 = 2.5
            table2 = np.array([
                np.clip(((i / 255.0) ** (1.0 / gamma2)) * 255, 0, 255)
                for i in range(256)
            ], dtype=np.uint8)
            variants['night_strong'] = cv2.LUT(std, table2)

            lab2 = cv2.cvtColor(std, cv2.COLOR_BGR2LAB)
            l2, a2, b2 = cv2.split(lab2)
            cl2 = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
            variants['night_clahe'] = cv2.cvtColor(
                cv2.merge([cl2.apply(l2), a2, b2]), cv2.COLOR_LAB2BGR)

        return variants

    @staticmethod
    def with_border(bgr):
        return cv2.copyMakeBorder(bgr, 20, 20, 20, 20,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))

    @staticmethod
    def upscale(bgr, factor=2):
        h, w = bgr.shape[:2]
        return cv2.resize(bgr, (w * factor, h * factor),
                          interpolation=cv2.INTER_CUBIC)


# ══════════════════════════════════════════
# STEP 5A — EASYOCR ENGINE
# ══════════════════════════════════════════
class EasyOCREngine:

    def __init__(self, reader):
        self.reader = reader

    def read(self, variants):
        if not self.reader:
            return []
        results = []
        for name, img in variants.items():
            for attempt, label in [
                (MultiPreprocessor.with_border(img), f"easy-{name}"),
                (MultiPreprocessor.with_border(MultiPreprocessor.upscale(img)),
                 f"easy-{name}-2x"),
            ]:
                r = self._read_one(attempt, label)
                if r:
                    results.append(r)
        return results

    def _read_one(self, bgr, label):
        try:
            res = self.reader.readtext(
                bgr,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                detail=1, paragraph=False,
                text_threshold=0.2, low_text=0.2,
                link_threshold=0.1, mag_ratio=1.5, min_size=5,
            )
            if not res:
                return None

            texts = []
            for (_, t, c) in res:
                cl = ''.join(x for x in t.upper() if x.isalnum())
                if cl:
                    texts.append((cl, c))

            if not texts:
                return None

            best_seg, best_conf = "", 0.0
            for seg, conf in texts:
                if MIN_CHARS <= len(seg) <= MAX_CHARS and conf > best_conf:
                    best_seg, best_conf = seg, conf

            if best_seg:
                return (best_seg, best_conf, label)

            combined = ''.join(t for t, c in texts)
            conf = sum(c for _, c in texts) / len(texts)
            if len(combined) > MAX_CHARS:
                combined = combined[:MAX_CHARS]
            if len(combined) < MIN_CHARS:
                return None
            return (combined, conf, label)

        except Exception:
            return None


# ══════════════════════════════════════════
# STEP 5B — PADDLEOCR ENGINE
# ══════════════════════════════════════════
class PaddleOCREngine:

    def __init__(self):
        self.ocr = None
        self.loaded = False
        self._error_count = 0
        self._MAX_LOG = 10

    def load(self):
        try:
            from paddleocr import PaddleOCR
            print("  Loading PaddleOCR...")
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=False,
                show_log=False,
                det_db_thresh=0.3,
                det_db_box_thresh=0.4,
            )
            self.loaded = True
            if self._verify():
                print("[OK] PaddleOCR — verified working")
            else:
                print("[OK] PaddleOCR — init OK (verify inconclusive)")
            return True
        except Exception as e:
            print(f"[FAIL] PaddleOCR: {e}")
            self.loaded = False
            return False

    def _verify(self):
        try:
            test = np.ones((80, 300, 3), dtype=np.uint8) * 255
            cv2.putText(test, "AB12CD3456", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
            result = self.ocr.ocr(test, cls=True)
            return bool(result and result[0])
        except Exception as e:
            print(f"  [PADDLE verify] {e}")
            return False

    def read(self, variants):
        if not self.loaded or self.ocr is None:
            return []
        results = []
        for name, img in variants.items():
            for attempt, label in [
                (MultiPreprocessor.with_border(img), f"paddle-{name}"),
                (MultiPreprocessor.with_border(MultiPreprocessor.upscale(img)),
                 f"paddle-{name}-2x"),
            ]:
                r = self._read_one(attempt, label)
                if r:
                    results.append(r)
        return results

    def _read_one(self, bgr, label):
        try:
            result = self.ocr.ocr(bgr, cls=True)
            if not result or not result[0]:
                return None
            texts, confs = [], []
            for line in result[0]:
                try:
                    if not line or len(line) < 2:
                        continue
                    txt_info = line[1]
                    if not txt_info:
                        continue
                    raw = str(txt_info[0])
                    conf = float(txt_info[1]) if len(txt_info) > 1 else 0.0
                    clean = ''.join(x for x in raw.upper() if x.isalnum())
                    if clean:
                        texts.append(clean)
                        confs.append(conf)
                except Exception:
                    continue

            if not texts:
                return None
            combined = ''.join(texts)
            if len(combined) > MAX_CHARS:
                combined = combined[:MAX_CHARS]
            if len(combined) < MIN_CHARS:
                return None
            return (combined, sum(confs) / len(confs), label)

        except Exception as e:
            self._error_count += 1
            if self._error_count <= self._MAX_LOG:
                print(f"[PADDLE ERR] {label}: {e}")
            return None


# ══════════════════════════════════════════
# STEP 6 — VOTING SYSTEM
# ══════════════════════════════════════════
class VotingSystem:

    PADDLE_BOOST = 1.15
    EASY_BOOST = 1.0

    @staticmethod
    def merge(paddle_results, easy_results):
        all_reads = []
        for t, c, label in paddle_results:
            t_clean = ''.join(x for x in t.upper() if x.isalnum())
            if MIN_CHARS <= len(t_clean) <= MAX_CHARS:
                all_reads.append((t_clean, c * VotingSystem.PADDLE_BOOST, label))
        for t, c, label in easy_results:
            t_clean = ''.join(x for x in t.upper() if x.isalnum())
            if MIN_CHARS <= len(t_clean) <= MAX_CHARS:
                all_reads.append((t_clean, c * VotingSystem.EASY_BOOST, label))
        if not all_reads:
            return '', 0.0, 'no reads'
        if len(all_reads) == 1:
            t, c, l = all_reads[0]
            return t, c, l
        return VotingSystem._vote(all_reads)

    @staticmethod
    def _vote(reads):
        by_len = defaultdict(list)
        for t, c, l in reads:
            by_len[len(t)].append((t, c, l))

        best_len = max(by_len, key=lambda k: len(by_len[k]) * 3 +
                       sum(c for _, c, _ in by_len[k]))

        aligned = list(by_len[best_len])
        for ln, items in by_len.items():
            if ln != best_len and abs(ln - best_len) == 1:
                for t, c, l in items:
                    t_adj = (t + '?' * max(0, best_len - len(t)))[:best_len]
                    aligned.append((t_adj, c * 0.7, l + '-adj'))

        pos_votes = [defaultdict(float) for _ in range(best_len)]
        for text, conf, _ in aligned:
            for i, ch in enumerate(text):
                if i < best_len and ch.isalnum():
                    pos_votes[i][ch] += conf

        merged, char_confs, details = [], [], []
        for i in range(best_len):
            votes = pos_votes[i]
            if not votes:
                merged.append('?')
                char_confs.append(0.0)
                details.append('?')
                continue
            best_ch = max(votes, key=votes.get)
            total_w = sum(votes.values())
            wconf = votes[best_ch] / max(total_w, 0.01)
            merged.append(best_ch)
            char_confs.append(min(wconf, 1.0))
            details.append(f"{best_ch}({wconf:.0%})")

        text = ''.join(merged).replace('?', '')
        avg_c = sum(char_confs) / len(char_confs) if char_confs else 0.0
        sources = list(set(l for _, _, l in aligned))
        detail = (f"Voted from {len(reads)} reads | "
                  f"Sources: {', '.join(sources[:4])} | "
                  f"Chars: {' '.join(details)}")
        return text, avg_c, detail


# ══════════════════════════════════════════
# STEP 7 — 65-PAIR CONFUSION RESOLVER
# RESTORED FROM v11.3 — ALL FIXES APPLIED
# ══════════════════════════════════════════
class ConfusionResolver:
    """
    RESTORED v11.3 logic with all fixes:
    - PROTECTED_FROM_DIGIT: Removed C, H, S
    - _get_protected_positions: Pattern 2 restored
    - _fix_format: Penalty -1 (not -5)
    - _fix_weak: Window ±2, threshold 3
    - _fix_isolated: Run-length ≥2
    """

    DD_PAIRS = {
        ('0', '6'): 0.50, ('0', '8'): 0.55, ('0', '9'): 0.50,
        ('1', '7'): 0.55, ('1', '4'): 0.50, ('2', '7'): 0.40,
        ('3', '8'): 0.58, ('3', '9'): 0.48, ('4', '9'): 0.38,
        ('5', '6'): 0.52, ('6', '8'): 0.45, ('6', '9'): 0.55,
        ('7', '9'): 0.35, ('8', '9'): 0.42, ('2', '3'): 0.38,
    }

    DL_PAIRS = {
        ('0', 'O'): 0.97, ('0', 'D'): 0.85, ('0', 'Q'): 0.75,
        ('0', 'C'): 0.45, ('1', 'I'): 0.93, ('1', 'L'): 0.85,
        ('1', 'T'): 0.55, ('1', 'J'): 0.50, ('2', 'Z'): 0.82,
        ('2', 'S'): 0.40, ('3', 'B'): 0.50, ('4', 'A'): 0.72,
        ('5', 'S'): 0.84, ('5', 'Z'): 0.40, ('6', 'G'): 0.78,
        ('7', 'T'): 0.73, ('8', 'B'): 0.87, ('8', 'R'): 0.35,
        ('9', 'Q'): 0.60, ('9', 'G'): 0.55,
    }

    LL_PAIRS = {
        ('A', 'H'): 0.50, ('A', 'R'): 0.40, ('B', 'R'): 0.45,
        ('B', 'D'): 0.62, ('C', 'G'): 0.65, ('C', 'O'): 0.55,
        ('D', 'O'): 0.75, ('D', 'Q'): 0.55, ('E', 'F'): 0.62,
        ('F', 'P'): 0.50, ('G', 'Q'): 0.50, ('H', 'N'): 0.70,
        ('H', 'K'): 0.40, ('I', 'J'): 0.55, ('I', 'T'): 0.72,
        ('I', 'L'): 0.55, ('K', 'X'): 0.50, ('M', 'N'): 0.80,
        ('N', 'H'): 0.70, ('P', 'R'): 0.58, ('Q', 'O'): 0.70,
        ('S', 'Z'): 0.45, ('T', 'Y'): 0.62, ('U', 'V'): 0.68,
        ('V', 'Y'): 0.60, ('W', 'V'): 0.45, ('W', 'U'): 0.40,
        ('W', 'N'): 0.72, ('W', 'M'): 0.80, ('W', 'K'): 0.38,
        ('W', 'O'): 0.45, ('A', 'M'): 0.60, ('L', 'I'): 0.55,
        ('X', 'K'): 0.50, ('Y', 'V'): 0.60, ('Z', 'S'): 0.45,
        ('M','H'): 0.65,
    }

    ALL = {}
    ALL.update(DD_PAIRS)
    ALL.update(DL_PAIRS)
    ALL.update(LL_PAIRS)

    # Letter to Digit mapping
    L2D = {
        'O': '0', 'I': '1', 'B': '8', 'S': '5', 'Z': '2', 'G': '6',
        'T': '7', 'A': '4', 'Q': '0', 'D': '0', 'C': '0', 'P': '9',
        'R': '8', 'H': '4', 'U': '0',
    }

    # Digit to Letter mapping
    D2L = {
        '0': 'O', '1': 'I', '2': 'Z', '3': 'B', '4': 'A',
        '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'Q',
    }

    # RESTORED: Removed C, H, S — they were breaking digit zone fixes
    PROTECTED_FROM_DIGIT = {
        'J',  # JK, JH, valid series
        'K',  # KA, KL, valid series
        'E',  # valid series letter
        'F',  # valid series letter
        'M',  # MH, MN, ML, MP, MZ state codes
        'N',  # NL, valid series
        'V',  # valid series letter
        'W',  # WB (West Bengal)
        'X',  # valid series letter
        'Y',  # valid series letter
        'L',  # LA (Ladakh), valid series
    }

    FORMATS = [
        "LLDDLLDDDD", "LLDDLLDD", "LLDDLDDDD", "LLDDLDDD",
        "LLDDLLLD", "LLDDLLD", "LLDDLD", "LLLDDDD",
        "LLDDLLL", "LLLDDDL", "DDDLLLL", "DDDDLLDD",
        "LLDDDDLL", "LLDDDLL", "LLLDDL", "LDDDDL",
        "LLDDDDD", "DDDLLL", "DDLLDDDD", "LDDDLLL",
        "LLLLDDDD", "DDDDLLLL", "LDDLLL", "LLLDDD",
        "DDDLL", "LLDDD", "DDDDLL", "LLDDDD",
        "LDDDLL", "LLLDDDD", "DDLLLDD", "LLLDDDDD",
        "LLDDDDDD", "LLDDDDDDD", "LLDDDDDDDD", "LLDDDDD",
        "LLDLDDDD", "LLDLDD", "LLDLLLDDDD", "LLDLLDDDD",
        "LLDLLDDD", "LLDLDDDDD","LLDDLLLDDDD"
    ]

    INDIAN_STATES = {
        'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN',
        'GA', 'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD',
        'MH', 'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'OR', 'PB', 'PY',
        'RJ', 'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB',
    }

    @staticmethod
    def _get_protected_positions(text):
        """
        RESTORED from v11.3 — full pattern coverage.
        Protects series letter positions from digit conversion.
        """
        n = len(text)
        protected = set()

        if n < 6:
            return protected

        # Pattern 1: LLDDLLDDDD (standard new format)
        # Example: MH12AB1234
        if (text[0].isalpha() and text[1].isalpha() and
                text[2].isdigit() and text[3].isdigit()):

            # Positions 4,5 are series letters — PROTECT
            if n > 4 and text[4].isalpha():
                protected.add(4)
            if n > 5 and text[5].isalpha():
                protected.add(5)

            # Some states use 3-letter series (position 6)
            if n > 7 and text[6].isalpha() and text[7].isdigit():
                protected.add(6)

        # Pattern 2: LLDDLDDDD (single-letter series + 4 digits) — RESTORED!
        # Example: DL8C1234 — position 4 is series letter
        elif (n >= 9 and
              text[0].isalpha() and text[1].isalpha() and
              text[2].isdigit() and text[3].isdigit() and
              text[4].isalpha()):

            protected.add(4)  # Series letter

            # Check if positions 5,6 are also series
            if n > 5 and text[5].isalpha():
                protected.add(5)
            if n > 6 and text[6].isalpha():
                protected.add(6)

        return protected

    @staticmethod
    def resolve(text):
        if not text or len(text) < MIN_CHARS:
            return text, []
        text = text.upper().strip()
        corr = []
        text, c1 = ConfusionResolver._fix_format(text)
        corr.extend(c1)
        text, c2 = ConfusionResolver._fix_strong(text)
        corr.extend(c2)
        text, c3 = ConfusionResolver._fix_weak(text)
        corr.extend(c3)
        text, c4 = ConfusionResolver._fix_state(text)
        corr.extend(c4)
        text, c5 = ConfusionResolver._fix_isolated(text)
        corr.extend(c5)
        return text, corr

    @staticmethod
    def _fix_format(text):
        """RESTORED: Penalty -1 for protected letters (not -5)"""
        best, bs, bc = text, -9999, []
        for fmt in ConfusionResolver.FORMATS:
            if len(fmt) != len(text):
                continue
            cand, score, chg = list(text), 0, []
            for i, (ch, exp) in enumerate(zip(text, fmt)):
                if exp == 'L':
                    if ch.isalpha():
                        score += 3
                    elif ch.isdigit():
                        sub = ConfusionResolver.D2L.get(ch)
                        if sub:
                            cand[i] = sub
                            score += 1
                            chg.append(f"p{i}:{ch}>{sub}")
                        else:
                            score -= 2
                elif exp == 'D':
                    if ch.isdigit():
                        score += 3
                    elif ch.isalpha():
                        if ch in ConfusionResolver.PROTECTED_FROM_DIGIT:
                            score -= 1  # RESTORED from -5
                        else:
                            sub = ConfusionResolver.L2D.get(ch)
                            if sub:
                                cand[i] = sub
                                score += 1
                                chg.append(f"p{i}:{ch}>{sub}")
                            else:
                                score -= 2
            if score > bs:
                bs = score
                best = "".join(cand)
                bc = chg
        return (best, bc) if bs > len(text) else (text, [])

    @staticmethod
    def _fix_strong(text):
        """Fix characters surrounded by opposite type (DxD or LxL)"""
        chars = list(text)
        corr = []
        n = len(chars)
        protected_pos = ConfusionResolver._get_protected_positions(text)

        for i in range(1, n - 1):
            ch = chars[i]
            p, nx = chars[i - 1], chars[i + 1]

            if ch.isalpha() and p.isdigit() and nx.isdigit():
                if i in protected_pos:
                    continue
                if ch in ConfusionResolver.PROTECTED_FROM_DIGIT:
                    continue
                sub = ConfusionResolver.L2D.get(ch)
                if sub:
                    chars[i] = sub
                    corr.append(f"p{i}:{ch}>{sub}(DxD)")

            elif ch.isdigit() and p.isalpha() and nx.isalpha():
                sub = ConfusionResolver.D2L.get(ch)
                if sub:
                    chars[i] = sub
                    corr.append(f"p{i}:{ch}>{sub}(LxL)")

        return "".join(chars), corr

    @staticmethod
    def _fix_weak(text):
        """RESTORED: Window ±2, threshold 3 (not ±3, 4)"""
        chars = list(text)
        corr = []
        n = len(chars)
        protected_pos = ConfusionResolver._get_protected_positions(text)

        for i in range(n):
            ch = chars[i]

            # RESTORED: ±2 window (not ±3)
            d = sum(1 for j in range(max(0, i - 2), min(n, i + 3))
                    if j != i and chars[j].isdigit())
            l = sum(1 for j in range(max(0, i - 2), min(n, i + 3))
                    if j != i and chars[j].isalpha())

            # RESTORED: threshold 3 (not 4)
            if ch.isalpha() and d >= 3 and l == 0:
                if i in protected_pos:
                    continue
                if ch in ConfusionResolver.PROTECTED_FROM_DIGIT:
                    continue
                sub = ConfusionResolver.L2D.get(ch)
                if sub:
                    chars[i] = sub
                    corr.append(f"p{i}:{ch}>{sub}(wD)")

            elif ch.isdigit() and l >= 3 and d == 0:
                sub = ConfusionResolver.D2L.get(ch)
                if sub:
                    chars[i] = sub
                    corr.append(f"p{i}:{ch}>{sub}(wL)")

        return "".join(chars), corr

    @staticmethod
    def _fix_state(text):
        """Fix state code (first 2 characters)"""
        corr, chars = [], list(text)
        if len(text) < 4:
            return text, corr

        # Convert digits to letters in state code positions
        for pos in range(2):
            if chars[pos].isdigit():
                sub = ConfusionResolver.D2L.get(chars[pos])
                if sub:
                    chars[pos] = sub

        text = "".join(chars)
        f2 = text[:2]

        if f2 in ConfusionResolver.INDIAN_STATES:
            return text, corr

        # Build swap map for similar letters
        SWAPS = defaultdict(set)
        for (a, b), s in ConfusionResolver.LL_PAIRS.items():
            if s >= 0.35:
                SWAPS[a].add(b)
                SWAPS[b].add(a)

        # Find best matching state code
        best, bs = None, 0
        for pos in range(2):
            for sw in SWAPS.get(f2[pos], set()):
                if not sw.isalpha():
                    continue
                cand = f2[:pos] + sw + f2[pos + 1:]
                if cand in ConfusionResolver.INDIAN_STATES:
                    sim = ConfusionResolver.ALL.get(
                        (f2[pos], sw),
                        ConfusionResolver.ALL.get((sw, f2[pos]), 0)
                    )
                    if sim > bs:
                        bs = sim
                        best = (pos, f2[pos], sw)

        if best and bs > 0.35:
            pos, old, new = best
            chars = list(text)
            chars[pos] = new
            text = "".join(chars)
            corr.append(f"state:{old}>{new}")

        return text, corr

    @staticmethod
    def _fix_isolated(text):
        """RESTORED: Run-length ≥2 (not ≥3)"""
        chars, corr, n = list(text), [], len(text)
        protected_pos = ConfusionResolver._get_protected_positions(text)

        # Build runs
        runs, i = [], 0
        while i < n:
            t = 'L' if chars[i].isalpha() else 'D'
            j = i
            while j < n:
                if ('L' if chars[j].isalpha() else 'D') != t:
                    break
                j += 1
            runs.append((i, j, t))
            i = j

        for idx, (s, e, rt) in enumerate(runs):
            if e - s != 1 or s in protected_pos:
                continue

            pr = runs[idx - 1] if idx > 0 else None
            nr = runs[idx + 1] if idx < len(runs) - 1 else None

            # RESTORED: run-length ≥2 (not ≥3)
            if not (pr and pr[2] != rt and (pr[1] - pr[0]) >= 2
                    and nr and nr[2] != rt
                    and (nr[1] - nr[0]) >= 2):
                continue

            ch = chars[s]
            if rt == 'L':
                if ch in ConfusionResolver.PROTECTED_FROM_DIGIT:
                    continue
                sub = ConfusionResolver.L2D.get(ch)
                if sub:
                    chars[s] = sub
                    corr.append(f"p{s}:{ch}>{sub}(iso)")
            else:
                sub = ConfusionResolver.D2L.get(ch)
                if sub:
                    chars[s] = sub
                    corr.append(f"p{s}:{ch}>{sub}(iso)")

        return "".join(chars), corr


# ══════════════════════════════════════════
# STEP 8 — REGEX VALIDATION
# ══════════════════════════════════════════
class RegexValidator:

    PATTERNS = [
        r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$',
        r'^[A-Z]{2}\d{2}[A-Z]{1}\d{4}$',
        r'^[A-Z]{2}\d{2}[A-Z]{3}\d{4}$',
        r'^[A-Z]{2}\d{2}[A-Z]{2}\d{3}$',
        r'^[A-Z]{2}\d{1}[A-Z]{2}\d{4}$',
        r'^[A-Z]{2}\d{2}[A-Z]{2}\d{2}$',
        r'^[A-Z]{2}\d{2}\d{4}$',
        r'^[A-Z]{2}\d{4}[A-Z]{2}$',
        r'^[A-Z]{2}\d{2}[A-Z]{1}\d{3}$',
        r'^[A-Z]{2}\d{2}[A-Z]{0,3}\d{1,6}$',
        r'^[A-Z]{2}\d{1}[A-Z]{1}[A-Z]{2}\d{4}$',
        r'^[A-Z]{2}\d{1}[A-Z]{1}[A-Z]{1}\d{4}$',
        r'^[A-Z]{2}\d{1}[A-Z]{1}[A-Z]{2}\d{3}$',
    ]

    @staticmethod
    def clean(text):
        return ''.join(c for c in text.upper() if c.isalnum())

    @staticmethod
    def validate(text):
        text = RegexValidator.clean(text)
        if not text or len(text) < MIN_CHARS:
            return False, 0.0, 'too_short'
        for pattern in RegexValidator.PATTERNS:
            if re.match(pattern, text):
                return True, 0.10, pattern
        if len(text) >= 2 and text[:2] in ConfusionResolver.INDIAN_STATES:
            return False, 0.05, 'valid_state_prefix'
        return False, 0.0, 'no_match'

    @staticmethod
    def best_of(candidates):
        scored = []
        for text, conf, detail in candidates:
            t = RegexValidator.clean(text)
            valid, boost, fmt = RegexValidator.validate(t)
            scored.append((t, conf + boost, detail, valid, fmt))
        if not scored:
            return '', 0.0, 'no candidates'
        valid_ones = [s for s in scored if s[3]]
        pool = valid_ones if valid_ones else scored
        best = max(pool, key=lambda x: x[1])
        return best[0], best[1], best[2]


# ══════════════════════════════════════════
# MAIN PIPELINE (FIXED)
# DISABLED: DegradedImageRestorer, TwoLinePlateHandler
# ══════════════════════════════════════════
class ANPRPipeline:

    def __init__(self, easy_engine, paddle_engine):
        self.easy = easy_engine
        self.paddle = paddle_engine

    def run(self, plate_bgr, is_camera=False):
        if plate_bgr is None or plate_bgr.size == 0:
            return '', 0.0, 'empty input'

        # DISABLED: DegradedImageRestorer — was causing false positives
        # plate_bgr, fixes = DegradedImageRestorer.restore(plate_bgr)
        fixes_str = 'disabled'

        # Low light enhancement (kept — this works well)
        is_dark, brightness = LowLightEnhancer.is_dark(plate_bgr)
        if is_dark:
            plate_bgr = LowLightEnhancer.enhance(plate_bgr)
            print(f"  [OCR] Enhanced dark plate (brightness={brightness:.0f})")

        if is_camera:
            plate_bgr = self._enhance_camera(plate_bgr)

        # DISABLED: TwoLinePlateHandler — false triggers on normal plates
        # if TwoLinePlateHandler.is_two_line(plate_bgr):
        #     return TwoLinePlateHandler.read(plate_bgr, self.easy, self.paddle)

        corrected = PerspectiveCorrector.correct(plate_bgr)
        variants = MultiPreprocessor.all_variants(corrected)
        orig_variants = MultiPreprocessor.all_variants(plate_bgr)

        easy_results = []
        paddle_results = []

        if self.easy:
            easy_results += self.easy.read(variants)
            easy_results += self.easy.read(orig_variants)

        if self.paddle:
            paddle_results += self.paddle.read(variants)
            paddle_results += self.paddle.read(orig_variants)

        print(f"  [OCR] Paddle={len(paddle_results)} Easy={len(easy_results)}")

        voted_text, voted_conf, vote_detail = VotingSystem.merge(
            paddle_results, easy_results
        )

        if not voted_text:
            return '', 0.0, 'no OCR output'

        resolved, corrections = ConfusionResolver.resolve(voted_text)

        candidates = [(voted_text, voted_conf, 'voted')]
        if resolved and resolved != voted_text:
            candidates.append((resolved, voted_conf + 0.05, 'resolved'))

        all_reads = paddle_results + easy_results
        for t, c, l in sorted(all_reads, key=lambda x: x[1], reverse=True)[:5]:
            r, _ = ConfusionResolver.resolve(t)
            candidates.append((r, c, l))

        final_text, final_conf, final_src = RegexValidator.best_of(candidates)

        is_valid, _, fmt = RegexValidator.validate(final_text)
        fix_str = '; '.join(corrections) if corrections else 'none'
        detail = (
            f"Pipeline: {final_src}\n"
            f"Restored: {fixes_str}\n"
            f"Voted: '{voted_text}' ({voted_conf:.1%})\n"
            f"Resolved: '{resolved}'\n"
            f"Fixes: {fix_str}\n"
            f"Valid format: {'YES - ' + fmt if is_valid else 'NO'}\n"
            f"Reads: Paddle={len(paddle_results)} Easy={len(easy_results)}"
        )
        return final_text, min(final_conf, 1.0), detail

    def _enhance_camera(self, bgr):
        h, w = bgr.shape[:2]
        if w < 300:
            s = 300 / w
            bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        den = cv2.fastNlMeansDenoisingColored(bgr, None, 8, 8, 7, 21)
        k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sh = np.clip(cv2.filter2D(den, -1, k), 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(sh, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(3.0, (8, 8))
        return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


# ══════════════════════════════════════════
# FAILURE LOGGER
# ══════════════════════════════════════════
class FailureLogger:

    COLUMNS = [
        "timestamp", "source", "predicted", "correct_text",
        "verdict", "confidence", "valid_format", "condition",
        "paddle_reads", "easy_reads", "voted", "detail",
    ]

    @staticmethod
    def save(predicted, correct_text, verdict, conf, valid, detail, source):
        condition = "UNKNOWN"
        paddle_reads = ""
        easy_reads = ""
        voted = ""
        if detail:
            for line in detail.split('\n'):
                line = line.strip()
                if line.startswith("Condition"):
                    condition = line.split(':', 1)[-1].strip()
                elif line.startswith("Voted"):
                    voted = line.split(':', 1)[-1].strip()
                elif line.startswith("Reads"):
                    parts = line.split(':', 1)[-1].strip()
                    for p in parts.split():
                        if p.startswith("Paddle="):
                            paddle_reads = p.split('=')[1]
                        elif p.startswith("Easy="):
                            easy_reads = p.split('=')[1]

        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source, predicted, correct_text, verdict,
            f"{conf * 100:.1f}%",
            "YES" if valid else "NO",
            condition, paddle_reads, easy_reads, voted,
            (detail[:200] if detail else "").replace('\n', ' | '),
        ]

        for _ in range(3):
            try:
                new = not os.path.isfile(FAILURE_LOG_FILE)
                with open(FAILURE_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(FailureLogger.COLUMNS)
                    w.writerow(row)
                return True
            except PermissionError:
                time.sleep(0.3)
            except Exception as e:
                print(f"[LOG ERR] {e}")
                return False
        return False


# ══════════════════════════════════════════
# PHONE CAMERA
# ══════════════════════════════════════════
class PhoneCamera:
    SETUP_TEXT = """
+----------------------------------------------+
|         PHONE CAMERA SETUP GUIDE             |
+----------------------------------------------+
|  OPTION A: DroidCam (RECOMMENDED)            |
|  1. Install DroidCam on phone (Play Store)   |
|  2. Install DroidCam Client on laptop        |
|  3. Connect via USB, click Start             |
|  4. Phone appears as camera index 1 or 2     |
|                                              |
|  OPTION B: IP Webcam (WiFi)                  |
|  1. Install IP Webcam on phone               |
|  2. Start Server, note the URL               |
|  3. Enter URL below                          |
+----------------------------------------------+
"""

    @staticmethod
    def find_cameras():
        available = []
        for i in range(6):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available.append((i, f"Camera {i} ({w}x{h})"))
                cap.release()
            else:
                cap.release()
        return available

    @staticmethod
    def open_url(url):
        for v in [url, f"http://{url}/video",
                  f"http://{url}:8080/video",
                  f"http://{url}:4747/video"]:
            try:
                cap = cv2.VideoCapture(v)
                if cap.isOpened():
                    return cap
                cap.release()
            except Exception:
                pass
        return None


# ══════════════════════════════════════════
# GUI APPLICATION
# ══════════════════════════════════════════
class CustomANPR:

    def __init__(self, root):
        self.root = root
        self.root.title(
            "ANPR v11.9 - FIXED | Night + "
            "PaddleOCR + EasyOCR + Voting + 65-Pair + Regex"
        )
        self.root.geometry("1300x900")
        self.root.configure(bg="#0c0c0c")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.yolo = None
        self.easy_reader = None
        self.easy_engine = None
        self.paddle_engine = PaddleOCREngine()
        self.pipeline = None

        self.current_image = None
        self.best_plate_crop = None
        self.cap = None
        self.is_webcam_running = False
        self.is_camera_mode = False
        self.is_frame_captured = False
        self.cancel_flag = threading.Event()
        self.task_id = 0
        self.lock = threading.Lock()
        self.result = ('', 0.0, '')
        self.pending_log = None

        self.setup_ui()
        threading.Thread(target=self.load_ai, daemon=True).start()

    def setup_ui(self):
        hdr = Frame(self.root, bg="#0a1628", height=52)
        hdr.pack(fill="x")
        Label(
            hdr,
            text="  ANPR v11.9 FIXED  |  Night + "
                 "PaddleOCR + EasyOCR -> Vote -> 65-Pair -> Regex",
            bg="#0a1628", fg="white",
            font=("Segoe UI", 10, "bold"),
        ).place(x=8, y=15)

        self.container = Frame(self.root, bg="#0c0c0c")
        self.container.pack(fill="both", expand=True)

        # HOME PAGE
        self.home_frame = Frame(self.container, bg="#0c0c0c")
        self.lbl_status = Label(
            self.home_frame, text="Initializing...",
            fg="#FFA500", bg="#0c0c0c",
            font=("Segoe UI", 15, "bold")
        )
        self.lbl_status.pack(pady=(40, 5))
        self.lbl_sub = Label(
            self.home_frame, text="",
            fg="#555", bg="#0c0c0c", font=("Segoe UI", 10)
        )
        self.lbl_sub.pack(pady=(0, 25))

        for txt, cmd, bg, attr in [
            ("  UPLOAD IMAGE", self.start_image, "#2962FF", "btn_img"),
            ("  LAPTOP CAMERA", lambda: self.start_camera(0), "#FF6D00", "btn_cam"),
            ("  PHONE CAMERA", self.start_phone_camera, "#00897B", "btn_phone"),
        ]:
            b = Button(
                self.home_frame, text=txt, command=cmd,
                bg=bg, fg="white",
                font=("Segoe UI", 13, "bold"),
                width=26, height=2, relief="flat",
                state="disabled"
            )
            b.pack(pady=8)
            setattr(self, attr, b)

        self.home_frame.pack(fill="both", expand=True)

        # SCANNER PAGE
        self.app_frame = Frame(self.container, bg="#0c0c0c")

        left = Frame(self.app_frame, bg="#0c0c0c")
        left.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        self.canvas = Label(
            left, bg="#161616", text="Waiting...",
            fg="#333", font=("Segoe UI", 12)
        )
        self.canvas.pack(fill="both", expand=True)

        self.cam_bar = Frame(left, bg="#1a1a2e", height=50)
        self.cam_bar.pack(fill="x", pady=(4, 0))
        self.cam_bar.pack_propagate(False)

        self.btn_capture = Button(
            self.cam_bar, text="CAPTURE",
            command=self.capture_frame,
            bg="#E65100", fg="white",
            font=("Segoe UI", 10, "bold"), relief="flat"
        )
        self.btn_capture.pack(side="left", fill="x", expand=True, padx=(8, 4), pady=8)

        self.btn_recapture = Button(
            self.cam_bar, text="RE-CAPTURE",
            command=self.recapture,
            bg="#37474F", fg="white",
            font=("Segoe UI", 9, "bold"), relief="flat"
        )
        self.btn_recapture.pack(side="left", padx=(4, 8), pady=8)

        self.lbl_cam_status = Label(
            self.cam_bar, text="", bg="#1a1a2e",
            fg="#FFA726", font=("Segoe UI", 8, "italic")
        )
        self.lbl_cam_status.pack(side="left", padx=8)

        # Right panel
        sb = Frame(self.app_frame, bg="#181818", width=480)
        sb.pack(side="right", fill="y", padx=(0, 8), pady=8)
        sb.pack_propagate(False)

        ctrl = Frame(sb, bg="#181818")
        ctrl.pack(fill="x", padx=10, pady=(10, 4))

        Button(
            ctrl, text="<- HOME", command=self.go_back,
            bg="#252525", fg="white",
            font=("Segoe UI", 9, "bold"),
            relief="flat", width=9
        ).pack(side="left", padx=(0, 6))

        self.btn_scan = Button(
            ctrl, text="READ PLATE",
            command=self.trigger_ocr,
            bg="#00C853", fg="white",
            font=("Segoe UI", 11, "bold"),
            relief="flat", height=2
        )
        self.btn_scan.pack(side="left", fill="x", expand=True)

        self.lbl_prog = Label(
            sb, text="", bg="#181818",
            fg="#FFA500", font=("Segoe UI", 8)
        )
        self.lbl_prog.pack()

        steps_f = Frame(sb, bg="#181818")
        steps_f.pack(fill="x", padx=10, pady=4)
        steps = [
            ("1.NIGHT", "#00BCD4"),
            ("2.YOLO", "#4CAF50"),
            ("3.PERSP", "#2196F3"),
            ("4.PREPROC", "#9C27B0"),
            ("5.PADDLE+EASY", "#FF9800"),
            ("6.VOTE", "#F44336"),
            ("7.65-PAIR", "#00BCD4"),
            ("8.REGEX", "#FFEB3B"),
            ("9.OUTPUT", "#69F0AE"),
        ]
        self.step_labels = {}
        for name, color in steps:
            lbl = Label(
                steps_f, text=name, bg="#222",
                fg="#555", font=("Consolas", 7, "bold"),
                padx=3, pady=2
            )
            lbl.pack(side="left", padx=1)
            self.step_labels[name] = (lbl, color)

        box = Frame(sb, bg="#0b1a0b", relief="ridge", bd=2)
        box.pack(fill="x", padx=10, pady=8)

        Label(
            box,
            text="FINAL OUTPUT — PaddleOCR + EasyOCR + Vote + Regex",
            bg="#0b1a0b", fg="#69F0AE",
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", padx=6, pady=(6, 0))

        self.lbl_plate = Label(
            box, text="---", bg="#0a3d0a", fg="white",
            font=("Courier New", 28, "bold"), height=1
        )
        self.lbl_plate.pack(fill="x", padx=6, pady=(4, 2))

        self.lbl_conf = Label(
            box, text="Confidence: ---",
            bg="#0b1a0b", fg="#69F0AE",
            font=("Arial", 10, "bold")
        )
        self.lbl_conf.pack(anchor="w", padx=6)

        self.lbl_valid = Label(
            box, text="",
            bg="#0b1a0b", fg="#FFD600",
            font=("Arial", 9, "bold")
        )
        self.lbl_valid.pack(anchor="w", padx=6)

        self.lbl_detail = Label(
            box, text="",
            bg="#0b1a0b", fg="#555",
            font=("Consolas", 6),
            wraplength=440, justify="left"
        )
        self.lbl_detail.pack(fill="x", padx=6, pady=(0, 2))

        self.btn_save = Button(
            box, text="SAVE TO CSV",
            command=self.save_result,
            bg="#1B5E20", fg="white",
            font=("Segoe UI", 9, "bold"),
            relief="flat", state="disabled", height=1
        )
        self.btn_save.pack(fill="x", padx=6, pady=(2, 8))

        # FAILURE LOGGER PANEL
        log_box = Frame(sb, bg="#1a1010", relief="ridge", bd=2)
        log_box.pack(fill="x", padx=10, pady=(0, 6))

        Label(
            log_box,
            text="Was this prediction correct?",
            bg="#1a1010", fg="#FF8A65",
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", padx=6, pady=(6, 2))

        Label(
            log_box,
            text="Mark every result — builds your failure dataset",
            bg="#1a1010", fg="#555",
            font=("Arial", 7, "italic")
        ).pack(anchor="w", padx=6)

        btn_row = Frame(log_box, bg="#1a1010")
        btn_row.pack(fill="x", padx=6, pady=4)

        self.btn_correct = Button(
            btn_row, text="CORRECT",
            command=self.mark_correct,
            bg="#1B5E20", fg="white",
            font=("Segoe UI", 9, "bold"),
            relief="flat", state="disabled", width=10
        )
        self.btn_correct.pack(side="left", padx=(0, 4))

        self.btn_wrong = Button(
            btn_row, text="WRONG",
            command=self.mark_wrong,
            bg="#B71C1C", fg="white",
            font=("Segoe UI", 9, "bold"),
            relief="flat", state="disabled", width=10
        )
        self.btn_wrong.pack(side="left", padx=(0, 4))

        self.btn_skip = Button(
            btn_row, text="SKIP",
            command=self.mark_skip,
            bg="#37474F", fg="white",
            font=("Segoe UI", 9, "bold"),
            relief="flat", state="disabled", width=8
        )
        self.btn_skip.pack(side="left")

        self.correct_frame = Frame(log_box, bg="#1a1010")
        self.correct_frame.pack(fill="x", padx=6, pady=(0, 2))

        Label(
            self.correct_frame,
            text="Type the correct plate number:",
            bg="#1a1010", fg="#FF8A65",
            font=("Arial", 8)
        ).pack(anchor="w")

        entry_row = Frame(self.correct_frame, bg="#1a1010")
        entry_row.pack(fill="x")

        self.correction_var = StringVar()
        self.entry_correction = tk.Entry(
            entry_row,
            textvariable=self.correction_var,
            font=("Courier New", 13, "bold"),
            bg="#2a1010", fg="white",
            insertbackground="white",
            relief="flat", width=16
        )
        self.entry_correction.pack(side="left", padx=(0, 6), ipady=4)

        self.btn_confirm_wrong = Button(
            entry_row, text="CONFIRM",
            command=self.confirm_wrong,
            bg="#B71C1C", fg="white",
            font=("Segoe UI", 8, "bold"),
            relief="flat"
        )
        self.btn_confirm_wrong.pack(side="left")
        self.correct_frame.pack_forget()

        self.lbl_log_status = Label(
            log_box, text="",
            bg="#1a1010", fg="#69F0AE",
            font=("Arial", 8, "bold")
        )
        self.lbl_log_status.pack(anchor="w", padx=6, pady=(0, 6))

        self.lbl_saved = Label(
            sb, text="", bg="#181818",
            fg="#00BCD4", font=("Arial", 9, "bold")
        )
        self.lbl_saved.pack(fill="x", padx=10, pady=2)

        self.lbl_dbg = Label(
            sb, text="", bg="#181818", fg="#444",
            font=("Consolas", 6),
            wraplength=460, justify="left"
        )
        self.lbl_dbg.pack(fill="x", padx=10, pady=(0, 4))

    def _step(self, name):
        for n, (lbl, color) in self.step_labels.items():
            if n == name:
                lbl.config(fg=color, bg="#333")
            else:
                lbl.config(fg="#555", bg="#222")

    def load_ai(self):
        try:
            self._sub("Step 2: Loading YOLOv8...")
            self.yolo = YOLO(MODEL_PATH)
            print("[OK] YOLO")

            self._sub("Step 5a: Loading PaddleOCR...")
            paddle_ok = self.paddle_engine.load()

            self._sub("Step 5b: Loading EasyOCR...")
            try:
                import easyocr
                os.makedirs(EASYOCR_MODEL_DIR, exist_ok=True)
                has = (os.path.exists(EASYOCR_MODEL_DIR)
                       and bool(os.listdir(EASYOCR_MODEL_DIR)))
                self.easy_reader = easyocr.Reader(
                    ['en'], gpu=True,
                    model_storage_directory=EASYOCR_MODEL_DIR,
                    download_enabled=not has,
                )
                self.easy_engine = EasyOCREngine(self.easy_reader)
                print("[OK] EasyOCR")
            except Exception as e:
                print(f"[WARN] EasyOCR: {e}")

            self.pipeline = ANPRPipeline(self.easy_engine, self.paddle_engine)
            self.root.after(0, lambda: self._ready(paddle_ok))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.lbl_status.config(
                text=f"Error: {e}", fg="red"))

    def _sub(self, t):
        self.root.after(0, lambda: self.lbl_sub.config(text=t))
        print(f"  {t}")

    def _ready(self, paddle_ok):
        parts = []
        if self.easy_reader:
            parts.append("EasyOCR")
        if paddle_ok:
            parts.append("PaddleOCR")
        parts += ["Voting", "65-Pair(FIXED)", "Regex", "NightVision"]
        self.lbl_status.config(text="ANPR v11.9 — Ready (FIXED)", fg="#00E676")
        self.lbl_sub.config(text=" + ".join(parts))
        for b in [self.btn_img, self.btn_cam, self.btn_phone]:
            b.config(state="normal")

    def cancel(self):
        with self.lock:
            self.cancel_flag.set()
            self.task_id += 1

    def cancelled(self, tid):
        with self.lock:
            return self.cancel_flag.is_set() or tid != self.task_id

    def go_back(self):
        self.cancel()
        self.stop_webcam()
        self.best_plate_crop = None
        self.is_frame_captured = False
        self.is_camera_mode = False
        self.reset_ui()
        self.app_frame.pack_forget()
        self.home_frame.pack(fill="both", expand=True)

    def reset_ui(self):
        self.lbl_plate.config(text="---")
        self.lbl_conf.config(text="Confidence: ---")
        self.lbl_valid.config(text="")
        self.lbl_detail.config(text="")
        self.lbl_saved.config(text="")
        self.lbl_dbg.config(text="")
        self.lbl_prog.config(text="")
        self.canvas.config(image="", text="Waiting...")
        self.btn_scan.config(state="normal")
        self.btn_save.config(state="disabled")
        self.result = ('', 0.0, '')
        self.pending_log = None
        self._reset_logger()
        for n, (lbl, _) in self.step_labels.items():
            lbl.config(fg="#555", bg="#222")

    def _reset_logger(self):
        self.btn_correct.config(state="disabled")
        self.btn_wrong.config(state="disabled")
        self.btn_skip.config(state="disabled")
        self.lbl_log_status.config(text="")
        self.correction_var.set("")
        self.correct_frame.pack_forget()

    def start_image(self):
        self.cancel()
        self.cancel_flag.clear()
        self.is_camera_mode = False
        self.home_frame.pack_forget()
        self.app_frame.pack(fill="both", expand=True)
        self.reset_ui()
        self.cam_bar.pack_forget()
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.current_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.run_yolo(self.current_image)
                return
        self.go_back()

    def start_camera(self, idx):
        self.cancel()
        self.cancel_flag.clear()
        self.is_camera_mode = True
        self.is_frame_captured = False
        self.home_frame.pack_forget()
        self.app_frame.pack(fill="both", expand=True)
        self.reset_ui()
        self.cam_bar.pack(fill="x", pady=(4, 0))
        self.btn_capture.config(state="normal")
        self.btn_recapture.config(state="disabled")
        self.btn_scan.config(state="disabled")
        self.lbl_cam_status.config(text="Live - position plate then CAPTURE")
        self.cap = cv2.VideoCapture(idx)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            self.is_webcam_running = True
            self.update_live()
        else:
            messagebox.showerror("Error", f"Cannot open camera {idx}")
            self.go_back()

    def start_camera_url(self, url):
        self.cancel()
        self.cancel_flag.clear()
        self.is_camera_mode = True
        self.is_frame_captured = False
        self.home_frame.pack_forget()
        self.app_frame.pack(fill="both", expand=True)
        self.reset_ui()
        self.cam_bar.pack(fill="x", pady=(4, 0))
        self.btn_capture.config(state="normal")
        self.btn_recapture.config(state="disabled")
        self.btn_scan.config(state="disabled")
        cap = PhoneCamera.open_url(url)
        if cap:
            self.cap = cap
            self.is_webcam_running = True
            self.lbl_cam_status.config(text="Phone connected - CAPTURE when ready")
            self.update_live()
        else:
            messagebox.showerror("Failed", f"Cannot connect: {url}")
            self.go_back()

    def start_phone_camera(self):
        d = tk.Toplevel(self.root)
        d.title("Phone Camera")
        d.geometry("500x400")
        d.configure(bg="#1a1a1a")
        d.transient(self.root)
        d.grab_set()
        Label(d, text="Phone Camera", bg="#1a1a1a", fg="white",
              font=("Segoe UI", 14, "bold")).pack(pady=(15, 5))
        txt = tk.Text(d, bg="#0a0a0a", fg="#aaa",
                      font=("Consolas", 8), height=10, width=55, relief="flat")
        txt.pack(padx=15, pady=5)
        txt.insert("1.0", PhoneCamera.SETUP_TEXT)
        txt.config(state="disabled")
        Label(d, text="Available Cameras:", bg="#1a1a1a",
              fg="#82B1FF", font=("Segoe UI", 10, "bold")).pack(pady=(8, 3))
        cams = PhoneCamera.find_cameras()
        cf = Frame(d, bg="#1a1a1a")
        cf.pack(fill="x", padx=15)
        if cams:
            for idx, name in cams:
                Button(cf, text=f"  {name}",
                       command=lambda i=idx: (d.destroy(), self.start_camera(i)),
                       bg="#00897B", fg="white",
                       font=("Segoe UI", 9, "bold"),
                       relief="flat", width=35).pack(pady=2)
        else:
            Label(cf, text="No cameras found.",
                  bg="#1a1a1a", fg="#FF8A65").pack(pady=5)
        Label(d, text="IP Webcam URL:", bg="#1a1a1a",
              fg="#FFD600", font=("Segoe UI", 9, "bold")).pack(pady=(8, 3))
        ipf = Frame(d, bg="#1a1a1a")
        ipf.pack(fill="x", padx=15)
        iv = StringVar(value="http://192.168.1.100:8080/video")
        tk.Entry(ipf, textvariable=iv, font=("Consolas", 10), width=35).pack(
            side="left", padx=(0, 5))
        Button(ipf, text="Connect",
               command=lambda: (d.destroy(), self.start_camera_url(iv.get())),
               bg="#FF6D00", fg="white",
               font=("Segoe UI", 9, "bold"), relief="flat").pack(side="left")
        Button(d, text="Cancel", command=d.destroy,
               bg="#333", fg="white", relief="flat").pack(pady=10)

    def update_live(self):
        if not self.is_webcam_running or self.is_frame_captured:
            return
        ret, frame = self.cap.read()
        if ret:
            self.current_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._show_live(self.current_image)
        self.root.after(33, self.update_live)

    def _show_live(self, arr):
        bgr_check = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        is_dark, _ = LowLightEnhancer.is_dark(bgr_check)
        if is_dark:
            bgr_check = LowLightEnhancer.enhance_for_yolo(bgr_check)
            arr = cv2.cvtColor(bgr_check, cv2.COLOR_BGR2RGB)

        vis = arr.copy()
        res = self.yolo(arr, conf=0.20, verbose=False)
        for r in res:
            for box in r.boxes:
                c = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(vis, (x1, y1), (x2, y2), (50, 255, 50), 3)
                cv2.putText(vis, f"Plate {c:.0%}",
                            (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)
        h, w = vis.shape[:2]
        sc = min(700 / max(w, 1), 450 / max(h, 1))
        pil = Image.fromarray(vis).resize(
            (max(1, int(w * sc)), max(1, int(h * sc))),
            Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil)
        self.canvas.config(image=tk_img, text="")
        self.canvas.image = tk_img

    def capture_frame(self):
        if not self.is_webcam_running:
            return
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showwarning("Failed", "Try again.")
            return
        self.is_frame_captured = True
        self.current_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.run_yolo(self.current_image)
        self.btn_capture.config(state="disabled")
        self.btn_recapture.config(state="normal")
        self.btn_scan.config(state="normal")
        self.lbl_cam_status.config(text="Captured! READ or RE-CAPTURE")

    def recapture(self):
        self.is_frame_captured = False
        self.best_plate_crop = None
        self.reset_ui()
        self.btn_capture.config(state="normal")
        self.btn_recapture.config(state="disabled")
        self.btn_scan.config(state="disabled")
        self.lbl_cam_status.config(text="Live - CAPTURE when ready")
        if self.is_webcam_running:
            self.update_live()

    def stop_webcam(self):
        self.is_webcam_running = False
        self.is_frame_captured = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def run_yolo(self, arr):
        self._step("2.YOLO")

        bgr_arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        is_dark, _ = LowLightEnhancer.is_dark(bgr_arr)
        if is_dark:
            bgr_arr = LowLightEnhancer.enhance_for_yolo(bgr_arr)
            arr = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2RGB)

        vis = arr.copy()
        res = self.yolo(arr, conf=0.20, verbose=False)
        self.best_plate_crop = None
        best_width = 0

        for r in res:
            for box in r.boxes:
                c = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                hi, wi = arr.shape[:2]

                px_l = int((x2 - x1) * 0.05)
                px_r = int((x2 - x1) * 0.20)
                py = int((y2 - y1) * 0.12)

                x1p = max(0, x1 - px_l)
                y1p = max(0, y1 - py)
                x2p = min(wi, x2 + px_r)
                y2p = min(hi, y2 + py)

                bw = x2p - x1p
                bh = y2p - y1p
                if bh > 0 and (bw / bh) < 2.5:
                    target_w = int(bh * 4.5)
                    expand = (target_w - bw) // 2
                    x1p = max(0, x1p - expand)
                    x2p = min(wi, x2p + expand)

                crop_w = x2p - x1p
                if crop_w > best_width:
                    best_width = crop_w
                    self.best_plate_crop = arr[y1p:y2p, x1p:x2p].copy()

                cv2.rectangle(vis, (x1, y1), (x2, y2), (50, 255, 50), 3)

        if not self.is_camera_mode:
            h, w = vis.shape[:2]
            sc = min(700 / max(w, 1), 450 / max(h, 1))
            pil = Image.fromarray(vis).resize(
                (max(1, int(w * sc)), max(1, int(h * sc))),
                Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil)
            self.canvas.config(image=tk_img, text="")
            self.canvas.image = tk_img

    def trigger_ocr(self):
        if self.is_camera_mode and not self.is_frame_captured:
            messagebox.showinfo("Capture First", "Click CAPTURE first.")
            return
        self.cancel()
        self.root.after(80, self._start)

    def _start(self):
        with self.lock:
            self.cancel_flag.clear()
            self.task_id += 1
            tid = self.task_id
        self.btn_scan.config(state="disabled")
        self.btn_save.config(state="disabled")
        self.lbl_saved.config(text="")
        threading.Thread(target=self.run_pipeline, args=(tid,), daemon=True).start()

    def run_pipeline(self, tid):
        if self.cancelled(tid):
            return
        self._ui(lambda: self.lbl_dbg.config(text=""))

        if self.best_plate_crop is None:
            self._ui(lambda: self.lbl_plate.config(text="No plate detected"))
            self._ui(lambda: self.btn_scan.config(state="normal"))
            return

        try:
            bgr = cv2.cvtColor(self.best_plate_crop, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            if w < 60 or h < 15:
                s = max(60 / max(w, 1), 15 / max(h, 1), 2.0)
                bgr = cv2.resize(bgr, None, fx=s, fy=s,
                                 interpolation=cv2.INTER_CUBIC)

            self._prog("Step 1: Low light check...")
            self._step("1.NIGHT")

            self._prog("Step 3: Perspective correction...")
            self._step("3.PERSP")

            self._prog("Step 4: Multi-preprocessing...")
            self._step("4.PREPROC")

            self._prog("Step 5: PaddleOCR + EasyOCR reading...")
            self._step("5.PADDLE+EASY")

            self._prog("Step 6: Voting...")
            self._step("6.VOTE")

            text, conf, detail = self.pipeline.run(bgr, self.is_camera_mode)

            self._step("7.65-PAIR")
            self._prog("Step 7: Confusion resolver done")

            self._step("8.REGEX")
            self._prog("Step 8: Regex validation done")

            self._step("9.OUTPUT")
            self._prog("Done")

            if text and len(text) >= MIN_CHARS:
                self.result = (text, conf, detail)
                valid, _, fmt = RegexValidator.validate(text)
                src = "Camera" if self.is_camera_mode else "Image"
                self.pending_log = {
                    'predicted': text,
                    'conf': conf,
                    'valid': valid,
                    'detail': detail,
                    'source': src,
                }
                flag = " VALID FORMAT" if valid else " CHECK FORMAT"
                fc = (" [HIGH]" if conf >= 0.70 else
                      " [MED]" if conf >= 0.40 else " [LOW]")
                self._ui(lambda: (
                    self.lbl_plate.config(text=text),
                    self.lbl_conf.config(text=f"Confidence: {conf * 100:.1f}%{fc}"),
                    self.lbl_valid.config(text=flag),
                    self.lbl_detail.config(text=detail[:400] if detail else ""),
                ))
                self._ui(lambda: self.btn_save.config(state="normal"))
                self._ui(lambda: (
                    self.btn_correct.config(state="normal"),
                    self.btn_wrong.config(state="normal"),
                    self.btn_skip.config(state="normal"),
                    self.lbl_log_status.config(
                        text="Mark this result before next image",
                        fg="#FFA500"),
                ))
                print(f"\n  RESULT: '{text}'  {conf:.1%}")
                print(f"  VALID : {valid}")
            else:
                self._ui(lambda: self.lbl_plate.config(text="No read"))
                self.pending_log = {
                    'predicted': '',
                    'conf': 0.0,
                    'valid': False,
                    'detail': detail or '',
                    'source': "Camera" if self.is_camera_mode else "Image",
                }
                self._ui(lambda: (
                    self.btn_skip.config(state="normal"),
                    self.lbl_log_status.config(
                        text="No read — SKIP to log this failure",
                        fg="#FF6B6B"),
                ))

            if not self.cancelled(tid):
                self._ui(lambda: self.lbl_dbg.config(
                    text=detail[:500] if detail else ""))

        except Exception as e:
            import traceback
            traceback.print_exc()
            if not self.cancelled(tid):
                self._prog(f"Error: {e}")
        finally:
            if not self.cancelled(tid):
                self._ui(lambda: self.btn_scan.config(state="normal"))

    def mark_correct(self):
        if not self.pending_log:
            return
        p = self.pending_log
        ok = FailureLogger.save(
            predicted=p['predicted'], correct_text=p['predicted'],
            verdict='CORRECT', conf=p['conf'],
            valid=p['valid'], detail=p['detail'], source=p['source'])
        self.lbl_log_status.config(
            text="Logged as CORRECT" if ok else "Log failed",
            fg="#69F0AE" if ok else "#FF6B6B")
        self.btn_correct.config(state="disabled")
        self.btn_wrong.config(state="disabled")
        self.btn_skip.config(state="disabled")
        self.pending_log = None
        print(f"[LOG] CORRECT: '{p['predicted']}'")

    def mark_wrong(self):
        if not self.pending_log:
            return
        self.btn_correct.config(state="disabled")
        self.btn_wrong.config(state="disabled")
        self.btn_skip.config(state="disabled")
        self.lbl_log_status.config(
            text="Type the correct plate number and click CONFIRM",
            fg="#FF8A65")
        self.correction_var.set(self.pending_log.get('predicted', ''))
        self.correct_frame.pack(fill="x", pady=(2, 4))
        self.entry_correction.focus_set()
        self.entry_correction.select_range(0, 'end')

    def confirm_wrong(self):
        if not self.pending_log:
            return
        correct_text = ''.join(
            c for c in self.correction_var.get().strip().upper() if c.isalnum())
        p = self.pending_log
        ok = FailureLogger.save(
            predicted=p['predicted'], correct_text=correct_text,
            verdict='WRONG', conf=p['conf'],
            valid=p['valid'], detail=p['detail'], source=p['source'])
        self.correct_frame.pack_forget()
        self.lbl_log_status.config(
            text=(f"Logged as WRONG  (correct: {correct_text})"
                  if ok else "Log failed"),
            fg="#FF8A65" if ok else "#FF6B6B")
        self.pending_log = None
        print(f"[LOG] WRONG: predicted='{p['predicted']}' correct='{correct_text}'")

    def mark_skip(self):
        if not self.pending_log:
            return
        p = self.pending_log
        ok = FailureLogger.save(
            predicted=p['predicted'], correct_text='',
            verdict='SKIP', conf=p['conf'],
            valid=p['valid'], detail=p['detail'], source=p['source'])
        self.lbl_log_status.config(
            text="Skipped" if ok else "Log failed",
            fg="#888" if ok else "#FF6B6B")
        self.btn_correct.config(state="disabled")
        self.btn_wrong.config(state="disabled")
        self.btn_skip.config(state="disabled")
        self.correct_frame.pack_forget()
        self.pending_log = None
        print(f"[LOG] SKIP: '{p['predicted']}'")

    def save_result(self):
        text, conf, detail = self.result
        if not text or len(text) < MIN_CHARS:
            messagebox.showwarning("No Result", "Nothing to save.")
            return
        valid, _, fmt = RegexValidator.validate(text)
        src = "Camera" if self.is_camera_mode else "Image"
        row = [
            datetime.now().strftime("%Y-%m-%d"),
            datetime.now().strftime("%H:%M:%S"),
            text, f"{conf * 100:.1f}%",
            "PaddleOCR+EasyOCR+Vote",
            src,
            "YES" if valid else "NO",
            detail[:120] if detail else "",
        ]
        for fn in [EXCEL_FILENAME, "Plate_Logs_backup.csv"]:
            for _ in range(3):
                try:
                    new = not os.path.isfile(fn)
                    with open(fn, 'a', newline='', encoding='utf-8') as f:
                        w = csv.writer(f)
                        if new:
                            w.writerow([
                                "Date", "Time", "Plate",
                                "Confidence", "Engine",
                                "Source", "ValidFormat", "Detail"
                            ])
                        w.writerow(row)
                    msg = f"Saved: {text}"
                    self._ui(lambda m=msg: self.lbl_saved.config(text=m))
                    print(f"[CSV] {msg}")
                    return
                except PermissionError:
                    time.sleep(0.4)
                except Exception as e:
                    print(f"[CSV ERR] {e}")
                    break
        self._ui(lambda: self.lbl_saved.config(text="Save failed"))

    def _prog(self, t):
        self._ui(lambda: self.lbl_prog.config(text=t))

    def _ui(self, fn):
        self.root.after(0, fn)

    def on_closing(self):
        self.cancel()
        self.stop_webcam()
        self.root.destroy()


# ══════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app = CustomANPR(root)
    root.mainloop()