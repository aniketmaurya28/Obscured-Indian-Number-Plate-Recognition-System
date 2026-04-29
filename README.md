# Detection and Recognition of Obscured Vehicle License Plates using Image Processing and Machine Learning
- A practical Automatic Number Plate Recognition (ANPR) system designed for Indian vehicle plates under difficult real-world conditions such as mud, rain, motion blur, glare, low light, perspective skew, non-standard fonts, and partial occlusion.
​

### Overview
- This project was developed as a final-year major project in the Department of Computer Engineering, Indala College of Engineering, Kalyan, under the University of Mumbai for the academic year 2025–26.
- The system focuses on improving number plate recognition reliability in challenging Indian traffic scenarios where conventional ANPR systems often fail. It combines image processing, machine learning, OCR, and rule-based post-processing to produce accurate and explainable outputs suitable for intelligent transportation use cases such as tolling, parking enforcement, surveillance, and public safety.
​

## Problem Statement
Traditional ANPR systems perform well in controlled environments, but their performance drops significantly in real traffic conditions because of:
- Motion blur from moving vehicles.
- Perspective distortion from angled camera views.
​- Low light and glare.
​- Dirt, mud, and rain covering number plates.
​- Non-standard Indian fonts, spacing, and layouts.
​- Frequent confusion between similar characters such as 0/O, 8/B, 5/S, and M/N.
​- This project addresses these issues with a hybrid ANPR pipeline optimized for robustness and CPU-efficient deployment.
​

## Objectives
- Detect Indian vehicle number plates reliably from images and live camera feeds.
- Improve recognition accuracy for obscured or degraded plates.
​- Reduce empty or incorrect outputs using multi-fallback preprocessing and multi-engine recognition.
- Correct systematic OCR mistakes using Indian plate format rules and a confusion resolver.
​- Provide a practical GUI-based application with result logging for real-world usage.
​

# Key Features
- YOLO-based number plate detection with padded cropping.
- Deskewing and geometric correction using minimum-area rectangle estimation.
- Multi-fallback preprocessing with CLAHE, denoising, sharpening, and adaptive binarization.
- Hybrid recognition using EasyOCR and a custom 36-class character CNN.
- Weighted voting fusion across recognition outputs.
​- 65-pair context-aware confusion resolver for ambiguous character correction.
​- Indian state-code and format validation.
​- Tkinter GUI for image upload and live camera input.
​- CSV logging for auditability and traceability.
​

### System Architecture
- The system follows an end-to-end modular pipeline.
- Image or camera input.
- YOLOv8-based plate detection.
- Perspective and geometric correction.
- Multi-preprocessing pipeline.
​- Dual recognition using OCR and CNN.
- Weighted voting and output fusion.
​- 65-pair confusion resolution.
​- Regex and Indian format validation.
​- Final output display and CSV logging.
​

## Tech Stack: 

Area	Tools / Methods
Programming	Python 
​
Detection	YOLO / YOLOv8 
​
OCR	EasyOCR, PaddleOCR (mentioned in methodology and evaluation flow) 
​
Deep Learning	Custom 36-class CNN 
​
Image Processing	OpenCV, CLAHE, thresholding, morphology, deskewing 
​
Interface	Tkinter GUI 
​
Logging	CSV-based audit logging 
​
## Methodology
### 1. Plate Detection
A YOLO-based detector identifies the vehicle number plate from an image or camera frame. The highest-confidence bounding box is selected and padded to avoid cutting off characters near the edges.
​

### 2. Preprocessing
The cropped plate is enhanced using several preprocessing strategies to make text easier to recognize:

Contrast enhancement with CLAHE.
​

Denoising and sharpening.
​

Deskewing for tilted plates.
​

Otsu, adaptive Gaussian, and adaptive mean thresholding.
​

Morphological cleanup and optional restoration filters for rain or mud artifacts.
​

### 3. Character Recognition
The project uses a hybrid recognition pipeline:

EasyOCR as a segmentation-free baseline for full-plate reading.
​

Character-level CNN for segmented character classification into 36 classes (A-Z, 0-9).
​

Multi-fallback recognition to retry difficult crops using alternate preprocessing variants.
​

### 4. Voting Fusion
Outputs from multiple engines are aligned by dominant string length and merged using confidence-weighted voting to improve final recognition reliability.
​

### 5. Confusion Resolution
A deterministic post-processing module handles 65 common ambiguous character pairs using:

Indian plate format matching.
​

State-code validation such as MH, DL, and others.
​

Local character context.
​

CNN confidence-gap analysis.
​

Run-isolation repair for inconsistent characters.
​

### Dataset
The dataset was prepared to reflect real Indian road conditions and includes:
- Images of private and commercial vehicles.
​- Samples captured using phone cameras and laptop webcams.
​- Plates with mud, dirt, rain streaks, glare, blur, and low-light effects.
​- Variations in font, spacing, angle, and plate style.
​- YOLO-style annotations with train/validation/test splits of 80/10/10.
​

### Results
According to the analysis, the proposed system achieved strong performance compared with standalone recognition baselines under difficult conditions.
​- Full combined system plate accuracy: 0.84.
​- EasyOCR-only plate accuracy: 0.78.
​- CNN-only plate accuracy: 0.74.
​- Failure rate of the combined system: approximately 1%.
​

# Condition-wise examples reported in the document include:
- Clean daylight: combined accuracy reached 94%.
​- Skewed views: combined accuracy reached 88%.
​- Motion blur: combined accuracy reached 78%.
​- Mud/dirt: combined accuracy reached 72%.
​- Rain/low contrast: combined accuracy reached 73%.
​- Night/glare: combined accuracy reached 63%.
​

## Contributions
This project contributes:

- A hybrid multi-engine ANPR recognition pipeline.
​- A multi-fallback preprocessing and recognition design to reduce empty outputs.
- A 65-pair context-aware confusion resolver specialized for Indian plates.
​- A practical GUI-based deployment workflow with CSV audit logs.
​- A CPU-efficient solution intended for real-world use on modest hardware.
​

## Use Cases
- Automated toll collection.
​- Parking management systems.
- Traffic surveillance and enforcement.
​- Vehicle access control.
​- Public safety and smart city transportation systems.
​

## Future Scope
- The report suggests several directions for improvement:
- Video-based temporal fusion across multiple frames.
- Stronger sequence-recognition models such as CRNN-based approaches.
​- Support for additional regional languages and scripts.
- Better performance under extreme weather or visibility loss.
​- More automated format and character-size detection.
- Image restoration for severely faded or occluded plates.
​

## Publication
The project report states that the work was published in IJIRSET, Volume 15, Special Issue 1, January 2026, and is also associated with an IJEDR 2026 paper version in the document's publication section.
​

## Team
Sandesh Lande 
​
Adarsh Mane 
​
Aniket Maurya 
​
Jayesh Wadekar 
​
Guide: Asst. Prof. Gauri A. Bhosale 
​

### License
Add a license that matches how you want others to use the project. For student and portfolio projects, the MIT License is often the simplest option.

### Notes for Future Developers
- Keep OCR outputs and resolver rules modular so they can be improved independently.
- Preserve CSV logs because they help with debugging and auditability.
- Maintain separate evaluation for plate accuracy, character accuracy, edit distance, and failure rate.
​- Document dataset sources and annotations clearly before retraining the models.
- If deploying on low-resource devices, prioritize lightweight models and preprocessing efficiency.
