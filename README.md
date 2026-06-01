# 🚗 Number Plate Detection System

A real-time vehicle number plate detection and owner lookup system built using YOLOv8 and Tesseract OCR, with a Streamlit web interface.

---

## 1. Project Overview

This system:
1. Detects the number plate from a car image using a fine-tuned YOLOv8s model
2. Extracts the plate text using Tesseract OCR
3. Looks up the owner details from a CSV database
4. Displays results in a clean Streamlit web app

---

## 2. Project Structure

```
number-plate-detection/
├── app.py                          # Streamlit web application
├── detect.py                       # Detection + OCR + lookup logic
├── finetuning_numberplatedetect.py # Model training script (Google Colab)
├── best.pt                         # Fine-tuned YOLOv8s model weights
├── data.csv                        # Owner database (plate, name, phone)
└── README.md
```

---

## 3. Dataset & Training

- **Model:** YOLOv8s fine-tuned on License Plate dataset
- **Epochs:** 10
- **Batch Size:** 8
- **Platform:** Google Colab
- **Task:** Object Detection (license plate localization)

---

## 4. How It Works

```
Car Image
    ↓
YOLOv8s → Detects plate location → Crops plate
    ↓
Tesseract OCR → Reads plate text
    ↓
CSV Lookup → Finds owner name + phone
    ↓
Streamlit UI → Displays results
```

---

## 5. Tech Stack

- Python
- YOLOv8s (Ultralytics)
- Tesseract OCR
- OpenCV
- Streamlit
- Pandas

---

## 6. How to Run

**Step 1 — Install dependencies:**
```bash
pip install ultralytics opencv-python pytesseract streamlit pandas
```

**Step 2 — Install Tesseract OCR:**

Download from 👉 https://github.com/UB-Mannheim/tesseract/wiki

**Step 3 — Run the app:**
```bash
streamlit run app.py
```

**Step 4 — Open browser:**

---

## 7. Database Format

The `data.csv` file should have these columns:

```csv
plate,name,phone
KL01AB1234,John Doe,9876543210
KL02CD5678,Jane Smith,9123456789
```

---

## 8. Features

- ✅ Upload any car image (JPG, PNG, JPEG)
- ✅ Auto detects and crops number plate
- ✅ Reads plate text using OCR
- ✅ Looks up owner from database
- ✅ Clean two-column layout

---

## 9. Demo Video
👉 [Watch Demo Video](https://drive.google.com/file/d/1vKTBSquQ0CpKTomtNYWEev850A2hwavL/view?usp=drive_link)

---
