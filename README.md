# Dutch Sign Language (NGT) Recognition System

Real-time fingerspelling detection tool for Nederlandse Gebarentaal (Dutch Sign Language) using computer vision and machine learning.

## 🎯 Project Overview

This system recognizes Dutch Sign Language fingerspelling in real-time using:
- **MediaPipe** for hand landmark detection
- **Template matching** for letter classification
- **Flask + HTML/CSS/JS** for a custom web interface (upgraded from Streamlit)

Recognizes all 26 letters (A-Z) including both static and dynamic letters.

**Originally developed in 2 weeks as part of the ADS&AI Block B Challenge. Extended in Block C/D with new features.**

---

## 👥 Team

- **Khaled Allouh** - Hand Detection Module
- **Ali Berk** - Data Collection & Classification
- **Abi Parodi** - User Interface

Breda University of Applied Sciences | ADS&AI Program | Block B/C Challenge

---

## 🆕 What's New (Block C/D Extension)

### ✍️ Sentence Builder
- Select a letter on the on-screen keyboard, then sign it with your hand
- Live confidence score shown in real time (green = confident, amber = unsure, red = wrong)
- Progress bar fills as you hold the correct sign — letter is added automatically once full
- Space, Delete, Clear, and Copy buttons for full sentence control

### ♿ Accessibility Features
- **4 colour themes:** Default (dark purple), High Contrast (black/yellow), Deuteranopia (blue palette), Protanopia (amber palette)
- **Adjustable font size:** Slider from 14px to 26px, resizes text across the entire app
- **Mirror mode:** Toggle for left-handed users

### 🌐 New Interface — Flask + HTML/CSS/JS
- Replaced Streamlit with a custom Flask web app for full UI control
- Camera runs directly in the browser via MediaPipe JS — no more page reloading or flickering
- Dark themed, custom fonts, smooth animations
- Single page app — no refreshes between actions

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Webcam/camera
- Windows, Mac, or Linux
- Google Chrome (recommended for camera/MediaPipe support)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/KhaledAllouh251769/sign-language-ngt-challenge.git
cd sign-language-ngt-challenge
```

2. **Create conda environment:**
```bash
conda create -n signlang python=3.11 -y
conda activate signlang
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Application

**New Flask App (recommended):**
```bash
conda activate signlang
cd ngt_flask
python app.py
```
Then open **http://localhost:5000** in Chrome.

**Legacy Streamlit App (still works):**
```bash
conda activate signlang
streamlit run src/ui.py
```

**Test individual modules:**
```bash
python src/detection.py          # test hand detection
python src/classification.py     # test classifier
python src/test_classification.py # live recognition test
```

**Collect training data:**
```bash
python src/data_collection.py          # static letters
python src/data_collection_dynamic.py  # dynamic letters (H,J,U,X,Z)
```

---

## 📖 How to Use

### Sentence Builder (New)

1. Open the app at `http://localhost:5000`
2. Go to **Sentence** mode
3. Click a letter on the keyboard
4. Sign that letter in front of your camera — hold until the progress bar fills
5. The letter is added to your sentence automatically
6. Use Space / Delete / Clear / Copy buttons as needed

### Practice Mode

1. Go to **Practice** mode
2. Click **Start Camera**
3. Sign any letter — see the detected letter and confidence in real time

### Learn Mode

1. Go to **Learn** mode
2. Click any letter to see how to sign it and whether it's static or dynamic

### Record Mode (Streamlit only)

1. Run `streamlit run src/ui.py`
2. Go to **Record** mode
3. Enter your name, select a letter, click Record

---

## 📁 Project Structure

```
sign-language-ngt-challenge/
├── ngt_flask/                        # New Flask web app
│   ├── app.py                        # Flask backend + classification API
│   ├── requirements.txt              # Flask-specific deps
│   ├── templates/
│   │   └── index.html                # Single page frontend
│   └── static/
│       ├── css/style.css             # Full custom stylesheet
│       └── js/app.js                 # MediaPipe + camera + UI logic
├── src/
│   ├── detection.py                  # Hand landmark detection (MediaPipe)
│   ├── classification.py             # Letter recognition algorithm
│   ├── ui.py                         # Legacy Streamlit interface
│   ├── data_collection.py            # Record static letters
│   ├── data_collection_dynamic.py    # Record dynamic letters
│   └── test_classification.py        # Live testing script
├── data/
│   ├── reference/                    # Training data (JSON files)
│   └── videos/                       # Tutorial videos (.mov files)
├── requirements.txt                  # Python dependencies
├── .gitignore
└── README.md
```

---

## 🔧 Technical Approach

### 1. Hand Detection
- **Technology:** Google MediaPipe Hand Landmarker (Python + JS)
- **Output:** 21 (x, y, z) landmark points per hand
- **Performance:** 15-20 FPS on standard laptop CPU

### 2. Data Collection
- **Method:** Team-recorded custom dataset
- **Size:** 390+ samples (26 letters × 15 samples average)
- **Format:** JSON files containing landmark coordinates

### 3. Classification Algorithm
- **Method:** Template matching (K-Nearest Neighbors approach)
- **Process:**
  1. Normalize landmarks (remove position/scale variations)
  2. Calculate Euclidean distance to all reference samples
  3. Return closest match with confidence score
  4. Apply smoothing buffer (5-frame average)

### 4. Dynamic Letter Handling
- **Letters:** H, J, U, X, Z (involve movement)
- **Method:** Frame averaging — capture frames during motion, average into single representative position

### 5. Sentence Builder Architecture
- Browser captures webcam frames via MediaPipe JS
- Landmarks sent to Flask `/classify` endpoint via fetch API
- Flask runs the Python classifier and returns letter + confidence
- JS updates UI in real time — no page reloads

### 6. Accessibility Implementation
- CSS custom properties (`--accent`, `--bg`, `--text`, etc.) power all themes
- Theme switching updates `data-theme` attribute on `<html>` — instant, no reload
- Font size slider updates `--font-size` CSS variable globally

---

## 📊 Features

✅ Real-time letter recognition (15-20 FPS)
✅ All 26 NGT letters (A-Z)
✅ Static and dynamic letter support
✅ Left-hand and right-hand support (mirror mode)
✅ Confidence scoring with colour feedback
✅ **Sentence builder with guided keyboard input** *(new)*
✅ **4 accessibility colour themes** *(new)*
✅ **Adjustable font size** *(new)*
✅ **Custom Flask frontend — no Streamlit glitching** *(new)*
✅ Three modes: Learn, Practice, Sentence
✅ User can add training data via data collection scripts

---

## 🛠️ Dependencies

**Backend:**
- `flask` - Web server
- `mediapipe` - Hand landmark detection
- `opencv-python` - Camera and video processing
- `numpy` - Mathematical operations

**Frontend (loaded via CDN — no install needed):**
- `@mediapipe/hands` - Browser-side hand detection
- `@mediapipe/camera_utils` - Browser camera access
- `Google Fonts (Syne, DM Mono)` - Typography

See `requirements.txt` for exact versions.

---

## 🎓 Design Decisions

### Why Keyboard-Guided Input for Sentence Builder?

Free-form recognition (sign any letter, system guesses) was implemented and tested but proved unreliable for a demo due to the limited training dataset (390 samples). A guided approach — where the user selects the target letter first — was chosen because:

1. **Reliability:** The system only needs to confirm one specific letter, not guess from 26
2. **UX:** Real assistive technology (AAC devices) uses guided input for the same reason
3. **Feedback:** Live confidence score tells the user exactly how well they're signing
4. **Accuracy:** Effectively 100% when the user holds the correct sign confidently

### Why Flask Instead of Streamlit?

Streamlit rerenders the entire page on every state change, causing the camera to restart and flicker every time a letter is added. Flask with a custom JS frontend keeps the camera running continuously in the browser, making the sentence builder smooth and usable.

---

## 🔮 Future Improvements

- Increase training data for better free-form recognition accuracy
- CNN/LSTM classifier for higher accuracy
- Full NGT vocabulary beyond fingerspelling
- Mobile app version
- Deploy as public web service

---

## 📝 Usage Notes

### For Best Results:
- Use good lighting
- Position hand clearly in camera view
- Hold static letters steady until the bar fills
- Perform dynamic letters (H, J, U, X, Z) smoothly during the recording window
- Use mirror mode if you're left-handed
- Use Chrome for the Flask app

### Known Limitations:
- Similar letters (M/N, E/S) may be confused with limited training data
- Dynamic letters have lower accuracy than static
- Best results with neutral background

---

## 🐛 Troubleshooting

### Flask app — "No reference data found"
- Make sure you run `python app.py` from inside the `ngt_flask/` folder
- Check that `data/reference/` exists in the project root with `.json` files

### "ModuleNotFoundError"
- Activate environment: `conda activate signlang`
- Reinstall: `pip install -r requirements.txt`

### Camera not working in browser
- Use Chrome (not Firefox or Edge)
- Allow camera permissions when prompted
- Close other apps using the camera (Zoom, Teams, etc.)

### Low accuracy
- Record more training samples via `data_collection.py`
- Ensure good lighting and clear hand position
- Enable mirror mode if using left hand

---

## 📄 License

Educational project — Breda University of Applied Sciences

---

## 🙏 Acknowledgments

- Nienke Fluitman — NGT Teacher & Project Sponsor
- Irene van Blerck / Karna Rewatkar — Project Supervisors
- Google MediaPipe Team — Hand tracking technology
- Dutch Deaf Community — Inspiration and purpose

---

**Built with ❤️ for the ADS&AI Sign Language Challenge — Breda University of Applied Sciences**