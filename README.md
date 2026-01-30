# Dutch Sign Language (NGT) Recognition System

Real-time fingerspelling detection tool for Nederlandse Gebarentaal (Dutch Sign Language) using computer vision and machine learning.

## 🎯 Project Overview

This system recognizes Dutch Sign Language fingerspelling in real-time using:
- **MediaPipe** for hand landmark detection
- **Template matching** for letter classification
- **Streamlit** for user-friendly web interface

Recognizes all 26 letters (A-Z) with 70-80% accuracy, including both static and dynamic letters.

**Developed in 2 weeks as part of the ADS&AI Block B Challenge.**

---

## 👥 Team

- **Khaled Allouh** - Hand Detection Module
- **Ali Berk** - Data Collection & Classification  
- **Abi Parodi** - User Interface

Breda University of Applied Sciences | ADS&AI Program | Block B Challenge | 2-Week Project

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Webcam/camera
- Windows, Mac, or Linux

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

**Main Application (Web Interface):**
```bash
conda activate signlang
streamlit run src/ui.py
```

Your browser will automatically open to `http://localhost:8501`

**Alternative: Test Individual Modules:**
```bash
# Test hand detection
python src/detection.py

# Test classification
python src/classification.py

# Test live recognition
python src/test_classification.py
```

---

## 📖 How to Use

### Practice Mode (Main Feature)

1. Open the application: `streamlit run src/ui.py`
2. Select **"Practice"** from the sidebar
3. Check the **"Start Camera"** box
4. Make signs from the NGT alphabet
5. See real-time detection results with confidence scores
6. Optional: Enable **"Mirror Mode"** in sidebar if you're left-handed

### Learn Mode

1. Select **"Learn"** from the sidebar
2. Choose a letter from the dropdown (A-Z)
3. Watch reference videos showing how to sign that letter
4. See whether the letter is static or dynamic

### Record Mode

1. Select **"Record"** from the sidebar
2. Enter your name
3. Select the letter you want to record
4. Click **"Record New Sample"**
5. Follow the countdown and make the sign clearly
6. Your sample is automatically saved to improve the system

---

## 📁 Project Structure

```
sign-language-ngt-challenge/
├── src/
│   ├── detection.py              # Hand landmark detection (MediaPipe)
│   ├── classification.py         # Letter recognition algorithm
│   ├── ui.py                     # Streamlit web interface
│   ├── data_collection.py        # Record static letters
│   ├── data_collection_dynamic.py # Record dynamic letters (H,J,U,X,Z)
│   └── test_classification.py    # Testing script
├── data/
│   ├── reference/                # Training data (JSON files)
│   └── videos/                   # Tutorial videos (.mov files)
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

## 🔧 Technical Approach

### 1. Hand Detection
- **Technology:** Google MediaPipe Hand Landmarker
- **Output:** 21 (x, y, z) landmark points per hand
- **Performance:** 15-20 FPS on standard laptop CPU

### 2. Data Collection
- **Method:** Team-recorded custom dataset
- **Size:** 390+ samples (26 letters × 15 samples average)
- **Format:** JSON files containing landmark coordinates
- **Contributors:** 3 team members, multiple recording sessions

### 3. Classification Algorithm
- **Method:** Template matching (K-Nearest Neighbors approach)
- **Process:**
  1. Normalize landmarks (remove position/scale variations)
  2. Calculate Euclidean distance to all reference samples
  3. Return closest match with confidence score
  4. Apply smoothing buffer (5-frame average)
- **Accuracy:** 70-80% across all letters

### 4. Dynamic Letter Handling
- **Letters:** H, J, U, X, Z (involve movement)
- **Method:** Frame averaging approach
- **Process:** Capture 60 frames during motion, average into single representative position

### 5. Mirror Mode
- **Purpose:** Support left-handed users
- **Method:** Flip x-coordinates (new_x = 1.0 - old_x)
- **Result:** Training data from right hands works for left hands

---

## 📊 Features

✅ Real-time letter recognition (15-20 FPS)
✅ All 26 NGT letters (A-Z)
✅ Static and dynamic letter support
✅ Left-hand and right-hand support (mirror mode)
✅ Confidence scoring
✅ Web-based interface (no installation for end users)
✅ Three modes: Learn, Practice, Record
✅ User can add training data through UI

---

## 🛠️ Dependencies

**Core Libraries:**
- `mediapipe` - Hand landmark detection
- `opencv-python` - Camera and video processing
- `numpy` - Mathematical operations
- `streamlit` - Web interface framework

**Standard Library:**
- `json` - Data storage
- `os` - File operations
- `time` - Timing operations
- `collections` - Data structures

See `requirements.txt` for exact versions.

---

## 🎓 Approaches Explained

### Why Template Matching Instead of Deep Learning?

**Template matching (K-NN)** was chosen for several reasons:
1. **Time constraint:** One-week project timeline
2. **Data size:** 390 samples is sufficient for template matching but insufficient for neural networks (which typically need 1000+ samples)
3. **Simplicity:** Easier to implement, debug, and understand
4. **Interpretability:** Clear why a prediction was made (closest match)
5. **No training time:** Instant setup, no GPU required

**Trade-off:** Lower accuracy (70-80%) vs deep learning (85-95%), but acceptable for an MVP.

### Finalized Approach Details

**Hand Detection Pipeline:**
```
Camera Frame → MediaPipe Hand Detection → 21 Landmarks (x,y,z) → Normalization
```

**Classification Pipeline:**
```
Normalized Landmarks → Distance Calculation (all 390 samples) → Closest Match → Smoothing → Result + Confidence
```

**Normalization:**
- Center around wrist: `centered = landmarks - wrist_position`
- Scale by hand size: `normalized = centered / hand_size`
- Result: Position and scale invariant features

**Distance Metric:**
- Euclidean distance: `distance = sqrt(sum((current - sample)²))`
- Average of top-3 closest samples per letter for stability
- Confidence: `confidence = 1 / (1 + distance * 0.5)`

**Smoothing:**
- Buffer size: 5 predictions
- Method: Mode (most common prediction in buffer)
- Threshold: Minimum 3 predictions before returning result

---

## 🔮 Future Considerations

### Short-Term Improvements
- Increase accuracy to 85-90% with more training data
- Optimize dynamic letter detection with proper motion tracking
- Implement neural network classifier (CNN or LSTM)
- Add data augmentation techniques
- Improve UI/UX design and animations

### Long-Term Vision
- Expand to full NGT vocabulary (not just fingerspelling)
- Add sentence building and word recognition
- Develop mobile app (iOS/Android)
- Real-time sentence translation
- Multi-user training data collection
- Support for other sign languages (ASL, BSL, etc.)
- Collaboration with deaf community for validation
- Deploy as public web service
- Integration with educational platforms

### Technical Enhancements
- Implement convolutional neural network for higher accuracy
- Use transfer learning from larger sign language datasets
- Add temporal models (LSTM/GRU) for better dynamic letter handling
- Implement attention mechanisms for feature weighting
- Add data augmentation (rotation, scaling, noise)
- Create confidence calibration for better uncertainty estimates

---

## 📝 Usage Notes

### For Best Results:
- Use good lighting (natural light or room light)
- Position hand clearly in camera view
- Hold static letters steady for 1-2 seconds
- Perform dynamic letters (H, J, U, X, Z) smoothly
- Use mirror mode if you're left-handed

### Known Limitations:
- Similar letters (M/N, R/U) may be confused
- Requires clear hand visibility (no partial occlusion)
- Dynamic letters have lower accuracy than static
- Camera must be functional and accessible
- Works best with neutral background

---

## 🐛 Troubleshooting

### "Cannot open camera"
- Check if camera is connected and functional
- Close other applications using the camera (Zoom, Teams, etc.)
- Try different camera index: `cv2.VideoCapture(1)` instead of `(0)`
- On Mac: Grant camera permissions in System Preferences

### "ModuleNotFoundError"
- Ensure conda environment is activated: `conda activate signlang`
- Reinstall requirements: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.11+)

### "Streamlit command not found"
- Activate environment first: `conda activate signlang`
- Reinstall Streamlit: `pip install streamlit`
- Use full path: `python -m streamlit run src/ui.py`

### Low accuracy / not recognizing letters
- Ensure you've recorded training data (check `data/reference/` folder)
- Try adjusting hand position/distance
- Use better lighting
- Enable mirror mode if using left hand
- Record more training samples for problematic letters

---

## 📄 License

Educational project - Breda University of Applied Sciences

---

## 🙏 Acknowledgments

- Nienke Fluitman - NGT Teacher & Project Sponsor
- Irene van Blerck / Karna Rewatkar - Project Supervisors
- Google MediaPipe Team - Hand tracking technology
- Dutch Deaf Community - Inspiration and purpose

---

## 📧 Contact

For questions or feedback about this project:
- GitHub: [KhaledAllouh251769](https://github.com/KhaledAllouh251769)
- Repository: [sign-language-ngt-challenge](https://github.com/KhaledAllouh251769/sign-language-ngt-challenge)

---

**Built with ❤️ in two weeks for the ADS&AI Sign Language Challenge**
