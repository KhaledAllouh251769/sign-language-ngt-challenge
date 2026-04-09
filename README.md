# Dutch Sign Language (NGT) Recognition System

Real-time fingerspelling detection for Nederlandse Gebarentaal (Dutch Sign Language) using computer vision and machine learning.

**Breda University of Applied Sciences | ADS&AI Program | Block B/C Challenge**

---

## 👥 Team

- **Khaled Allouh** — Hand Detection
- **Ali Berk** — Data Collection & Classification
- **Abi Parodi** — User Interface

---

## ▶️ How to Open the App

> **Do this every time you want to run the app.**

**Step 1 — Open Anaconda Prompt** (search for it in your Windows start menu)

**Step 2 — Activate the environment:**
```bash
conda activate signlang
```

**Step 3 — Go to the Flask folder:**
```bash
cd C:\path\to\sign-language-ngt-challenge\ngt_flask
```
*(replace `C:\path\to\` with wherever you cloned the repo)*

**Step 4 — Start the app:**
```bash
python app.py
```

**Step 5 — Open Chrome and go to:**
```
http://localhost:5000
```

That's it — the app is now running. To stop it press `Ctrl + C` in the Anaconda Prompt.

> ⚠️ Use **Google Chrome**. Firefox and Edge may have issues with the camera.

---

## 🔧 First Time Setup

Only do this once when setting up the project for the first time.

**1. Clone the repository:**
```bash
git clone https://github.com/KhaledAllouh251769/sign-language-ngt-challenge.git
cd sign-language-ngt-challenge
```

**2. Create the conda environment:**
```bash
conda create -n signlang python=3.11 -y
conda activate signlang
```

**3. Install all dependencies:**
```bash
pip install -r requirements.txt
```

---

## 📖 How to Use the App

### ✍️ Sentence Builder (main feature)
1. Go to the **Sentence** tab
2. **Click a letter** on the keyboard — the camera opens automatically
3. **Sign that letter** in front of your camera
4. Watch the right panel — it shows what letter it detects and how confident it is:
   - 🟢 Green = confident and correct
   - 🟡 Amber = not totally sure
   - 🔴 Red = wrong letter, adjust your hand
5. **Hold the sign** until the purple progress bar fills up → letter gets added
6. Click the next letter and repeat
7. Use the buttons for **Space**, **Delete**, **Clear**, and **Copy**

### 🎯 Practice Mode
1. Go to the **Practice** tab
2. Click **Start Camera**
3. Sign any letter — see what the system detects in real time with a confidence score
4. Click **Stop** when done

### 📚 Learn Mode
1. Go to the **Learn** tab
2. Click any letter to see:
   - Whether it's static (hold position) or dynamic (involves motion)
   - A description of how to sign it

---

## ♿ Accessibility Settings

All settings are in the **left sidebar:**

| Setting | What it does |
|---|---|
| **Color theme** | Switch between Default, High Contrast, Deuteranopia, Protanopia |
| **Font size** | Make all text bigger or smaller (14px–26px) |
| **Mirror mode** | Flip the camera for left-handed users |

---

## 🆕 What's New vs Block B

| Feature | Block B | Block C/D |
|---|---|---|
| Interface | Streamlit | Custom Flask + HTML/CSS/JS |
| Sentence building | ❌ | ✅ Keyboard-guided input |
| Accessibility themes | ❌ | ✅ 4 colour themes |
| Font size control | ❌ | ✅ Adjustable slider |
| Camera flickering | ❌ Frequent | ✅ Fixed — runs in browser |
| Live confidence display | ❌ | ✅ Colour-coded feedback |

---

## 📁 Project Structure

```
sign-language-ngt-challenge/
├── ngt_flask/                    ← New Flask web app
│   ├── app.py                    ← Run this to start the app
│   ├── templates/index.html      ← Frontend (HTML)
│   └── static/
│       ├── css/style.css         ← Styling
│       └── js/app.js             ← Camera + detection logic
├── src/
│   ├── classification.py         ← Letter recognition
│   ├── detection.py              ← Hand detection (MediaPipe)
│   ├── ui.py                     ← Old Streamlit app (still works)
│   ├── data_collection.py        ← Record static letters
│   └── data_collection_dynamic.py← Record dynamic letters
├── data/
│   └── reference/                ← Training data (.json files)
├── requirements.txt
└── README.md
```

---

## 📊 Adding More Training Data

If the system isn't recognising a letter well, record more samples:

**Static letters (A–G, I, K–N, O–S, T, V, W, Y):**
```bash
conda activate signlang
cd sign-language-ngt-challenge
python src/data_collection.py
```

**Dynamic letters (H, J, U, X, Z — involve motion):**
```bash
python src/data_collection_dynamic.py
```

Follow the on-screen instructions. Aim for at least 5 samples per letter per person.

---

## 🔧 Technical Overview

| Component | Technology |
|---|---|
| Hand detection | MediaPipe (Python + JS) |
| Classification | K-Nearest Neighbours (template matching) |
| Backend | Flask (Python) |
| Frontend | HTML / CSS / JavaScript |
| Camera | Browser WebRTC via MediaPipe JS |

**How classification works:**
1. Browser captures webcam frame
2. MediaPipe JS extracts 21 hand landmarks (x, y, z)
3. Landmarks sent to Flask `/classify` endpoint
4. Python normalises landmarks and compares to training data
5. Returns best match + confidence score
6. JS updates the UI instantly

---

## 🐛 Troubleshooting

**"No reference data found" when starting app:**
- Make sure you run `python app.py` from inside the `ngt_flask/` folder
- Check that `data/reference/` exists in the project root and has `.json` files

**Camera not working in browser:**
- Use Chrome — Firefox/Edge may not work
- Allow camera permissions when the browser asks
- Close other apps using the camera (Zoom, Teams, etc.)

**"ModuleNotFoundError":**
- Make sure the conda environment is active: `conda activate signlang`
- Run `pip install -r requirements.txt` again

**Letter not being recognised correctly:**
- Check the confidence colour — red means it's guessing wrong
- Adjust your hand position or lighting
- Record more training data for that letter

---

## 📄 License

Educational project — Breda University of Applied Sciences

## 🙏 Acknowledgments

- Nienke Fluitman — NGT Teacher & Project Sponsor
- Irene van Blerck / Karna Rewatkar — Project Supervisors
- Google MediaPipe Team — Hand tracking technology