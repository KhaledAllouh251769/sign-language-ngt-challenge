# Dutch Sign Language Recognition

## Setup
```bash
conda create -n signlang python=3.11 -y
conda activate signlang
pip install -r requirements.txt
```

## Usage
```bash
python src/detection.py
python src/data_collection.py
streamlit run src/ui.py
```
