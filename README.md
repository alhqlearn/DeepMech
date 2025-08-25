# DeepMech

DeepMech is an interpretable graph-based deep learning framework for predicting complete chemical reaction mechanisms (CRMs).  

---

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/alhqlearn/DeepMech.git
cd DeepMech

# Create and activate virtual environment
conda create -c conda-forge -n deepmech python=3.6 -y
conda activate deepmech

# Install dependencies
conda install pytorch cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge rdkit -y
conda install -c dglteam dgl-cuda11.3 -y
pip install dgllife
```
---


## 📂 Download Dataset and Model  

Download the dataset and model from the following link, and place them in the respective folders:  

- Dataset → `./data/ID_Split42`  
- Model → `./models`  

🔗 [Google Drive Link](https://drive.google.com/drive/folders/1t3pK0Qg9OHp7b5rTq1Db2zrVqq1fQD_1?usp=sharing)  

If you want to create a new dataset, prepare the `.txt` files and place them in e.g., `./data/my_data`

Required files:  
- `train.txt`  
- `val.txt`  
- `test.txt`  

Each file should contain **atom-mapped elementary steps** in **SMILES format**, one per line.  

---

## ⚙️ Preprocessing

Run preprocessing scripts inside `./preprocessing`:

```bash
# Step 1: Extract information from training data
python preprocessing/Extract_from_train_data.py -d ID_Split42

# Step 2: Preprocess dataset
python preprocessing/Run_preprocessing.py -d ID_Split42
```

---

## 🏋️ Training

Train your model using:

```bash
python scripts/TrainWithAcc.py -d ID_Split42 -m Split42
```

---

## ✅ Testing

### (a) Single-step Prediction
Run:
```bash
python SingleStepPred.py
```

### (b) Full ID-CRM Prediction

1. **Train the Reaction Classifier**  
   ```bash
   Reaction_Classifier/Training_Reaction_Classifier.ipynb
   ```

2. **Load the Trained Model**  
   - Update the file `reactivity_classifier_AttentiveFP.py` with the trained model path.  

3. **Run Inference**  
   ```bash
   python AttFP_ID_CRM_infer.py
   ```

---

## 🔮 New Prediction

For predicting complete CRMs, run:

```bash
python NewCRM_Prediction.py
```

---

## 📑 Citation

If you use **DeepMech** in your research, please cite it as:  

```bibtex
@misc{deepmech2025,
  title        = {DeepMech: Interpretable Graph-based Deep Learning for Predicting Complete Chemical Reaction Mechanisms},
  author       = {},
  year         = {2025},
  howpublished = {\url{https://github.com/alhqlearn/DeepMech}},
}
```

---

