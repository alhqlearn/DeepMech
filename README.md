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

If you want to create a new dataset, prepare the `.txt` files and place them in:  

Required files:  
- `train.txt`  
- `val.txt`  
- `test.txt`  

Each file should contain **atom-mapped elementary steps** in **SMILES format**, one per line.  

---

## 📂 Prepare Your Dataset

1. Create a directory inside `./data`, e.g.  

   ```bash
   mkdir ./data/my_data
   ```

2. Place the following files in `./data/my_data`:  
   - `train.txt`  
   - `val.txt`  
   - `test.txt`  

   Each file should contain **atom-mapped elementary steps (SMILES format)**, one per line.  

---

## ⚙️ Preprocessing

Run preprocessing scripts inside `./preprocessing`:

```bash
# Step 1: Extract information from training data
python preprocessing/Extract_from_train_data.py -d my_data

# Step 2: Preprocess dataset
python preprocessing/Run_preprocessing.py -d my_data
```

---

## 🏋️ Training

Train your model using:

```bash
python scripts/TrainWithAcc.py -d my_data -m my_model
```

---

## ✅ Testing

### (a) Single-step prediction
```bash
python SingleStepPred.py 
```

### (b) Full ID-CRM prediction
1. Train the Reaction Classifier. Load the trained model in the reactivity_classifier_AttentiveFP.py file
2. Run inference:  

```bash
python AttFP_ID_CRM_infer.py 
```

---

## 🔮 Prediction

For predicting complete CRMs, run:

```bash
python xyz.py -d my_data -m my_model
```

---

## 📌 Notes

- Ensure all SMILES in your dataset are **atom-mapped**.  
- The same `-d my_data` and `-m my_model` flags should be consistently used across preprocessing, training, and testing.  
