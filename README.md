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
pip install rdkit-pypi
conda install pytorch cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge rdkit -y #not required
conda install -c dglteam dgl-cuda11.3 -y
pip install dgllife

# You can also set up the environment using the environment.yml file by running:
`conda env create --name deepmech -f environment.yml`
```

---


## 📂 Download Dataset and Model  

Download the dataset and model from the following zenodo link, and place them in the respective folders:  

- Dataset → `./data/ID_Split42`  
- Model → `./models`  

🔗 https://doi.org/10.5281/zenodo.20305780

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

To predict complete CRMs, specify the input SMILES in the `NewCRM_Prediction.py` script and execute

```bash
python NewCRM_Prediction.py
```

---

## 📑 Citation

If you use **DeepMech** in your research, please cite it as:  

```bibtex
@article{10.1039/d6sc02809h,
    author = {Das, Manajit and Hoque, Ajnabiul and Baranwal, Mayank and Sunoj, Raghavan B.},
    title = {DeepMech: a machine learning framework for chemical reaction mechanism prediction},
    journal = {Chemical Science},
    year = {2026},
    month = {06},
    abstract = { The ability to predict the complete, step-by-step mechanism of chemical reactions from first principles remains a grand challenge in science. The importance of chemical reaction mechanisms (CRMs) pervades almost all domains such as prebiotic chemistry, drug discovery, materials science and so on. Uncovering CRMs remains a complex task, traditionally reliant on expert-driven experiments or expensive quantum chemical computations. While deep learning (DL) studies have shown promise in predicting reaction outcomes, they have largely ignored important intermediates and mechanistic steps en route to the product of immediate interest. Since sequence-to-sequence models that generate products character by character are prone to hallucination, we consider it important to develop DL models that prioritize reactivity over learning the syntax and semantics of sequences. The progress in reaction mechanism predictions is further limited by the lack of large-scale, mass-balanced datasets with mechanistic annotations. Motivated by these key lacunae, we introduce DeepMech, an interpretable graph-based DL framework that employs attention mechanisms at both atom and bond levels. Instead of end-to-end learning from reactants to products, our model incorporates a template of mechanistic operations (TMOp) for the generation of intermediates in elementary mechanistic steps. It leverages TMOps, to predict step-by-step mechanisms toward realizing full CRMs for a multitude of reaction classes of high contemporary significance. To train our DeepMech model, first we construct ReactMech, a meticulously curated dataset of about 30 K full reaction mechanisms, each comprising several atom-mapped elementary steps (totaling 100 K). DeepMech achieves a state-of-the-art accuracy of 98.98 ± 0.12\% in predicting elementary steps and 95.94 ± 0.21\% in complete CRM tasks. The model maintains high fidelity even in out-of-distribution scenarios involving unseen catalysts, ligands, and/or mechanistic classes. In so far as the generalizability goes, DeepMech effectively reconstructs multistep CRMs relevant to prebiotic chemistry, beginning from trivial primordial substrates such as nitrogen, ammonia, methane, water, and hydrogen cyanide, to complex biomolecules like serine and aldopentose. Further, attention-based interpretability analysis reveals that DeepMech correctly identifies reactive atoms and bonds, in line with chemical intuition. Collectively, DeepMech offers a promising step toward data-driven prediction of CRMs, with potential to expedite mechanistic understanding and reaction design across domains. },
    issn = {2041-6520},
    doi = {10.1039/d6sc02809h},
    url = {https://doi.org/10.1039/d6sc02809h},
    eprint = {https://pubs.rsc.org/sc/article-pdf/doi/10.1039/d6sc02809h/13418534/d6sc02809h.pdf},
}
```

---

