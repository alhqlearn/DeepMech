# DeepMech


How to train with your own data?
1. Make a directory inside ‘./data’ e.g. ‘./data/my_data’
2. Keep three .txt files in the folder ‘./data/my_data’ i.e. train.txt, val.txt and test.txt. Each file should contain atom-mapped elementary steps (SMILES format) line by line
3. Preprocessing
3a. Run the file ‘Extract_from_train_data.py’ in the ‘./preprocessing’ folder
Example command: python Extract_from_train_data.py –d my_data
3b. Run the file Run_preprocessing.py in the ‘./preprocessing’ folder
Example command: python Run_preprocessing.py
4. Train
Run the TrainWithAcc.py file in the ‘./scripts’ folder
Example command: python TrainWithAcc.py –d my_data –m my_model
5. Test
5a. Single step prediction
Run the SingleStepPred.py file in the ‘./’ directory 
5b. Full ID CRM prediction
Step1:  Train Reaction Classifier
Step2: Run the AttFP_ID_CRM_infer.py file in the ‘./’ directory 
6. Predict 
Run xyz.py file to get predicted CRM





Create a virtual environment
git clone ***
cd DeepMech
conda create -c conda-forge -n rdenv  python=3.6 -y
conda activate rdenv
conda install pytorch cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge rdkit -y
conda install -c dglteam dgl-cuda11.3
pip install dgllife

