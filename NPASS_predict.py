import os
os.chdir('../')
import DeepPurpose.DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *
from DeepPurpose import utils, dataset
import DeepPurpose.DTI as models

from time import time
t1 = time()
import pandas as pd

test_df_path = '/content/drive/MyDrive/Colab Notebooks/data/NPASS_test_for_deeppurpose.csv'
test_df = pd.read_csv(test_df_path, error_bad_lines=False, encoding="Latin-1" )
print(len(test_df))

smiles = list(test_df['SMILES'])
Target_sequence = list(test_df['Target_sequence'])
labels = list(test_df['log_IC50'])


model = models.model_pretrained(path_dir = '/content/drive/MyDrive/Colab Notebooks/models/DeepPurpose_IC50_models/NPASS_tranformer_protein_1')
print(model.config)

drug_encoding = 'CNN'
target_encoding = 'Transformer'

X_pred = utils.data_process(smiles, target_sequence, labels, 
                                drug_encoding, target_encoding, 
                                split_method='no_split')


y_pred = model.predict(X_pred)

df = pd.DataFrame(y_pred)
df.to_csv('deeppurpose_IC50_predictions.csv')
!cp deeppurpose_IC50_predictions.csv "/content/drive/MyDrive/Colab Notebooks/data"

print('The predicted score is ' + str(y_pred))
print(len(y_pred))
