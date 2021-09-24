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

taxane_df_path = '/content/drive/MyDrive/Colab Notebooks/data/taxane_activity_with_openbabel_smiles.csv'
taxane_df = pd.read_csv(taxane_df_path, error_bad_lines=False, encoding="Latin-1" )
print(len(taxane_df))
taxane_df = taxane_df.sort_values(['IC50 A2780AD nM'], ascending=[False])
taxane_df['Q13509'] ='MREIVHIQAGQCGNQIGAKFWEVISDEHGIDPSGNYVGDSDLQLERISVYYNEASSHKYVPRAILVDLEPGTMDSVRSGAFGHLFRPDNFIFGQSGAGNNWAKGHYTEGAELVDSVLDVVRKECENCDCLQGFQLTHSLGGGTGSGMGTLLISKVREEYPDRIMNTFSVVPSPKVSDTVVEPYNATLSIHQLVENTDETYCIDNEALYDICFRTLKLATPTYGDLNHLVSATMSGVTTSLRFPGQLNADLRKLAVNMVPFPRLHFFMPGFAPLTARGSQQYRALTVPELTQQMFDAKNMMAACDPRHGRYLTVATVFRGRMSMKEVDEQMLAIQSKNSSYFVEWIPNNVKVAVCDIPPRGLKMSSTFIGNSTAIQELFKRISEQFTAMFRRKAFLHWYTGEGMDEMEFTEAESNMNDLVSEYQQYQDATAEEEGEMYEDDEEESEAQGPK'

smiles = list(taxane_df['openbabel_can'])
target_sequence = list(taxane_df['Q13509'])
labels = list(taxane_df['IC50 A2780AD nM'])


model = models.model_pretrained(path_dir = '/content/drive/MyDrive/Colab Notebooks/models/DeepPurpose_IC50_models/NPASS_tranformer_protein_1')
print(model.config)

drug_encoding = 'CNN'
target_encoding = 'Transformer'

X_pred = utils.data_process(smiles, target_sequence, labels, 
                                drug_encoding, target_encoding, 
                                split_method='no_split')


y_pred = model.predict(X_pred)

df = pd.DataFrame(y_pred)
df.to_csv('/content/drive/MyDrive/Colab Notebooks/data/deeppurpose_IC50_taxane_predictions.csv')

print('The predicted score is ' + str(y_pred))
print(len(y_pred))
