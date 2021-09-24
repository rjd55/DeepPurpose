import os
os.chdir('../')
import DeepPurpose.DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *
from DeepPurpose import utils, dataset
import DeepPurpose.DTI as models

from time import time

t1 = time()


#path = utils.download_pretrained_model('/content/drive/MyDrive/Colab Notebooks/models/DeepPurpose_IC50_models/NPASS_tranformer_protein_1')
#net = models.DBTA.model_pretrained(path_dir = path)

path = '/content/drive/MyDrive/Colab Notebooks/models/DeepPurpose_IC50_models/NPASS_tranformer_protein_1'
model = models.DBTA.load_pretrained(path)
model.config
print(model.config)
