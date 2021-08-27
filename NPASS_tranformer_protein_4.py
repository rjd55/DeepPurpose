import os
os.chdir('../')
import DeepPurpose.DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *
from DeepPurpose import utils, dataset
import DeepPurpose.DTI as models

from time import time

t1 = time()

path_to_text_file = '/content/drive/MyDrive/Colab Notebooks/data/NPASS_for_DeepPurpose.txt'

X_drugs, X_targets, y = dataset.read_file_training_dataset_drug_target_pairs(path_to_text_file)


drug_encoding = 'CNN'
target_encoding = 'Transformer'
train, val, test = data_process(X_drugs, X_targets, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2], random_seed = 5)

# use the parameters setting provided in the paper: https://arxiv.org/abs/1801.10193
config = generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding, 
                         cls_hidden_dims = [1024,1024,512], 
                         train_epoch = 100, 
                         test_every_X_epoch = 100, 
                         LR = 0.001, 
                         batch_size = 128,
                         hidden_dim_drug = 128,
                         cnn_drug_filters = [32,64,96],
                         cnn_drug_kernels = [4,8,12],
                         transformer_n_layer_target = 2
                        )
model = models.model_initialize(**config)
t2 = time()
print("cost about " + str(int(t2-t1)) + " seconds")
model.train(train, val, test)

model.save_model('./model_CNN_Transformer_NPASS_4')
