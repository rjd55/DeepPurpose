import os
os.chdir('../')
import DeepPurpose.DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *
from DeepPurpose import utils, dataset
import DeepPurpose.DTI as models
import random
from time import time

t1 = time()

path_to_text_file = '/content/drive/MyDrive/Colab Notebooks/data/NPASS_for_DeepPurpose_IC50.txt'

X_drugs, X_targets, y = dataset.read_file_training_dataset_drug_target_pairs(path_to_text_file)

print(X_drugs[:5])
print(X_targets[:5])
print(y[:5])

drug_encoding = 'CNN'
target_encoding = 'Transformer'

x = random.randint(1, 4294967295)

train, val, test = data_process(X_drugs, X_targets, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.85,0.1,0.05], random_seed = x)




# use the parameters setting provided in the paper: https://arxiv.org/abs/1801.10193
config = generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding, 
                         cls_hidden_dims = [1024,1024,512], 
                         train_epoch = 100, 
                     #    test_every_X_epoch = 100, 
                         LR = 0.001, 
                         batch_size = 32,
                         hidden_dim_drug = 128,
                         cnn_drug_filters = [32,64,96],
                         cnn_drug_kernels = [4,6,8],
                      #   transformer_n_layer_target = 2,
                         result_folder = "/content/drive/MyDrive/Colab Notebooks/DeepPurpose_results/NPASS_tranformer_protein_1"
                        )
model = models.model_initialize(**config)
t2 = time()
print("cost about " + str(int(t2-t1)) + " seconds")
model.train(train, val, test)
model.save_model('/content/drive/MyDrive/Colab Notebooks/models/NPASS_tranformer_protein_1')
