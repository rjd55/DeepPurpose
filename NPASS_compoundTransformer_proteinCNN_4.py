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

print(X_drugs[:5])
print(X_targets[:5])
print(y[:5])

drug_encoding = 'Transformer'
target_encoding = 'CNN'
train, val, test = data_process(X_drugs, X_targets, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.85,0.1,0.05])

config = generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding, 
                         cls_hidden_dims = [1024,1024,512], 
                         train_epoch = 100, 
                         LR = 0.001, 
                         batch_size = 32,
                         cnn_target_filters = [32,64,96],
                         cnn_target_kernels = [4,8,12],
                         result_folder = "/content/drive/MyDrive/Colab Notebooks/DeepPurpose_results/NPASS_compoundTransformer_proteinCNN_4",
                         transformer_n_layer_drug = 10,
                         
                        )


model = models.model_initialize(**config)
t2 = time()
print("cost about " + str(int(t2-t1)) + " seconds")
model.train(train, val, test)

model.save_model('/content/drive/MyDrive/Colab Notebooks/models/NPASS_compoundTransformer_proteinCNN_4')
