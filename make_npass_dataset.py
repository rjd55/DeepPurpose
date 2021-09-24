import os
os.chdir('../')

from DeepPurpose import utils, DTI, dataset

path_to_text_file = '/content/drive/MyDrive/Colab Notebooks/data/NPASS_for_DeepPurpose.txt'

X_drugs, X_targets, y = dataset.read_file_training_dataset_drug_target_pairs(path_to_text_file)

print('type X_drugs: ' + str(type(X_drugs)))
print('type X_targets: ' + str(type(X_targets)))
print('type y: ' + str(type(y)))

print('There are ' + str(len(X_drugs)) + ' drug-target pairs.')
print(X_drugs[:5])
print(X_targets[:5])
print(y[:5])
