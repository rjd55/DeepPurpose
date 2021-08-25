import os
os.chdir('../')

from DeepPurpose import CompoundPred as models
from DeepPurpose.utils import *
from tdc import BenchmarkGroup
group = BenchmarkGroup(name = 'ADMET_Group', path = 'data/')

import warnings
warnings.filterwarnings("ignore")

## 0.1.2 new supported models: 
## DGL_GCN, DGL_NeuralFP, DGL_GIN_AttrMasking, DGL_GIN_ContextPred, DGL_AttentiveFP  
drug_encoding = 'DGL_GCN'
    
benchmark = group.get('Caco2_Wang')

train, valid = group.get_train_valid_split(benchmark = benchmark['name'], split_type = 'default', seed = 1)

train = data_process(X_drug = train.Drug.values, y = train.Y.values, 
                drug_encoding = drug_encoding,
                split_method='no_split')

val = data_process(X_drug = valid.Drug.values, y = valid.Y.values, 
                drug_encoding = drug_encoding,
                split_method='no_split')

test = data_process(X_drug = benchmark['test'].Drug.values, y = benchmark['test'].Y.values, 
                drug_encoding = drug_encoding,
                split_method='no_split')

config = generate_config(drug_encoding = drug_encoding, 
                         cls_hidden_dims = [512], 
                         train_epoch = 10, 
                         LR = 0.001, 
                         batch_size = 128,
                        )

model = models.model_initialize(**config)
model.train(train, val, test, verbose = True)
y_pred = model.predict(test)
