from DeepPurpose import oneliner
from DeepPurpose.dataset import *
oneliner.repurpose(*load_SARS_CoV2_Protease_3CL(), *load_antiviral_drugs(no_cid = True))
