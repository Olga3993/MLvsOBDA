from datasets_ablation.deezer import Deezer
from datasets_ablation.polblogs import Polblogs
from datasets_ablation.soc_pokec import Soc_pokec
from .les_miserables import LesMiserables


def init_dataset(dataset_params, feature_params, masking_params):
    if dataset_params['name'] == 'polblogs':
        return Polblogs(feature_params, masking_params)
    if dataset_params['name'] == 'deezer':
        return Deezer(feature_params, masking_params)
    if dataset_params['name'] == 'pokec-1':
        return Soc_pokec(feature_params, masking_params)
    if dataset_params['name'] == 'les_miserables':
        return LesMiserables(feature_params, masking_params)