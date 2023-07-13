from .deezer import Deezer
from .polblogs import Polblogs
from .soc_pokec import Soc_pokec
from .les_miserables import LesMiserables


def init_dataset(dataset_name):
    if dataset_name == 'polblogs':
        return Polblogs()
    if dataset_name == 'deezer':
        return Deezer()
    if dataset_name == 'pokec-1':
        return Soc_pokec()
    if dataset_name == 'les_miserables':
        return LesMiserables()