from dirs import DATA_PATH
from .dataset import Dataset
from pathlib import Path


class LesMiserables(Dataset):
    def __init__(self,
                 graph_path=Path(DATA_PATH / 'les_miserables/edges.csv'),
                 labels_path=Path(DATA_PATH / 'les_miserables/labels.csv')):
        super(Polblogs, self).__init__(graph_path, labels_path, 'les_miserables')

    def init_dataset_folder(self):
        return str(DATA_PATH / 'les_miserables')
