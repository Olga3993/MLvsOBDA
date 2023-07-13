from dirs import DATA_PATH
from .dataset import Dataset
from pathlib import Path


class Polblogs(Dataset):
    def __init__(self,
                 graph_path=Path(DATA_PATH / 'polblogs/edges.csv'),
                 labels_path=Path(DATA_PATH / 'polblogs/labels.csv')):
        super(Polblogs, self).__init__(graph_path, labels_path, 'polblogs')

    def init_dataset_folder(self):
        return str(DATA_PATH / 'polblogs')
