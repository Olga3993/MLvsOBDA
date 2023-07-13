from dirs import DATA_PATH
from .dataset import Dataset
import pathlib
from pathlib import Path

# CURRENT_DIR = pathlib.Path(__file__).parent.absolute()


class Deezer(Dataset):
    def __init__(self,
                 feature_params,
                 masking_params,
                 graph_path=Path(DATA_PATH / 'deezer/edges.txt'),
                 labels_path=Path(DATA_PATH / 'deezer/labels.txt')):
        super(Deezer, self).__init__(graph_path, labels_path, feature_params, masking_params)

    def init_dataset_folder(self):
        return str(DATA_PATH / 'deezer')