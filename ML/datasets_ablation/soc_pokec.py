from dirs import DATA_PATH
from .dataset import Dataset
from pathlib import Path

CURRENT_DIR = Path(__file__).parent.absolute()


class Soc_pokec(Dataset):
    def __init__(self,
                 feature_params,
                 masking_params,
                 graph_path=Path(DATA_PATH / 'pokec-1/edges.csv'),
                 labels_path=Path(DATA_PATH / 'pokec-1/labels.csv')):
        super(Soc_pokec, self).__init__(graph_path, labels_path, feature_params, masking_params)

    def init_dataset_folder(self):
        return str(DATA_PATH / 'pokec-1')
