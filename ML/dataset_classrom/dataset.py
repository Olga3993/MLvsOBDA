import os
from pathlib import Path

import networkx as nx
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import sparse

from datasets_ablation.mask import mask_graph
from dirs import DATA_PATH


class Classroom:
    def __init__(self, sep=','):
        graph_path = Path(DATA_PATH / 'classroom/edges.txt')
        labels_path = Path(DATA_PATH / 'classroom/labels.txt')
        self.dataset_folder = self.init_dataset_folder()
        labels_df = pd.read_csv(labels_path, header=None, names=['id', 'label'])
        print(labels_df)
        print(pd.read_csv(graph_path, names=['v1', 'v2']))
        G = nx.from_edgelist(pd.read_csv(graph_path, names=['v1', 'v2']).values.tolist())
        
        A_1 = nx.DiGraph(G)
        for e in A_1.edges():
            A_1.add_edge(e[1],e[0])
        
        self.graph = A_1
        self.ids = labels_df['id'].values
        self.labels = labels_df['label'].values
        self.n_classes = len(np.unique(self.labels))
        self.id_to_label = dict(zip(self.ids, self.labels))
        self.n_nodes = len(A_1.nodes())
        self.feature_params = None

    def set_feat_params(self, feature_params):
        self.feature_params = feature_params
        self.features = self.init_features(self.n_nodes, feature_params)

    def init_dataset_folder(self):
        return Path(DATA_PATH / 'classroom')

    def init_features(self, n_nodes, params):
        feat_dim = params['feat_dim']
        if params['feat_type'] == 'random':
            return np.random.uniform(0,1,(n_nodes,feat_dim))
        elif params['feat_type'] == 'dummy':
            return np.eye(n_nodes) # np.eye(n_nodes)
        elif params['feat_type'] == 'Node2Vec':
            node2vec_path = os.path.join(self.dataset_folder, params['Node2Vec_file'])
            return np.load(node2vec_path)
        elif params['feat_type'] == 'dummy_small':
            res = np.zeros((n_nodes, feat_dim))
            res[:, 0] = range(1, n_nodes + 1)
            return res

    def mask_nodes(self, output_dict):
        # node_train = [x for x in self.id_to_label if self.id_to_label[x] != 'A']
        nodes_to_mask = [x for x in self.id_to_label if self.id_to_label[x] == 2]
        # label_train = [self.id_to_label[x] for x in self.id_to_label if self.id_to_label[x] != 'A']
        # label_mask = [self.id_to_label[x] for x in self.id_to_label if self.id_to_label[x] == 'A']
        ids = [i for i in output_dict['ids'] if i not in nodes_to_mask]
        eval_mask = np.zeros(len(output_dict['ids']))
        eval_mask[nodes_to_mask] = True
        return {
            'graph': mask_graph(self.graph, nodes_to_mask),
            'features': self.features[ids] ,#if self.feature_params['feat_type'] != 'dummy' else coo_matr_by_ids(len(output_dict['ids']), ids),
            'ids': np.array(range(len(ids))),
            'labels': [label for i, label in enumerate(output_dict['labels']) if i not in nodes_to_mask],
            'main_ids': np.array(range(len(ids))),
            'main_labels': [label for i, label in enumerate(output_dict['labels']) if i not in nodes_to_mask],
            'n_classes': self.n_classes,
        }, eval_mask

    def get_data(self):
        full_res = {
            'graph': self.graph,
            'features': self.features,
            'ids': self.ids,
            'labels': self.labels,
            'main_ids':  self.ids,
            'main_labels': self.labels,
            'n_classes': self.n_classes
        }
        masked_res, eval_mask = self.mask_nodes(full_res)
        full_res['eval_mask'] = eval_mask
        return masked_res, full_res
