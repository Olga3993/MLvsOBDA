import os

import networkx as nx
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import sparse

from datasets_ablation.mask import mask_graph


class Dataset:
    def __init__(self, graph_path, labels_path, feature_params, masking_params, sep=','):
        self.dataset_folder = self.init_dataset_folder()
        labels_df = pd.read_csv(labels_path, header=None, sep=sep, names=['id', 'label'])
        G = nx.from_edgelist(pd.read_csv(graph_path, sep=sep).values.tolist())
        
        A_1 = nx.DiGraph(G)
        for e in A_1.edges():
            A_1.add_edge(e[1],e[0])
        
        self.graph = A_1
        self.ids = labels_df['id'].values
        self.labels = labels_df['label'].values
        self.n_classes = len(np.unique(self.labels))
        self.features = self.init_features(len(A_1.nodes()), feature_params) # np.random.uniform(0,1,(len(A_1.nodes()),256))
        self.masking_params = masking_params
        self.feature_params = feature_params

    def init_dataset_folder(self):
        return ''

    def init_features(self, n_nodes, params):
        feat_dim = params['feat_dim']
        if params['feat_type'] == 'random':
            return np.random.uniform(0,1,(n_nodes,feat_dim))
        elif params['feat_type'] == 'dummy':
            return np.eye(n_nodes) # sparse.eye(n_nodes, format='coo', dtype=float) # np.eye(n_nodes)
        elif params['feat_type'] == 'Node2Vec':
            node2vec_path = os.path.join(self.dataset_folder, params['Node2Vec_file'])
            return np.load(node2vec_path)
        elif params['feat_type'] == 'dummy_small':
            res = np.zeros((n_nodes, feat_dim))
            res[:, 0] = range(1, n_nodes + 1)
            return res

    def mask_nodes(self, output_dict, seed):
        node_mask = self.masking_params['node_masking']
        node_train, nodes_to_mask, label_train, label_mask = train_test_split(output_dict['ids'],
                                                          output_dict['labels'],
                                                          stratify=output_dict['labels'],
                                                          test_size=node_mask,
                                                          random_state=seed)
        ids = [i for i in output_dict['ids'] if i not in nodes_to_mask]
        eval_mask = np.zeros(len(output_dict['ids']))
        eval_mask[nodes_to_mask] = True
        return {
            'graph': mask_graph(self.graph, nodes_to_mask),
            'features': self.features[ids], # if self.feature_params['feat_type'] != 'dummy' else coo_matr_by_ids(len(output_dict['ids']), ids),
            'ids': np.array(range(len(ids))),
            'labels': [label for i, label in enumerate(output_dict['labels']) if i not in nodes_to_mask],
            'main_ids': np.array(range(len(ids))),
            'main_labels': [label for i, label in enumerate(output_dict['labels']) if i not in nodes_to_mask],
            'n_classes': self.n_classes,
        }, eval_mask

    def get_data(self, seed):
        full_res = {
            'graph': self.graph,
            'features': self.features,
            'ids': self.ids,
            'labels': self.labels,
            'main_ids':  self.ids,
            'main_labels': self.labels,
            'n_classes': self.n_classes
        }
        masked_res, eval_mask = self.mask_nodes(full_res, seed)
        full_res['eval_mask'] = eval_mask
        return masked_res, full_res


# def coo_matr_by_ids(initial_size, ids):
#     row, col, data = [], [], []
#     for i in range(len(ids)):
#         row.append(i)
#         col.append(ids[i])
#     data = [1 for _ in range(len(ids))]
#     return sparse.coo_matrix((data, (row, col)), shape=(len(ids), initial_size))
