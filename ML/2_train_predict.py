import os
import time

import torch
import numpy as np
import dgl


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset_classrom.dataset import Classroom
from models.gnn import GNN
from params import build_combs_from_config
from utils import read_json


#from datasets import Cora, CiteseerM10, Dblp
#from text_transformers import TFIDF, Index, BOW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_sparse_feats(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def get_feats_torch(feats, feat_type):
    # if feat_type == 'dummy':
    #     return get_sparse_feats(feats)
    return torch.FloatTensor(feats)


def get_masks(n,
              main_ids,
              main_labels,
              test_ratio,
              val_ratio,
              seed=1):
    train_mask = np.zeros(n)
    val_mask = np.zeros(n)
    test_mask = np.zeros(n)

    x_dev, x_test, y_dev, y_test = train_test_split(main_ids,
                                                    main_labels,
                                                    stratify=main_labels,
                                                    test_size=test_ratio,
                                                    random_state=seed)

    x_train, x_val, y_train, y_val = train_test_split(x_dev,
                                                      y_dev,
                                                      stratify=y_dev,
                                                      test_size=val_ratio,
                                                      random_state=seed)

    train_mask[x_train] = 1
    val_mask[x_val] = 1
    test_mask[x_test] = 1


    return train_mask, val_mask, test_mask


def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask].detach().cpu().numpy()
        _, predicted = torch.max(logits, dim=1)
        predicted = predicted.detach().cpu().numpy()
        f1 = f1_score(labels, predicted, average='micro')
        return f1


def predict(model, graph, features, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        _, predicted = torch.max(logits, dim=1)
        predicted = predicted.detach().cpu().numpy()
        return predicted


def train_predict_node_clf(dataset,
                    feature_params,
                   test_ratio=0.5,
                   val_ratio=0.1,
                   seed=1,
                   h_dims=(64,),
                   conv_name='GCN',
                   normalization='None',
                   gnn_use_input_weighting=True,
                   use_skip=True,
                   n_epochs=200,
                   lr=1e-2,
                   weight_decay=5e-4,
                   dropout=0.5,
                   verbose=True,
                   cnt=-1,
                   **kwargs):
    dataset.set_feat_params(feature_params)
    masked_data, full_data = dataset.get_data()
    masked_nodes = np.where(full_data['eval_mask'])[0]
    masked_data['features'] = get_feats_torch(masked_data['features'], dataset.feature_params['feat_type']).to(device)
    masked_data['labels'] = torch.LongTensor(masked_data['labels']).to(device)
    # print(masked_data['labels'])
    n_nodes_in_masked = len(masked_data['ids'])
    # print(n, max(data['main_ids']), max(data['main_labels']))
    train_mask, val_mask, test_mask = get_masks(n_nodes_in_masked,
                                                masked_data['main_ids'],
                                                masked_data['main_labels'],
                                                test_ratio=test_ratio,
                                                val_ratio=val_ratio,
                                                seed=seed)
    
    # print( np.sum(train_mask), np.sum(data['main_labels'] *train_mask))
    # print( np.sum(val_mask), np.sum(data['main_labels'] *val_mask))
    # print( np.sum(test_mask), np.sum(data['main_labels'] *test_mask))

    train_mask = torch.BoolTensor(train_mask).to(device)
    val_mask = torch.BoolTensor(val_mask).to(device)
    test_mask = torch.BoolTensor(test_mask).to(device)

    g = dgl.from_networkx(masked_data['graph']).to(device)
    if conv_name in ['GCN', 'GAT']:
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    #
    # degs = g.in_degrees().float()
    # norm = torch.pow(degs, -0.5)
    # norm[torch.isinf(norm)] = 0
    #
    # if cuda:
    #     norm = norm.cuda()
    #
    # g.ndata['norm'] = norm.unsqueeze(1)
    # in_feats = features.shape[1]

    n_classes = 2 # full_data['n_classes']
    h_dims = list(h_dims)
    h_dims.append(n_classes)
    # model = GCN(g,
    #             in_feats=in_feats,
    #             n_hidden=n_hidden,
    #             n_classes=n_classes,
    #             activation=F.relu,
    #             dropout=dropout,
    #             use_embs=use_embs,
    #             pretrained_embs=pretrained_embs,
    #             pad_ix=pad_ix,
    #             n_tokens=n_tokens)
    model = GNN(feat_dim=masked_data['features'].shape[1],
                h_dims=h_dims,
                conv_name=conv_name,
                normalization=normalization,
                dropout=dropout,
                use_input_weighting=gnn_use_input_weighting,
                use_skip=use_skip,
                **kwargs).to(device)

    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, min_lr=1e-10)

    best_f1 = -100
    # initialize graph
    dur = []
    # print("train gcn")
    for epoch in range(n_epochs):

        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        # mask_probs = torch.empty(masked_data['features'].shape).uniform_(0, 1).to(device)

        # mask_features = torch.where(mask_probs > 0.2, masked_data['features'], torch.zeros_like(masked_data['features']))
        logits = model(g, masked_data['features'])
        loss = loss_fcn(logits[train_mask], masked_data['labels'][train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        f1 = evaluate(model, g, masked_data['features'], masked_data['labels'], val_mask)
        scheduler.step(1 - f1)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_model.pt')

        if verbose:
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | F1 {:.4f} | "
                  "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                                f1, n_edges / np.mean(dur) / 1000))

    model.load_state_dict(torch.load('best_model.pt'))
    f1_test = evaluate(model, g, masked_data['features'], masked_data['labels'], test_mask)
    f1_train = evaluate(model, g, masked_data['features'], masked_data['labels'], train_mask)

    g = g.to("cpu")
    masked_data['features'] = masked_data['features'].to("cpu")
    masked_data['labels'] = masked_data['labels'].to("cpu")
    torch.cuda.empty_cache()

    g_full = dgl.from_networkx(full_data['graph']).to(device)
    if conv_name in ['GCN', 'GAT']:
        g_full = dgl.add_self_loop(g_full)
    full_data['eval_mask'] = torch.BoolTensor(full_data['eval_mask']).to(device)
    full_data['features'] = get_feats_torch(full_data['features'], dataset.feature_params['feat_type']).to(device)
    # print(full_data['features'].shape, masked_data['features'].shape)
    # f1_test_full = evaluate(model, g_full, full_data['features'], full_data['labels'], full_data['eval_mask'])
    start_time = time.time()
    predicted = predict(model, g_full, full_data['features'], full_data['eval_mask'])
    end_time = time.time()
    # if verbose:
    # print()
    # print(predicted)
    # print("Train F1 {:.3}".format(f1_train), "Test F1 {:.3}".format(f1_test))
    # print('---------------')
    # model = model.detach().cpu()
    del model

    model_name = f'{conv_name}_skip' if use_skip else conv_name
    save_fname = f'{model_name}'

    predicted_with_mask = np.vstack([predicted, masked_nodes])
    np.save(dataset.dataset_folder / 'preds_with_masks' / f'{save_fname}.npy', predicted_with_mask)


def main():
    dataset_to_feat_dim = {
        'classroom': 64,
        'polblogs': 64,
        'deezer': 128,
        'pokec-1': 128,
    }
    dataset_to_models_params = read_json('config/best_params2.json')['model_params']
    # for dct in params_lst:
    #     print(dct)
    #     print('=========')
    dataset_to_models_params = {'classroom': dataset_to_models_params['classroom']}
    for dataset_name in (dataset_to_models_params):
        print(dataset_name)
        dataset = Classroom(dataset_name)
        os.makedirs(dataset.dataset_folder / 'models', exist_ok=True)
        os.makedirs(dataset.dataset_folder / 'preds_with_masks', exist_ok=True)
        for model_params in dataset_to_models_params[dataset_name]:
            print(model_params['gnn_type'])
            feature_params = {
                "feat_type": model_params['features'],
                "feat_dim": dataset_to_feat_dim[dataset_name],
                "Node2Vec_file": model_params['Node2Vec_file'] if 'Node2Vec_file' in model_params else '',
            }
            train_predict_node_clf(dataset,
                                   feature_params=feature_params,
                                   val_ratio=0.2,
                                   seed=42,
                                   h_dims=model_params['hidden_dims'],
                                   conv_name=model_params['gnn_type'],
                                   normalization=model_params['normalization'],
                                   gnn_use_input_weighting=model_params['use_input_weighting'],
                                   use_skip=model_params['add_skip'],
                                   n_epochs=30,  # 120
                                   lr=1e-3,
                                   weight_decay=5e-4,
                                   dropout=0.5,
                                   verbose=False)
            torch.cuda.empty_cache()
        # break


if __name__ == '__main__':
    import torch

    torch.manual_seed(456)
    main()
