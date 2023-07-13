import time

import torch
import numpy as np
import dgl


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

from datasets_ablation.init_dataset import init_dataset
from models.gnn import GNN
from params import build_combs_from_config
from utils import read_json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_sparse_feats(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def get_feats_torch(feats, feat_type):
    if feat_type == 'dummy':
        return get_sparse_feats(feats)
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


def train_node_clf(dataset,
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
    masked_data, full_data = dataset.get_data(seed)
    masked_data['features'] = get_feats_torch(masked_data['features'], dataset.feature_params['feat_type']).to(device)
    masked_data['labels'] = torch.LongTensor(masked_data['labels']).to(device)
    n_nodes_in_masked = len(masked_data['ids'])

    train_mask, val_mask, test_mask = get_masks(n_nodes_in_masked,
                                                masked_data['main_ids'],
                                                masked_data['main_labels'],
                                                test_ratio=test_ratio,
                                                val_ratio=val_ratio,
                                                seed=seed)

    train_mask = torch.BoolTensor(train_mask).to(device)
    val_mask = torch.BoolTensor(val_mask).to(device)
    test_mask = torch.BoolTensor(test_mask).to(device)

    g = dgl.from_networkx(masked_data['graph']).to(device)
    if conv_name in ['GCN', 'GAT']:
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()


    n_classes = full_data['n_classes']
    h_dims = list(h_dims)
    h_dims.append(n_classes)

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
    full_data['labels'] = torch.LongTensor(full_data['labels']).to(device)
    # print(full_data['features'].shape, masked_data['features'].shape)
    f1_test_full = evaluate(model, g_full, full_data['features'], full_data['labels'], full_data['eval_mask'])

    # if verbose:
    # print()
    print(cnt, "Train F1 {:.3}".format(f1_train), "Test F1 {:.3}".format(f1_test), "Test F1 Full {:.3}".format(f1_test_full))
    print('---------------')
    # model = model.detach().cpu()
    del model


def main():
    comb_params = read_json('config/combination_params.json')
    params_lst = build_combs_from_config(comb_params)
    print(len(params_lst))

    for i, params in (enumerate(params_lst)):
        if params['model']['gnn_type'] in ['GAT', 'GAT-sep'] and not params['model']['use_input_weighting']:
            print('======', i+1, '======')
            continue
        print(params)
        dataset = init_dataset(params['dataset'], params['features'], params['masking'])

        train_node_clf(dataset,
                       test_ratio=params['masking']['test_ratio'],
                       val_ratio=0.1,
                       seed=params['masking']['seed'],
                       h_dims=params['model']['hidden_dims'],
                       conv_name=params['model']['gnn_type'],
                       normalization=params['model']['normalization'],
                       gnn_use_input_weighting=params['model']['use_input_weighting'],
                       use_skip=params['model']['add_skip'],
                       n_epochs=80, # 120
                       lr=1e-3,
                       weight_decay=5e-4,
                       dropout=0.5,
                       verbose=False,
                       cnt=i+1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
