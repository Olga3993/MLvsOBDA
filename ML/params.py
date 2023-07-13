from copy import deepcopy
from itertools import product

comb_fields = [
    'model__gnn_type',
    'features__feat_type',
    'model__hidden_dims',
    'model__add_skip',
    'model__use_input_weighting',
    'model__normalization',
    'features__feat_dim',
    'masking__node_masking',
    'masking__test_ratio',
    'masking__seed',
]

def build_combs_from_config(comb_config):
    vals_lst = []
    for comb_field in comb_fields:
        field1, field2 = comb_field.split('__')
        vals_lst.append(comb_config[field1][field2])

    all_val_combs = product(*vals_lst)
    res = []
    for comb in all_val_combs:
        cur_config = deepcopy(comb_config)
        for i in range(len(comb)):
            field1, field2 = comb_fields[i].split('__')
            cur_config[field1][field2] = comb[i]
        res.append(cur_config)
    return res
