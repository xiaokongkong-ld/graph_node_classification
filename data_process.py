import os
import pickle as pkl
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from visualization import visualize

# ############### FEATURES PROCESSING ####################################

def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features

def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg

def load_disease_data(dataset, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    edges_tensor = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
        edges_tensor.append([i,j])
    edges_tensor = np.array(edges_tensor)
    edges_tensor = torch.from_numpy(edges_tensor.T)
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.

    features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset)))
    # features = torch.tensor(features)

    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset)))

    val_prop, test_prop = 0.10, 0.60
    idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=1234)

    labels = torch.LongTensor(labels)
    data = {'adj_train': sp.csr_matrix(adj), 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val,
            'idx_test': idx_test, 'edge_index': edges_tensor, 'adj':adj}
    data['adj_train_norm'], data['features'] = process(
        data['adj_train'], data['features'], 1, 1)
    return data

def count_labels(label_list):

    num_label = []
    for labels in label_list:
        if labels >= len(num_label):
            for i in range(len(num_label), labels + 1):
                num_label.append(0)
            num_label[labels] = 1
        else:
            num_label[labels] += 1
    label_length = len(label_list)
    label_number = len(num_label)
    print('.............................................')
    print('..............label information..............')
    print(f'There are {label_length} labels')
    print(f'There are {label_number} label types')
    print('Each label number: ')
    for i in range(len(num_label)):
        print(f'label {i}: {num_label[i]}')
    print('.............................................')
    print('.............................................')

if __name__ == '__main__':
    DATA = "disease_nc"
    dataset = load_disease_data(DATA, './disease_nc')
    adj = dataset['adj_train']
    feat = dataset['features']
    label = dataset['labels']
    idtrain = dataset['idx_train']
    idval = dataset['idx_val']
    idtest = dataset['idx_test']
    edge_index = dataset['edge_index']
    print(f'edge index: {edge_index}')
    print(adj.shape)
    print(f'feature dimension: {feat.shape}')
    print(f'IDs to train dimension: {len(idtrain)}')
    print(idtrain)
    print(f'IDs to test: {len(idtest)}')
    print(idtest)
    print(f'IDs to validate: {len(idval)}')
    print(idval)
    print('All labels')
    # o = 0
    # for j in label:
    #     o += 1
    #     print(f'num: {o}')
    #     print(j)

    count_labels(label)
    visualize(feat, label)