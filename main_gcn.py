from data_process import split_data, load_disease_data, count_labels
from visualization import visualize
from model_new import MLP,GCN,HGCN,HGCN_pyg
import torch
from geometric_dataset import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from visualization import visualize
import networkx as nx
import numpy as np
import scipy.sparse as sp
from data_process import sparse_mx_to_torch_sparse_tensor,process
from warnings import simplefilter
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=FutureWarning)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())[0]
    # dataset = load_disease_data("disease_nc",'./disease_nc')
    datapath = 'data/Planetoid/Cora'
    # graph = dataset.graph
    feat = dataset.x.numpy()
    label = dataset.y
    train_mask = dataset.train_mask

    test_mask = dataset.test_mask

    edge_index_ori = dataset.edge_index
    edge_index = dataset.edge_index.numpy().T
    # print(edge_index.shape)
    leng = len(label)
    adj = np.zeros((leng, leng))
    # print(adj.shape)
    edge_index = edge_index.tolist()

    # for i, j in edge_index:
    n=0
    for p in edge_index:
        n += 1
        i = p[0]
        j = p[1]
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    adj = sp.csr_matrix(adj)
    adj, feat = process(
        adj, feat, 1, 1)



    ##############disease#####################
    # feat = dataset['features']
    # label = dataset['labels']
    # print('labels:')
    # print(len(label))
    # train_mask = dataset['idx_train']
    # print(len(train_mask))
    # test_mask = dataset['idx_test']
    # print(len(test_mask))
    # edge_index = dataset['edge_index']
    # adj = dataset['adj_train_norm']

    # model = MLP(hidden_channels=16)
    # model = HGCN(1)
    # model = GCN(hidden_channels=86)
    model = HGCN_pyg(c=0.2, hidden_channels=86)
    print(model)
    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    optimizer = torch.optim.Adam(model.parameters()
                                 , lr=0.01
                                 , weight_decay=5e-4
                                 # , weight_decay=0.001
                                 )  # Define optimizer.

    def train():
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(
                      feat
                    # , adj
                    , edge_index = edge_index_ori
                    )  # Perform a single forward pass.
        # print('=================================================================================================')

        #########################################
        ############# hyperbolic gcn#############
        #########################################

        pred = out.argmax(dim=1)

        train_correct = pred[train_mask] == label[train_mask]
        # Derive ratio of correct predictions.
        # train_acc = int(train_correct.sum()) / int(len(train_mask))
        train_acc = int(train_correct.sum()) / int(train_mask.sum())
        # print(f'train acc: {train_acc}')
        loss = criterion(
                         out[train_mask],
                         label[train_mask]
                        )  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss, train_acc

    def test():
        model.eval()
        out = model(feat
                    # , edge_index
                    ,edge_index_ori
                    # ,adj
                    )

        pred = out.argmax(dim=1)  # Use the class with highest probability.

        test_correct = pred[test_mask] == label[test_mask]
        print(test_correct.sum())
        # test_acc = int(test_correct.sum()) / int(test_mask.sum()) # Check against ground-truth labels.

        # test_acc = int(test_correct.sum()) / int(len(idtest).sum())  # Derive ratio of correct predictions.
        # test_acc = int(test_correct.sum()) / int(len(test_mask))
        test_acc = int(test_correct.sum()) / int(test_mask.sum())
        return test_acc


    for epoch in range(1, 201):
        loss, train_acc = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Accuracy: {train_acc:.4f}')

    test_acc = test()
    print(f'Test Accuracy: {test_acc:.4f}')

    model.eval()

    out = model(feat
                , edge_index_ori
                # , adj
                )
    visualize(out, color=label)