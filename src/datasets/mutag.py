# https://github.com/flyingdoog/PGExplainer/blob/master/MUTAG.ipynb

import yaml
import torch
import numpy as np
import pickle as pkl
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data

import matplotlib.pyplot as plt 
import networkx as nx 
from torch_geometric.utils import to_networkx
from rdkit import Chem 


class Mutag(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root=root)
        # self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Mutagenicity_A.txt', 'Mutagenicity_edge_gt.txt', 'Mutagenicity_edge_labels.txt',
                'Mutagenicity_graph_indicator.txt', 'Mutagenicity_graph_labels.txt', 'Mutagenicity_label_readme.txt',
                'Mutagenicity_node_labels.txt', 'Mutagenicity.pkl']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        raise NotImplementedError

    def process(self):
        with open(self.raw_dir + '/Mutagenicity.pkl', 'rb') as fin:
            _, original_features, original_labels = pkl.load(fin)

        #needs to be synced up to mutag_dual
        with open(self.raw_dir + "/mask_log.txt", "w") as file:

            edge_lists, graph_labels, edge_label_lists, node_type_lists = self.get_graph_data()

            data_list = []
            for i in range(original_labels.shape[0]):
                num_nodes = len(node_type_lists[i])
                edge_index = torch.tensor(edge_lists[i], dtype=torch.long).T

                y = torch.tensor(graph_labels[i]).float().reshape(-1, 1)
                x = torch.tensor(original_features[i][:num_nodes]).float()
                assert original_features[i][num_nodes:].sum() == 0
                edge_label = torch.tensor(edge_label_lists[i]).float()
                if y.item() != 0:
                    edge_label = torch.zeros_like(edge_label).float()

                node_label = torch.zeros(x.shape[0])
                signal_nodes = list(set(edge_index[:, edge_label.bool()].reshape(-1).tolist()))
                if y.item() == 0:
                    node_label[signal_nodes] = 1

                if len(signal_nodes) != 0:
                    node_type = torch.tensor(node_type_lists[i])
                    node_type = set(node_type[signal_nodes].tolist())
                    assert node_type in ({4, 1}, {4, 3}, {4, 1, 3})  # NO or NH

                if y.item() == 0 and len(signal_nodes) == 0:
                    file.write("0\n")
                    continue
                
                file.write("1\n")
                if i<5:
                    # print("\U0001F96D node_label.shape:", node_label.shape)
                    # print("\U0001F96D I:", i)
                    #try:
                    if i < 5:
                        # edge_index = torch.tensor([[0, 1, 0, 2, 1, 3],
                        #        [1, 0, 2, 0, 3, 1]], dtype=torch.long)
                        num_unique = torch.unique(edge_index).numel()
                        # print("unique:", torch.unique(edge_index))
                        # print ("\U0001F96D num_unique", num_unique)
                        edge_att = torch.ones(edge_index.shape[1])
                        # node_label = torch.zeros(4)
                        #node_label = torch.zeros(num_unique)
                        fig, ax = plt.subplots()
                        fig.canvas.manager.set_window_title(f"Primal Graph v2 {i}")
                        # print("\U0001F96D edge_index:", edge_index)
                        # print("\U0001F96D edge_att:", edge_att)
                        # print("\U0001F96D node_label:", node_label)
                        # print("\U0001F96D node_label.size(0)", node_label.size(0))
                        
                        visualize_a_graph(edge_index, edge_att, node_label, 'mutag', ax, coor=None, norm=False, mol_type=None, nodesize=300)
                        
                        plt.show(block=True)
                        plt.close(fig)

                data_list.append(Data(x=x, y=y, edge_index=edge_index, node_label=node_label, edge_label=edge_label, node_type=torch.tensor(node_type_lists[i])))

        print("len_primal_datalist: ", len(data_list))
        input("continue check primal len")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_graph_data(self):
        pri = self.raw_dir + '/Mutagenicity_'

        file_edges = pri + 'A.txt'
        # file_edge_labels = pri + 'edge_labels.txt'
        file_edge_labels = pri + 'edge_gt.txt'
        file_graph_indicator = pri + 'graph_indicator.txt'
        file_graph_labels = pri + 'graph_labels.txt'
        file_node_labels = pri + 'node_labels.txt'

        edges = np.loadtxt(file_edges, delimiter=',').astype(np.int32)
        try:
            edge_labels = np.loadtxt(file_edge_labels, delimiter=',').astype(np.int32)
        except Exception as e:
            print(e)
            print('use edge label 0')
            edge_labels = np.zeros(edges.shape[0]).astype(np.int32)

        graph_indicator = np.loadtxt(file_graph_indicator, delimiter=',').astype(np.int32)
        graph_labels = np.loadtxt(file_graph_labels, delimiter=',').astype(np.int32)

        try:
            node_labels = np.loadtxt(file_node_labels, delimiter=',').astype(np.int32)
        except Exception as e:
            print(e)
            print('use node label 0')
            node_labels = np.zeros(graph_indicator.shape[0]).astype(np.int32)

        graph_id = 1
        starts = [1]
        node2graph = {}
        for i in range(len(graph_indicator)):
            if graph_indicator[i] != graph_id:
                graph_id = graph_indicator[i]
                starts.append(i+1)
            node2graph[i+1] = len(starts)-1
        # print(starts)
        # print(node2graph)
        graphid = 0
        edge_lists = []
        edge_label_lists = []
        edge_list = []
        edge_label_list = []
        for (s, t), l in list(zip(edges, edge_labels)):
            sgid = node2graph[s]
            tgid = node2graph[t]
            if sgid != tgid:
                print('edges connecting different graphs, error here, please check.')
                print(s, t, 'graph id', sgid, tgid)
                exit(1)
            gid = sgid
            if gid != graphid:
                edge_lists.append(edge_list)
                edge_label_lists.append(edge_label_list)
                edge_list = []
                edge_label_list = []
                graphid = gid
            start = starts[gid]
            edge_list.append((s-start, t-start))
            edge_label_list.append(l)

        edge_lists.append(edge_list)
        edge_label_lists.append(edge_label_list)

        # node labels
        node_label_lists = []
        graphid = 0
        node_label_list = []
        for i in range(len(node_labels)):
            nid = i+1
            gid = node2graph[nid]
            # start = starts[gid]
            if gid != graphid:
                node_label_lists.append(node_label_list)
                graphid = gid
                node_label_list = []
            node_label_list.append(node_labels[i])
        node_label_lists.append(node_label_list)

        return edge_lists, graph_labels, edge_label_lists, node_label_lists
    
def visualize_a_graph(edge_index, edge_att, node_label, dataset_name, ax, coor=None, norm=False, mol_type=None, nodesize=300):
    if norm:  # for better visualization
        edge_att = edge_att**10
        edge_att = (edge_att - edge_att.min()) / (edge_att.max() - edge_att.min() + 1e-6)

    if mol_type is None or dataset_name == 'Graph-SST2':
        atom_colors = {0: '#E49D1C', 1: '#FF5357', 2: '#a1c569', 3: '#69c5ba'}
        node_colors = [None for _ in range(node_label.shape[0])]
        for y_idx in range(node_label.shape[0]):
            node_colors[y_idx] = atom_colors[node_label[y_idx].int().tolist()]
    else:
        node_color = ['#29A329', 'lime', '#F0EA00',  'maroon', 'brown', '#E49D1C', '#4970C6', '#FF5357']
        element_idxs = {k: Chem.PeriodicTable.GetAtomicNumber(Chem.GetPeriodicTable(), v) for k, v in mol_type.items()}
        node_colors = [node_color[(v - 1) % len(node_color)] for k, v in element_idxs.items()]

    #assert node_label.size(0) >= edge_index.max().item() + 1, "node_label too short"

    data = Data(edge_index=edge_index, att=edge_att, y=node_label, num_nodes=node_label.size(0)).to('cpu')
    G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])

    # from itertools import combinations

    # def repel_nodes(pos, min_dist=0.05):
    #     for i, j in combinations(pos, 2):
    #         pi, pj = pos[i], pos[j]
    #         dist = np.linalg.norm(pi - pj)
    #         if dist < min_dist:
    #             direction = (pi - pj) / (dist + 1e-6)
    #             shift = 0.5 * (min_dist - dist) * direction
    #             pos[i] += shift
    #             pos[j] -= shift
    #     return pos

    #pos = nx.kamada_kawai_layout(G)
    #pos = repel_nodes(pos, min_dist=0.1)

    # calculate Graph positions
    if coor is None:
        pos = nx.kamada_kawai_layout(G)
        #pos = nx.spring_layout(G, seed=42)
        # pos = repel_nodes(pos, min_dist=0.1)

    else:
        pos = {idx: each.tolist() for idx, each in enumerate(coor)}

    for source, target, data in G.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops=dict(
                arrowstyle="->" if dataset_name == 'Graph-SST2' else '-',
                lw=max(data['att'], 0) * 3,
                alpha=max(data['att'], 0),  # alpha control transparency
                color='black',  # color control color
                shrinkA=np.sqrt(nodesize) / 2.0 + 1,
                shrinkB=np.sqrt(nodesize) / 2.0 + 1,
                connectionstyle='arc3,rad=0.4' if dataset_name == 'Graph-SST2' else 'arc3'
            ))

    if mol_type is not None:
        nx.draw_networkx_labels(G, pos, mol_type, ax=ax)

    if dataset_name != 'Graph-SST2':
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=nodesize, ax=ax)
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', arrows=False, alpha=0.1, ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', arrows=False, alpha=0.1, ax=ax, connectionstyle='arc3,rad=0.4')

    labels = {i: str(i) for i in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=12, ax=ax)

    # Get all x and y positions from the layout
    x_values = [p[0] for p in pos.values()]
    y_values = [p[1] for p in pos.values()]

    # Compute padding and limits
    x_margin = (max(x_values) - min(x_values)) * 0.2 + 0.1
    y_margin = (max(y_values) - min(y_values)) * 0.2 + 0.1

    ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
    ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)

    ax.set_aspect('equal') #this
    ax.axis('off') #this