import yaml
import scipy
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_sparse import transpose
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph, is_undirected
from ogb.graphproppred import Evaluator
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from torch.nn import functional as F
from typing import Tuple, Iterator
from scipy.stats import truncnorm
from torch_scatter import scatter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from pretrain_clf import train_clf_one_seed
from utils import Writer, Criterion, MLP, visualize_a_graph, save_checkpoint, load_checkpoint, get_preds, get_lr, set_seed, process_data
from utils import get_local_config_name, get_model, get_data_loaders, write_stat_from_metric_dicts, reorder_like, init_metric_dict


class GSAT(nn.Module):

    def __init__(self, clf, extractor, optimizer, scheduler, writer, device, model_dir, dataset_name, num_class, multi_label, random_state,
                 method_config, shared_config, c1, c2, c3, c4, c5):
        super().__init__()
        self.clf = clf
        self.extractor = extractor
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.writer = writer
        self.device = device
        self.model_dir = model_dir
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.method_name = method_config['method_name']

        self.learn_edge_att = shared_config['learn_edge_att']
        self.k = shared_config['precision_k']
        self.num_viz_samples = shared_config['num_viz_samples']
        self.viz_interval = shared_config['viz_interval']
        self.viz_norm_att = shared_config['viz_norm_att']

        self.epochs = method_config['epochs']
        self.pred_loss_coef = c1
        self.info_loss_coef = c2
        
        self.inv_loss_coeff = c3
        self.sparse_loss_coeff = c4
        self.sym_loss_coeff = c5
        #self.entropy_loss_coeff = method_config['entropy_loss_coeff']
        #self.neigh_loss_coeff = method_config['neigh_loss_coeff']

        self.fix_r = method_config.get('fix_r', None)
        self.decay_interval = method_config.get('decay_interval', None)
        self.decay_r = method_config.get('decay_r', None)
        self.final_r = method_config.get('final_r', 0.1)
        self.init_r = method_config.get('init_r', 0.9)

        self.multi_label = multi_label
        self.criterion = Criterion(num_class, multi_label)
        
        #self.rad = torch.nn.Parameter()
        
    def sim(self, a, b):
        return F.cosine_similarity(a.T, b.T)

    def f1_sparsity_loss(self, p_uv, y_uv, eps=1e-6):
        # print(f"p_uv.shape: {p_uv.shape}, y_uv.shape: {y_uv.shape}")
        # print("p_uv: ", p_uv)
        # print("y_uv: ", y_uv)
        TP = (p_uv.view(-1) * y_uv.view(-1)).sum()


        # print(f"p_uv.shape: {p_uv.shape}, y_uv.shape: {y_uv.shape}")
        # print("p_uv: ", p_uv)
        # print("y_uv: ", y_uv)
        # print("p_uv * y_uv: ", (p_uv.view(-1) * y_uv.view(-1)))
        P = p_uv.sum()

        G = y_uv.sum()

        precision = TP / (P + eps)
        recall = TP / (G + eps)

        assert (p_uv >= 0).all() and (p_uv <= 1).all()
        assert (y_uv >= 0).all() and (y_uv <= 1).all()

        f1 = 2 * precision * recall / (precision + recall + eps)

        l1_loss = p_uv.abs().mean()

        total_loss = (1 - f1) + l1_loss

        return total_loss
    
    def __loss__(self, att, clf_logits, clf_labels, epoch, old_emb, emb, edge_index, batch):   
        old_preds = self.clf.get_pred_from_emb(old_emb, batch)
        new_preds = self.clf.get_pred_from_emb(emb, batch)
        # print('assdfads', att.shape)
        inv_loss = 1 - self.sim(old_preds, new_preds)[0]
        #print(inv_loss)
        #print("number of nodes: ", old_emb.shape(0))
        #print("pred from old_emb: ", self.clf.get_pred_from_emb(old_emb, batch))
        param_dict = dict(self.clf.named_parameters())
        eps_matrix = param_dict['convs.0.eps'].detach().cpu()
        sparse_loss = 6 - torch.mean(torch.abs(eps_matrix))
        print("att before node_att: ", att.shape)
        #print("edge_index: ", edge_index, edge_index.shape)
        
        node_att = scatter(att, edge_index[0], dim=0, dim_size=emb.shape[0], reduce='mean').squeeze() # convert ba2motifs edge att to node att
        # num_nodes = torch.max(edge_index)+1
        # np_edges = np.concatenate((edge_index[0].detach().cpu().numpy(), edge_index[1].detach().cpu().numpy()), axis=0)
        # np_attens = np.concatenate((att.detach().numpy(), att.detach().numpy()), axis=0).flatten()
        # sums = np.bincount(np_edges, weights=np_attens, minlength=num_nodes)
        # counts = np.bincount(np_edges, minlength=num_nodes)
        # # node_att = sums/counts
        # node_att = torch.from_numpy(sums/counts)
        # print("att after node_att: ", node_att.shape)
        s = 1 / (node_att + 1) # sigma
        # print("s shape: ", s.shape, self.extractor.radius[0:s.shape[0]].shape)
        # print("s: ", s.shape)
        #print("s: ", torch.isnan(s).any())
        k = 1 / (s * np.sqrt(torch.pi/2) + self.extractor.radius[0:s.shape[0]].squeeze()) # scale factor
        # print("k shape: ", k.shape)
        f_area = 1 - k*s*np.sqrt(torch.pi/2)
        
        perts = torch.empty_like(s).float()
        # print("perts: ", perts.shape)
        # print("radius: ", self.extractor.radius.shape)
        
        choice = torch.rand(self.extractor.radius.shape[0], 1, device=self.extractor.radius.device)[0:s.shape[0]].squeeze() < f_area
        # print("choice: ", choice.shape)
        perts[choice] = torch.rand(choice.sum(), device=self.extractor.radius.device) * self.extractor.radius[0:s.shape[0]][choice].squeeze()
        choice = ~choice
        g_samples = s.detach().numpy() * np.sqrt(2*torch.pi) * k.detach().numpy() * truncnorm.rvs(a=0, b=np.inf, loc=self.extractor.radius[0:s.shape[0]].detach().numpy().squeeze(), scale=s.detach().numpy().squeeze())
        # print("g_samples: ", g_samples.shape)
        perts[choice] = torch.from_numpy(g_samples[choice].squeeze()).float().squeeze()
        # print("perts: ", perts.shape)
        # print("emb shape: ", emb.shape)
        perts2 = torch.randn(emb.shape[0], 64, device=perts.device)
        perts2 = perts2 / perts2.norm(dim=1, keepdim=True)
        perts2 = perts2 * perts.unsqueeze(1)
        
        perturbed_preds = self.clf.get_pred_from_emb(emb + perts2, batch)
        print("pred values: ", torch.sum(perturbed_preds), torch.sum(new_preds))
        # print("perturbed preds: ", perturbed_preds)
        # print("new preds: ", new_preds)
        sym_loss = self.sim(perturbed_preds, new_preds) / (emb.shape[0])
        # sym_loss = (F.kl_div(torch.log(perturbed_preds / torch.sum(perturbed_preds)) , new_preds / torch.sum(new_preds))) / (emb.shape[0])
        # entropy_loss = 
        # neigh_loss = a
        
        inv_loss = inv_loss * self.inv_loss_coeff
        print('inv_loss: ', inv_loss)
        sparse_loss = sparse_loss * self.sparse_loss_coeff
        print('sparse_loss: ', sparse_loss)
        sym_loss = sym_loss * self.sym_loss_coeff
        print('sym_loss: ', sym_loss)
        #entropy_loss = entropy_loss * self.entropy_loss_coeff
        #neigh_loss = neigh_loss * self.neigh_loss_coeff
        

        pred_loss = self.criterion(clf_logits, clf_labels)

        r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
        info_loss = (att * torch.log(att/r + 1e-6) + (1-att) * torch.log((1-att)/(1-r+1e-6) + 1e-6)).mean()

        pred_loss = pred_loss * self.pred_loss_coef
        info_loss = info_loss * self.info_loss_coef
        loss = pred_loss + info_loss + sparse_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss.item()}
        #print(loss)
        return loss, loss_dict

    def forward_pass(self, data, epoch, training):
        print('data: ', data)
        #print('data.x: ', data.x)
        old_emb, emb = self.clf.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr, action=True)
        print("old_emb: ", old_emb.shape)
        print("emb: ", emb.shape)
        #print("batch: ", data.batch.shape)
        #print(data.edge_index.shape)
        #print("aaaaaa", data.x.shape)
        att_log_logits = self.extractor(emb, data.edge_index, data.batch, mess=None)
        print("att_log_logits", torch.isnan(att_log_logits).any())
        att = self.sampling(att_log_logits, epoch, training)
        print("att: ", torch.isnan(att).any())
        # print("att: ", att.shape)

        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)

        clf_logits = self.clf(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
        loss, loss_dict = self.__loss__(att, clf_logits, data.y, epoch, old_emb=old_emb, emb=emb, edge_index=data.edge_index, batch=data.batch)
        return edge_att, loss, loss_dict, clf_logits

    @torch.no_grad()
    def eval_one_batch(self, data, epoch):
        self.extractor.eval()
        self.clf.eval()

        att, loss, loss_dict, clf_logits = self.forward_pass(data, epoch, training=False)
        return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()

    def train_one_batch(self, data, epoch):
        self.extractor.train()
        self.clf.train()

        att, loss, loss_dict, clf_logits = self.forward_pass(data, epoch, training=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()
    
    def visualize_edge_perturbations(self, epoch, emb_2d, edge_index, a, A_phi_2d, topk=50):
        """
        emb_2d: [num_nodes, 2]  - 2D node embeddings (after PCA)
        edge_index: [2, num_edges] - graph edge list
        a: [num_edges, 1] - edge importance
        A_phi_2d: [2, 2] - 2D antisymmetric matrix to perturb edges
        """
        emb = emb_2d.cpu().numpy()
        print("visualizing emb_2d: ", emb.shape)
        edge_index = edge_index.cpu().numpy()
        a = a.squeeze().cpu().numpy()  # shape [num_edges]
        A_phi = A_phi_2d.cpu().numpy()  # shape [2, 2]    # Get top and bottom k edges by importance
        topk_idx = np.argsort(a)[-topk:]
        print("topk: ", topk_idx.shape)
        botk_idx = np.argsort(a)[:topk:] 
        print("bottomk: ", botk_idx.shape)
        def plot_edges(idx_list, color, alpha, label):
            # print(idx_list)
            for idx in idx_list:
                i, j = edge_index[:, idx]
                #print(i, j)
                xi, xj = emb[i], emb[j]
                #print("xixj: ", xi, xj)
                delta = A_phi @ (xj - xi)
                # print("delta: ", delta)
                x_new = xi + delta            # Draw arrow from xi to xi + delta
                plt.arrow(xi[0], xi[1], delta[0], delta[1], color=color, alpha=alpha, head_width=0.02, length_includes_head=True)    
        plt.figure(figsize=(10, 8))
        plt.scatter(emb[:, 0], emb[:, 1], s=10, color='gray', alpha=0.6, label='Node embeddings')    
        plot_edges(topk_idx, color='red', alpha=0.8, label='Most important edges')
        plot_edges(botk_idx, color='blue', alpha=0.5, label='Least important edges')    
        plt.title(f"Perturbation Effects via $A_\\phi$ on Most/Least Important Edges")
        plt.axis('off')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figs/epoch_{epoch}.png')
    
    def run_one_epoch(self, data_loader, epoch, phase, use_edge_attr):
        loader_len = len(data_loader)
        run_one_batch = self.train_one_batch if phase == 'train' else self.eval_one_batch
        phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

        all_loss_dict = {}
        all_exp_labels, all_att, all_clf_labels, all_clf_logits, all_precision_at_k = ([] for i in range(5))
        pbar = tqdm(data_loader)
        # print("pbar: ")
        # for data in pbar:
        #     print(data.x)
        for idx, data in enumerate(pbar):
            data = process_data(data, use_edge_attr)
            #data.rad = torch.nn.Parameter(torch.ones(data.batch.size(0)))
            # print("data: ", data)
            att, loss_dict, clf_logits = run_one_batch(data.to(self.device), epoch)

            exp_labels = data.edge_label.data.cpu()
            precision_at_k = self.get_precision_at_k(att, exp_labels, self.k, data.batch, data.edge_index)
            desc, _, _, _, _, _ = self.log_epoch(epoch, phase, loss_dict, exp_labels, att, precision_at_k,
                                                 data.y.data.cpu(), clf_logits, batch=True)
            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v

            all_exp_labels.append(exp_labels), all_att.append(att), all_precision_at_k.extend(precision_at_k)
            all_clf_labels.append(data.y.data.cpu()), all_clf_logits.append(clf_logits)

            if idx == loader_len - 1:
                all_exp_labels, all_att = torch.cat(all_exp_labels), torch.cat(all_att),
                all_clf_labels, all_clf_logits = torch.cat(all_clf_labels), torch.cat(all_clf_logits)

                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / loader_len
                desc, att_auroc, precision, clf_acc, clf_roc, avg_loss = self.log_epoch(epoch, phase, all_loss_dict, all_exp_labels, all_att,all_precision_at_k, all_clf_labels, all_clf_logits, batch=False)
            pbar.set_description(desc)
            
        all_embs = []
        all_atts = []
        all_edge_indices = [] 
        print("in run one epoch")
        
        param_dict = dict(self.clf.named_parameters())
        A_phi = param_dict['convs.0.eps'].detach().cpu()  # [D, D]
        
        for data in data_loader:
            print("graphing data loop: ", data)
            att, _, _ = self.eval_one_batch(data, epoch)
            print("yz")
            #emb, trash = self.clf.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr) 
            emb = self.clf.get_graph_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
            print("graphing single emb shape: ", emb)
            all_embs.append(emb)
            print("ab")
            all_atts.append(att.view(-1, 1))
            print("cd")
            all_edge_indices.append(data.edge_index.T)  # shape [num_edges_in_batch, 2]
            print("ef")
            edge_index_last = data.edge_index
            print("gh")
            
            with torch.no_grad():
                p_np = emb.detach().cpu().numpy().astype(np.float32)
                d_np = (emb @ A_phi).detach().cpu().numpy().astype(np.float32)
                
                print("emb shape:", emb.shape)
                print("A_phi shape:", A_phi.shape)
                print("d_np shape:", d_np.shape)

                print("here2")
                all_embeddings = np.vstack([p_np, d_np])
                labels = np.array(['Primal'] * len(p_np) + ['Dual'] * len(d_np))

                print("here3")
                
                tsne = TSNE(n_components=2, perplexity=30, random_state=42)
                emb_2d = tsne.fit_transform(all_embeddings)

                print("here4")
                plt.figure(figsize=(8, 6))

                # Draw lines connecting matching primal/dual embeddings
                print("here4")
                for j in range(len(p_np)):
                    p = emb_2d[j]
                    d = emb_2d[j + len(p_np)]
                    plt.plot([p[0], d[0]], [p[1], d[1]], color='gray', linewidth=0.5, alpha=0.4, zorder=1)

                # Scatter points
                plt.scatter(emb_2d[:len(p_np), 0], emb_2d[:len(p_np), 1], label='Primal', alpha=0.8, zorder=2)
                plt.scatter(emb_2d[len(p_np):, 0], emb_2d[len(p_np):, 1], label='Dual', alpha=0.8, zorder=2)

                plt.title(f"t-SNE: Primal vs Dual Graph Embeddings", fontsize = 20)
                plt.legend(fontsize = 16)
                plt.tight_layout()
                plt.savefig(f'test/test_{epoch}')
                plt.close()
        emb = torch.cat(all_embs, dim=0)
        a = torch.cat(all_atts, dim=0)
        # # Concatenate all edge indices to match shape of `a`
        # print("graphing embs shape: ", emb.shape)
        # edge_index_full = torch.cat(all_edge_indices, dim=0).T  # shape [2, total_edges]        
        # pca = PCA(n_components=2)
        # emb_np = emb.detach().cpu().numpy()
        # emb_2d_np = pca.fit_transform(emb_np)
        # emb_2d = torch.tensor(emb_2d_np, dtype=torch.float32)
        # print('emb_2d: ', emb_2d.shape)
        # # emb_2d = torch.tensor(PCA(n_components=2).fit_transform(emb.detach().cpu().numpy()), dtype=torch.float32)
        # # Get PCA basis
        # W = torch.tensor(pca.components_.T, dtype=torch.float32)  # [D, 2]
        # A_phi_2d = W.T @ A_phi @ W
        # # A_phi_2d = 0.5 * (A_phi_2d - A_phi_2d.T)
        # A_phi_2d = 0.5 * A_phi_2d
        # self.visualize_edge_perturbations(epoch, emb_2d, edge_index_full, a, A_phi_2d, topk=50)
        
        # t-SNE projection
        # with torch.no_grad():
        #     p_np = emb.detach().cpu().numpy().astype(np.float32)
        #     d_np = (emb @ A_phi).detach().cpu().numpy().astype(np.float32)
            
        #     print("emb shape:", emb.shape)
        #     print("A_phi shape:", A_phi.shape)
        #     print("d_np shape:", d_np.shape)

        #     all_embeddings = np.vstack([p_np, d_np])
        #     labels = np.array(['Primal'] * len(p_np) + ['Dual'] * len(d_np))

        #     tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
        #     emb_2d = tsne.fit_transform(all_embeddings)

        #     plt.figure(figsize=(8, 6))

        #     # Draw lines connecting matching primal/dual embeddings
        #     for j in range(len(p_np)):
        #         p = emb_2d[j]
        #         d = emb_2d[j + len(p_np)]
        #         plt.plot([p[0], d[0]], [p[1], d[1]], color='gray', linewidth=0.5, alpha=0.4, zorder=1)

        #     # Scatter points
        #     plt.scatter(emb_2d[:len(p_np), 0], emb_2d[:len(p_np), 1], label='Primal', alpha=0.8, zorder=2)
        #     plt.scatter(emb_2d[len(p_np):, 0], emb_2d[len(p_np):, 1], label='Dual', alpha=0.8, zorder=2)

        #     plt.title(f"t-SNE: Primal vs Dual Graph Embeddings", fontsize = 20)
        #     plt.legend(fontsize = 16)
        #     plt.tight_layout()
        #     plt.savefig(f'test/test_{epoch}.png')
        #     plt.close()
        
        return att_auroc, precision, clf_acc, clf_roc, avg_loss

    def train(self, loaders, test_set, metric_dict, use_edge_attr):
        viz_set = self.get_viz_idx(test_set, self.dataset_name)
        for epoch in range(self.epochs):
            train_res = self.run_one_epoch(loaders['train'], epoch, 'train', use_edge_attr)
            valid_res = self.run_one_epoch(loaders['valid'], epoch, 'valid', use_edge_attr)
            test_res = self.run_one_epoch(loaders['test'], epoch, 'test', use_edge_attr)
            self.writer.add_scalar('gsat_train/lr', get_lr(self.optimizer), epoch)

            assert len(train_res) == 5
            main_metric_idx = 3 if 'ogb' in self.dataset_name else 2  # clf_roc or clf_acc
            if self.scheduler is not None:
                self.scheduler.step(valid_res[main_metric_idx])

            r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
            if (r == self.final_r or self.fix_r) and epoch > 10 and ((valid_res[main_metric_idx] > metric_dict['metric/best_clf_valid']) or (valid_res[main_metric_idx] == metric_dict['metric/best_clf_valid'] and valid_res[4] < metric_dict['metric/best_clf_valid_loss'])):
                metric_dict = {'metric/best_clf_epoch': epoch, 'metric/best_clf_valid_loss': valid_res[4],
                               'metric/best_clf_train': train_res[main_metric_idx], 'metric/best_clf_valid': valid_res[main_metric_idx], 'metric/best_clf_test': test_res[main_metric_idx],
                               'metric/best_x_roc_train': train_res[0], 'metric/best_x_roc_valid': valid_res[0], 'metric/best_x_roc_test': test_res[0],
                               'metric/best_x_precision_train': train_res[1], 'metric/best_x_precision_valid': valid_res[1], 'metric/best_x_precision_test': test_res[1]}
                save_checkpoint(self.clf, self.model_dir, model_name='gsat_clf_epoch_' + str(epoch))
                save_checkpoint(self.extractor, self.model_dir, model_name='gsat_att_epoch_' + str(epoch))

            for metric, value in metric_dict.items():
                metric = metric.split('/')[-1]
                self.writer.add_scalar(f'gsat_best/{metric}', value, epoch)

            if self.num_viz_samples != 0 and (epoch % self.viz_interval == 0 or epoch == self.epochs - 1):
                if self.multi_label:
                    raise NotImplementedError
                for idx, tag in viz_set:
                    self.visualize_results(test_set, idx, epoch, tag, use_edge_attr)

            if epoch == self.epochs - 1:
                save_checkpoint(self.clf, self.model_dir, model_name='gsat_clf_epoch_' + str(epoch))
                save_checkpoint(self.extractor, self.model_dir, model_name='gsat_att_epoch_' + str(epoch))

            print(f'[Seed {self.random_state}, Epoch: {epoch}]: Best Epoch: {metric_dict["metric/best_clf_epoch"]}, '
                  f'Best Val Pred ACC/ROC: {metric_dict["metric/best_clf_valid"]:.3f}, Best Test Pred ACC/ROC: {metric_dict["metric/best_clf_test"]:.3f}, '
                  f'Best Test X AUROC: {metric_dict["metric/best_x_roc_test"]:.3f}')
            print('====================================')
            print('====================================')
            # with open('logsy.txt', 'a') as file:
            #     file.write(f'[Seed {self.random_state}, Epoch: {epoch}]: Best Epoch: {metric_dict["metric/best_clf_epoch"]}, Best Val Pred ACC/ROC: {metric_dict["metric/best_clf_valid"]:.3f}, Best Test Pred ACC/ROC: {metric_dict["metric/best_clf_test"]:.3f}, Best Test X AUROC: {metric_dict["metric/best_x_roc_test"]:.3f}')
        return metric_dict

    def log_epoch(self, epoch, phase, loss_dict, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch):
        desc = f'[Seed {self.random_state}, Epoch: {epoch}]: gsat_{phase}........., ' if batch else f'[Seed {self.random_state}, Epoch: {epoch}]: gsat_{phase} finished, '
        for k, v in loss_dict.items():
            if not batch:
                self.writer.add_scalar(f'gsat_{phase}/{k}', v, epoch)
            desc += f'{k}: {v:.3f}, '

        eval_desc, att_auroc, precision, clf_acc, clf_roc = self.get_eval_score(epoch, phase, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch)
        desc += eval_desc
        return desc, att_auroc, precision, clf_acc, clf_roc, loss_dict['pred']

    def get_eval_score(self, epoch, phase, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch):
        clf_preds = get_preds(clf_logits, self.multi_label)
        clf_acc = 0 if self.multi_label else (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]

        if batch:
            return f'clf_acc: {clf_acc:.3f}', None, None, None, None

        precision_at_k = np.mean(precision_at_k)
        clf_roc = 0
        if 'ogb' in self.dataset_name:
            evaluator = Evaluator(name='-'.join(self.dataset_name.split('_')))
            clf_roc = evaluator.eval({'y_pred': clf_logits, 'y_true': clf_labels})['rocauc']

        att_auroc, bkg_att_weights, signal_att_weights = 0, att, att
        if np.unique(exp_labels).shape[0] > 1:
            att_auroc = roc_auc_score(exp_labels, att)
            bkg_att_weights = att[exp_labels == 0]
            signal_att_weights = att[exp_labels == 1]

        self.writer.add_histogram(f'gsat_{phase}/bkg_att_weights', bkg_att_weights, epoch)
        self.writer.add_histogram(f'gsat_{phase}/signal_att_weights', signal_att_weights, epoch)
        self.writer.add_scalar(f'gsat_{phase}/clf_acc/', clf_acc, epoch)
        self.writer.add_scalar(f'gsat_{phase}/clf_roc/', clf_roc, epoch)
        self.writer.add_scalar(f'gsat_{phase}/att_auroc/', att_auroc, epoch)
        self.writer.add_scalar(f'gsat_{phase}/precision@{self.k}/', precision_at_k, epoch)
        self.writer.add_scalar(f'gsat_{phase}/avg_bkg_att_weights/', bkg_att_weights.mean(), epoch)
        self.writer.add_scalar(f'gsat_{phase}/avg_signal_att_weights/', signal_att_weights.mean(), epoch)
        self.writer.add_pr_curve(f'PR_Curve/gsat_{phase}/', exp_labels, att, epoch)

        desc = f'clf_acc: {clf_acc:.3f}, clf_roc: {clf_roc:.3f}, ' + \
               f'att_roc: {att_auroc:.3f}, att_prec@{self.k}: {precision_at_k:.3f}'
        return desc, att_auroc, precision_at_k, clf_acc, clf_roc

    def get_precision_at_k(self, att, exp_labels, k, batch, edge_index):
        precision_at_k = []
        for i in range(batch.max()+1):
            nodes_for_graph_i = batch == i
            edges_for_graph_i = nodes_for_graph_i[edge_index[0]] & nodes_for_graph_i[edge_index[1]]
            labels_for_graph_i = exp_labels[edges_for_graph_i]
            mask_log_logits_for_graph_i = att[edges_for_graph_i]
            precision_at_k.append(labels_for_graph_i[np.argsort(-mask_log_logits_for_graph_i)[:k]].sum().item() / k)
        return precision_at_k

    def get_viz_idx(self, test_set, dataset_name):
        y_dist = test_set.data.y.numpy().reshape(-1)
        num_nodes = np.array([each.x.shape[0] for each in test_set])
        classes = np.unique(y_dist)
        res = []
        for each_class in classes:
            tag = 'class_' + str(each_class)
            if dataset_name == 'Graph-SST2':
                condi = (y_dist == each_class) * (num_nodes > 5) * (num_nodes < 10)  # in case too short or too long
                candidate_set = np.nonzero(condi)[0]
            else:
                candidate_set = np.nonzero(y_dist == each_class)[0]
            idx = np.random.choice(candidate_set, self.num_viz_samples, replace=False)
            res.append((idx, tag))
        print(res)
        return res

    def visualize_results(self, test_set, idx, epoch, tag, use_edge_attr):
        viz_set = test_set[idx]
        data = next(iter(DataLoader(viz_set, batch_size=len(idx), shuffle=False)))
        data = process_data(data, use_edge_attr)
        batch_att, _, clf_logits = self.eval_one_batch(data.to(self.device), epoch)
        imgs = []
        for i in tqdm(range(len(viz_set))):
            mol_type, coor = None, None
            if self.dataset_name == 'mutag':
                node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
                mol_type = {k: node_dict[v.item()] for k, v in enumerate(viz_set[i].node_type)}
            elif self.dataset_name == 'Graph-SST2':
                mol_type = {k: v for k, v in enumerate(viz_set[i].sentence_tokens)}
                num_nodes = data.x.shape[0]
                x = np.linspace(0, 1, num_nodes)
                y = np.ones_like(x)
                coor = np.stack([x, y], axis=1)
            elif self.dataset_name == 'ogbg_molhiv':
                element_idxs = {k: int(v+1) for k, v in enumerate(viz_set[i].x[:, 0])}
                mol_type = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v)) for k, v in element_idxs.items()}
            elif self.dataset_name == 'mnist':
                raise NotImplementedError

            node_subset = data.batch == i
            _, edge_att = subgraph(node_subset, data.edge_index, edge_attr=batch_att)

            node_label = viz_set[i].node_label.reshape(-1) if viz_set[i].get('node_label', None) is not None else torch.zeros(viz_set[i].x.shape[0])
            fig, img = visualize_a_graph(viz_set[i].edge_index, edge_att, node_label, self.dataset_name, norm=self.viz_norm_att, mol_type=mol_type, coor=coor)
            imgs.append(img)
        imgs = np.stack(imgs)
        self.writer.add_images(tag, imgs, epoch, dataformats='NHWC')

    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    def sampling(self, att_log_logits, epoch, training):
        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern


class ExtractorMLP(nn.Module):
    def __init__(self, hidden_size, shared_config):
        super().__init__()
        self.learn_edge_att = shared_config['learn_edge_att']
        dropout_p = shared_config['extractor_dropout_p']
        
        self.radius = torch.nn.Parameter(torch.ones(10000,1))

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch, mess=None):
        if mess is not None:
            print(mess, " to forward::ExtractorMLP")
       #print('emb in extractormlp forward: ', emb)
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            #print('f12', f12)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits


def train_gsat_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, method_name, device, random_state, c1, c2, c3, c4, c5):
    print('====================================')
    print('====================================')
    print(f'[INFO] Using device: {device}')
    print(f'[INFO] Using random_state: {random_state}')
    print(f'[INFO] Using dataset: {dataset_name}')
    print(f'[INFO] Using model: {model_name}')

    set_seed(random_state)

    model_config = local_config['model_config']
    data_config = local_config['data_config']
    method_config = local_config[f'{method_name}_config']
    shared_config = local_config['shared_config']
    assert model_config['model_name'] == model_name
    assert method_config['method_name'] == method_name

    batch_size, splits = data_config['batch_size'], data_config.get('splits', None)
    loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(data_dir, dataset_name, batch_size, splits, random_state, data_config.get('mutag_x', False))

    model_config['deg'] = aux_info['deg']
    model = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device)
    print('====================================')
    print('====================================')

    log_dir.mkdir(parents=True, exist_ok=True)
    if not method_config['from_scratch']:
        print('[INFO] Pretraining the model...')
        train_clf_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, device, random_state,
                           model=model, loaders=loaders, num_class=num_class, aux_info=aux_info)
        pretrain_epochs = local_config['model_config']['pretrain_epochs'] - 1
        load_checkpoint(model, model_dir=log_dir, model_name=f'epoch_{pretrain_epochs}')
    else:
        print('[INFO] Training both the model and the attention from scratch...')

    extractor = ExtractorMLP(model_config['hidden_size'], shared_config).to(device)
    lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
    optimizer = torch.optim.Adam(list(extractor.parameters()) + list(model.parameters()), lr=lr, weight_decay=wd)
    #print(list(extractor.parameters()) + list(model.parameters()))
    #print("optim: ", optimizer)
    print("model: ")
    for name, param in list(model.named_parameters()):
        print(name, param.shape)

    scheduler_config = method_config.get('scheduler', {})
    scheduler = None if scheduler_config == {} else ReduceLROnPlateau(optimizer, mode='max', **scheduler_config)

    writer = Writer(log_dir=log_dir)
    hparam_dict = {**model_config, **data_config}
    hparam_dict = {k: str(v) if isinstance(v, (dict, list)) else v for k, v in hparam_dict.items()}
    metric_dict = deepcopy(init_metric_dict)
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)

    print('====================================')
    print('[INFO] Training GSAT...')
    gsat = GSAT(model, extractor, optimizer, scheduler, writer, device, log_dir, dataset_name, num_class, aux_info['multi_label'], random_state, method_config, shared_config, c1, c2, c3, c4, c5)
    metric_dict = gsat.train(loaders, test_set, metric_dict, model_config.get('use_edge_attr', True))
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
    
        
    param_dict = dict(gsat.clf.named_parameters())
    A_phi = param_dict['convs.0.eps'].detach().cpu()  # [D, D]
    
    file_path = 'my_tensor1.pt'

    # Save the tensor to the file
    torch.save(A_phi, file_path)
    
    
    return hparam_dict, metric_dict


def dwain(c1, c2, c3, c4, c5):
    import argparse
    parser = argparse.ArgumentParser(description='Train GSAT')
    parser.add_argument('--dataset', type=str, help='dataset used')
    parser.add_argument('--backbone', type=str, help='backbone model used')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu')
    args = parser.parse_args()
    dataset_name = args.dataset
    model_name = args.backbone
    cuda_id = args.cuda

    torch.set_num_threads(1)
    config_dir = Path('./configs')
    method_name = 'GSAT'

    print('====================================')
    print('====================================')
    print(f'[INFO] Running {method_name} on {dataset_name} with {model_name}')
    print('====================================')

    global_config = yaml.safe_load((config_dir / 'global_config.yml').open('r'))
    local_config_name = get_local_config_name(model_name, dataset_name)
    local_config = yaml.safe_load((config_dir / local_config_name).open('r'))

    data_dir = Path(global_config['data_dir'])
    num_seeds = global_config['num_seeds']

    time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')

    metric_dicts = []
    for random_state in range(num_seeds):
        log_dir = data_dir / dataset_name / 'logs' / (time + '-' + dataset_name + '-' + model_name + '-seed' + str(random_state) + '-' + method_name)
        hparam_dict, metric_dict = train_gsat_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, method_name, device, random_state, c1, c2, c3, c4, c5)
        metric_dicts.append(metric_dict)

    log_dir = data_dir / dataset_name / 'logs' / (time + '-' + dataset_name + '-' + model_name + '-seed99-' + method_name + '-stat')
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = Writer(log_dir=log_dir)
    write_stat_from_metric_dicts(hparam_dict, metric_dicts, writer)

def main():
    for c1 in [1]:
        for c2 in [1]:
            for c3 in [1]:
                for c4 in [0.01]:
                    for c5 in [1]:
                        # with open('logsy.txt', 'a') as file:
                        #     file.write(f"new run: {c1}, {c2}, {c3}, {c4}, {c5}\n")
                        dwain(c1, c2, c3, c4, c5)

if __name__ == '__main__':
    main()
