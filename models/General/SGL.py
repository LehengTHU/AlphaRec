import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base.abstract_model import AbstractModel
from .base.abstract_RS import AbstractRS
from .base.abstract_data import AbstractData
from tqdm import tqdm

from scipy.sparse import csr_matrix

import random
import scipy.sparse as sp

class SGL_RS(AbstractRS):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1

    def train_one_epoch(self, epoch):
        running_loss, running_mf_loss, running_cl_loss, running_reg_loss, num_batches = 0, 0, 0, 0, 0

        pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total = len(self.data.train_loader))
        for batch_i, batch in pbar:          
            
            batch = [x.cuda(self.device) for x in batch]
            users, pos_items, users_pop, pos_items_pop = batch[0], batch[1], batch[2], batch[3]

            if self.args.infonce == 0 or self.args.neg_sample != -1:
                neg_items = batch[4]
                neg_items_pop = batch[5]

            self.model.train()
            dropped_adj1 = self.model.get_enhanced_adj(self.data.ui_mat, self.model.droprate)
            dropped_adj2 = self.model.get_enhanced_adj(self.data.ui_mat, self.model.droprate)
            mf_loss, cl_loss, reg_loss = self.model(users, pos_items, neg_items, dropped_adj1, dropped_adj2)
            loss = mf_loss + cl_loss + reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach().item()
            running_mf_loss += mf_loss.detach().item()
            running_cl_loss += cl_loss.detach().item()
            running_reg_loss += reg_loss.detach().item()
            num_batches += 1
        return [running_loss/num_batches, running_mf_loss/num_batches, running_cl_loss/num_batches, running_reg_loss/num_batches]

class SGL_Data(AbstractData):
    def __init__(self, args):
        super().__init__(args)
    
    def add_special_model_attr(self, args):
        try:
            self.ui_mat = sp.load_npz(self.path + '/ui_mat.npz')
            print("successfully loaded ui_mat...")
        except:
            self.trainItem = np.array(self.trainItem)
            self.trainUser = np.array(self.trainUser)
            self.ui_mat = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                    shape=(self.n_users, self.n_items))
            sp.save_npz(self.path + '/ui_mat.npz', self.ui_mat)
            print("successfully saved ui_mat...")

class SGL(AbstractModel):
    def __init__(self, args, data) -> None:
        super().__init__(args, data)
        self.temp_cl = args.temp_cl
        self.lambda_cl = args.lambda_cl
        self.droprate = args.droprate

    def compute(self, perturbed_adj=None):
        users_emb = self.embed_user.weight
        items_emb = self.embed_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]


        for layer in range(self.n_layers):
            if perturbed_adj is not None:
                all_emb = torch.sparse.mm(perturbed_adj, all_emb)
            else:
                all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        users, items = torch.split(light_out, [self.data.n_users, self.data.n_items])

        return users, items
    
    def get_enhanced_adj(self, ui_mat, droprate):
        adj_shape = ui_mat.get_shape()
        edge_count = ui_mat.count_nonzero()
        row_idx, col_idx = ui_mat.nonzero()
        keep_idx = random.sample(range(edge_count), int(edge_count * (1 - droprate)))
        user_np = np.array(row_idx)[keep_idx]
        item_np = np.array(col_idx)[keep_idx]
        edges = np.ones_like(user_np, dtype=np.float32)
        dropped_adj = sp.csr_matrix((edges, (user_np, item_np)), shape=adj_shape)

        adj_shape = dropped_adj.get_shape()
        n_nodes = adj_shape[0]+adj_shape[1]
        (user_np_keep, item_np_keep) = dropped_adj.nonzero()
        ratings_keep = dropped_adj.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])),shape=(n_nodes, n_nodes),dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T

        shape = tmp_adj.get_shape()
        rowsum = np.array(tmp_adj.sum(1))

        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(tmp_adj)
        norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)

        coo = norm_adj_mat.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        enhanced_adj_mat = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

        return enhanced_adj_mat.coalesce().cuda(self.device)

    def InfoNCE(self, view1, view2, temperature, b_cos = True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score+10e-6)
        return torch.mean(cl_loss)

    def cal_cl_loss(self, idx, perturbed_mat1, perturbed_mat2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda(self.device)
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda(self.device)
        user_view_1, item_view_1 = self.compute(perturbed_mat1)
        user_view_2, item_view_2 = self.compute(perturbed_mat2)
        # view1 = torch.cat((user_view_1[u_idx],item_view_1[i_idx]),0)
        # view2 = torch.cat((user_view_2[u_idx],item_view_2[i_idx]),0)
        user_cl_loss = self.InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp_cl)
        item_cl_loss = self.InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp_cl)
        return user_cl_loss + item_cl_loss
        # return self.InfoNCE(view1,view2,self.temp_cl)

    def forward(self, users, pos_items, neg_items, dropped_adj1, dropped_adj2):
        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        # contrastive loss
        cl_loss = self.lambda_cl * self.cal_cl_loss([users,pos_items], dropped_adj1, dropped_adj2)

        # main loss
        # use cosine similarity as prediction score
        if(self.train_norm == True):
            users_emb = F.normalize(users_emb, dim = -1)
            pos_emb = F.normalize(pos_emb, dim = -1)
            neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-6)
        mf_loss = torch.negative(torch.mean(maxi))

        # regularizer loss
        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return mf_loss, cl_loss, reg_loss
    

