import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base.abstract_model import AbstractModel
from .base.abstract_RS import AbstractRS
from .base.abstract_data import AbstractData
from tqdm import tqdm

class XSimGCL_RS(AbstractRS):
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
            mf_loss, cl_loss, reg_loss = self.model(users, pos_items, neg_items)
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
    

class XSimGCL(AbstractModel):
    def __init__(self, args, data) -> None:
        super().__init__(args, data)
        self.temp_cl = args.temp_cl
        self.layer_cl = args.layer_cl
        self.lambda_cl = args.lambda_cl
        self.eps_XSimGCL = args.eps_XSimGCL

    def compute(self, perturbed=False):
        users_emb = self.embed_user.weight
        items_emb = self.embed_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = []
        emb_cl = all_emb
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            if perturbed:
                random_noise = torch.rand_like(all_emb).cuda(self.device) # add noise
                all_emb += torch.sign(all_emb) * F.normalize(random_noise, dim=-1) * self.eps_XSimGCL
            embs.append(all_emb)
            if layer==self.layer_cl-1:
                emb_cl = all_emb
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        users, items = torch.split(light_out, [self.data.n_users, self.data.n_items])
        users_cl, items_cl = torch.split(emb_cl, [self.data.n_users, self.data.n_items]) # view of noise

        if perturbed:
            return users, items, users_cl, items_cl
        return users, items
    
    def InfoNCE(self, view1, view2, temperature, b_cos = True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score+10e-6)
        return torch.mean(cl_loss)
    
    def cal_cl_loss(self, idx, user_view1,user_view2,item_view1,item_view2):
        # 算的一个batch中的
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda(self.device)
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda(self.device)
        user_cl_loss = self.InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp_cl)
        item_cl_loss = self.InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp_cl)
        return user_cl_loss + item_cl_loss

    def forward(self, users, pos_items, neg_items):
        all_users, all_items, all_users_cl, all_items_cl = self.compute(perturbed=True)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        # contrastive loss
        cl_loss = self.lambda_cl * self.cal_cl_loss([users,pos_items], all_users, all_users_cl, all_items, all_items_cl)

        # main loss
        # use cosine similarity to calculate the scores
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