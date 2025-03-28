import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base.abstract_model import AbstractModel
from .base.abstract_RS import AbstractRS
from .base.abstract_data import AbstractData, helper_load, helper_load_train
from tqdm import tqdm

from .base.evaluator import ProxyEvaluator
from .base.utils import *

from functools import partial
from .MoE import MoE

class AlphaRec_RS(AbstractRS):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)

    def train_one_epoch(self, epoch):
        running_loss, num_batches = 0, 0

        pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total=len(self.data.train_loader))
        for batch_i, batch in pbar:

            batch = [x.to(self.device) for x in batch]
            users, pos_items, users_pop, pos_items_pop = batch[0], batch[1], batch[2], batch[3]

            if self.args.infonce == 0 or self.args.neg_sample != -1:
                neg_items = batch[4]
                neg_items_pop = batch[5]

            self.model.train()

            loss = self.model(users, pos_items, neg_items)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach().item()
            num_batches += 1

        return [running_loss / num_batches]


class AlphaRec_Data(AbstractData):
    def __init__(self, args):
        super().__init__(args)

    def add_special_model_attr(self, args):
        self.lm_model = args.lm_model
        self.random_user_emb: bool = args.random_user_emb
        loading_path = args.data_path + args.dataset + '/item_info/'
        embedding_path_dict = {
            'bert': 'item_cf_embeds_bert_array.npy',
            'roberta': 'item_cf_embeds_roberta_array.npy',
            'v2': 'item_cf_embeds_array.npy',
            'v3': 'item_cf_embeds_large3_array.npy',
            'v3_shuffle': "item_cf_embeds_large3_array_shuffle.npy",
            'llama2_7b': 'item_cf_embeds_llama2_7b_array.npy',
            'llama3_7b': 'item_cf_embeds_llama3_7b_instruct_array.npy',
            'mistral_7b': 'item_cf_embeds_Norm_Mistral-7B-v0.1_array.npy',
            'SFR': 'item_cf_embeds_Norm_SFR-Embedding-Mistral_7b_array.npy',
            'GritLM_7b': 'item_cf_embeds_Norm_GritLM-7B_array.npy',
            'e5_7b': 'item_cf_embeds_Norm_e5-mistral-7b-instruct_array.npy',
            'echo_7b': 'item_cf_embeds_Norm_echo-mistral-7b-instruct-lasttoken_array.npy',
        }
        self.item_cf_embeds = np.load(loading_path + embedding_path_dict[self.lm_model])

        # self.train_user_list
        if not self.random_user_emb:
            def group_agg(group_data, embedding_dict, key='item_id'):
                ids = group_data[key].values
                embeds = [embedding_dict[id] for id in ids]
                embeds = np.array(embeds)
                return embeds.mean(axis=0)

            pairs = []
            for u, v in self.train_user_list.items():
                for i in v:
                    pairs.append((u, i))
            pairs = pd.DataFrame(pairs, columns=['user_id', 'item_id'])

            # User CF Embedding: the average of item embeddings
            groups = pairs.groupby('user_id')
            item_cf_embeds_dict = {i: self.item_cf_embeds[i] for i in range(len(self.item_cf_embeds))}
            user_cf_embeds = groups.apply(group_agg, embedding_dict=item_cf_embeds_dict, key='item_id')
            user_cf_embeds_dict = user_cf_embeds.to_dict()
            user_cf_embeds_dict = dict(sorted(user_cf_embeds_dict.items(), key=lambda item: item[0]))
            self.user_cf_embeds = np.array(list(user_cf_embeds_dict.values())) #TODO: random init embeddings


class AlphaRec(AbstractModel):
    def __init__(self, args, data) -> None:
        super().__init__(args, data)
        self.tau = args.tau
        self.embed_size = args.hidden_size
        self.lm_model = args.lm_model
        self.model_version = args.model_version
        self.random_user_emb: bool = args.random_user_emb

        self.init_item_cf_embeds = data.item_cf_embeds
        self.init_item_cf_embeds = torch.tensor(self.init_item_cf_embeds, dtype=torch.float32).to(self.device)
        self.init_embed_shape = self.init_item_cf_embeds.shape[1]

        if not self.random_user_emb:
            self.init_user_cf_embeds = data.user_cf_embeds
            self.init_user_cf_embeds = torch.tensor(self.init_user_cf_embeds, dtype=torch.float32).to(self.device)
        else:
            print('user embeddings were initalized randomly')
            self.init_user_cf_embeds = nn.Embedding(self.data.n_users, self.init_embed_shape)
            nn.init.xavier_normal_(self.init_user_cf_embeds.weight)



        # To keep the same parameter size
        multiplier_dict = {
            'bert': 8,
            'roberta': 8,
            'v2': 2,
            'v3': 1 / 2,
            'v3_shuffle': 1 / 2,
        }
        if (self.lm_model in multiplier_dict):
            multiplier = multiplier_dict[self.lm_model]
        else:
            multiplier = 9 / 32  # for dimension = 4096

        if (self.model_version == 'homo'):  # Linear mapping
            self.mlp = nn.Sequential(
                nn.Linear(self.init_embed_shape, self.embed_size, bias=False)  # homo
            )
            if self.random_user_emb:
                self.mlp_user = nn.Sequential(
                    nn.Linear(self.init_embed_shape, self.embed_size, bias=False)  # homo
                )

        else:  # MLP
            # self.mlp = nn.Sequential(
            #     nn.Linear(self.init_embed_shape, int(multiplier * self.init_embed_shape)),
            #     nn.LeakyReLU(),
            #     nn.Linear(int(multiplier * self.init_embed_shape), self.embed_size)
            # )
            self.mlp = MoE(d_in=self.init_embed_shape,
                           d_out=self.embed_size,
                           n_blocks=1,
                           d_block=8 * int(multiplier * self.init_embed_shape),
                           dropout=None,
                           activation='LeakyReLU',
                           num_experts=8,
                           gating_type='gumbel',
                           default_num_samples=10,
                           tau=1.0)

            if self.random_user_emb:
                self.mlp_user = nn.Sequential(
                    nn.Linear(self.init_embed_shape, self.embed_size, bias=False)  # homo
                )

    def init_embedding(self):
        pass

    def compute(self):
        users_cf_emb = self.mlp(self.init_user_cf_embeds) if not self.random_user_emb \
            else self.mlp_user(self.init_user_cf_embeds.weight)

        items_cf_emb = self.mlp(self.init_item_cf_embeds)

        users_emb = users_cf_emb
        items_emb = items_cf_emb

        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.data.n_users, self.data.n_items])

        return users, items

    def forward(self, users, pos_items, neg_items):

        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        if (self.train_norm):
            users_emb = F.normalize(users_emb, dim=-1)
            pos_emb = F.normalize(pos_emb, dim=-1)
            neg_emb = F.normalize(neg_emb, dim=-1)

        pos_ratings = torch.sum(users_emb * pos_emb, dim=-1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1),
                                   neg_emb.permute(0, 2, 1)).squeeze(dim=1)

        numerator = torch.exp(pos_ratings / self.tau)

        denominator = numerator + torch.sum(torch.exp(neg_ratings / self.tau), dim=1)

        ssm_loss = torch.mean(torch.negative(torch.log(numerator / denominator)))

        return ssm_loss

    # @torch.no_grad()
    @torch.inference_mode()
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.data.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).to(self.device)]
        items = all_items[torch.tensor(items).to(self.device)]

        if (self.pred_norm == True):
            users = F.normalize(users, dim=-1)
            items = F.normalize(items, dim=-1)
        items = torch.transpose(items, 0, 1)
        rate_batch = torch.matmul(users, items)  # user * item

        return rate_batch.cpu().detach().numpy()
