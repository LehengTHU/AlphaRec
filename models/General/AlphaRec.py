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
from .utils import Expert, kmeans_dot_product, apply_cluster_mlps, assign_users_to_centroids, count_cluster_sizes, \
    supcon_loss

from .SparseMoE import SparseMoE
from .utils import find_user_connected_components

from scipy.sparse import csr_matrix
class AlphaRec_RS(AbstractRS):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)

    def train_one_epoch(self, epoch):
        running_loss, num_batches = 0, 0

        pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total=len(self.data.train_loader))
        for batch_i, batch in pbar:
            batch = [x.to(self.device) for x in batch]
            users, pos_items, users_pop, pos_items_pop, mask = batch[0], batch[1], batch[2], batch[3], \
                batch[6]

            if self.args.infonce == 0 or self.args.neg_sample != -1:
                neg_items = batch[4]
                neg_items_pop = batch[5]

            self.model.train()

            loss = self.model(users, pos_items, neg_items, mask)

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
            self.user_cf_embeds = np.array(list(user_cf_embeds_dict.values()))  # TODO: random init embeddings


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


        # cluster_dict, labels, n_components = find_user_connected_components(self.data.UserItemNet) did not work, since only one big component
        # from .utils import get_user_neighbors_and_union_items
        # user_neighbors = get_user_neighbors_and_union_items(self.data.UserItemNet)
        # print(f'maximum number of excluded items:{np.array([v[2] for v in user_neighbors.values()])}')

        self.is_kmeans = False
        self.num_clusters = 4
        if self.is_kmeans:
            # Apply KMeans with dot product
            self.item_cluster_labels, centroids = kmeans_dot_product(self.init_item_cf_embeds,
                                                                     num_clusters=self.num_clusters)
            # Save the centroids
            self.item_cf_centroids = centroids  # shape: (self.num_clusters, embedding_dim)
            self.item_cf_cluster_labels = self.item_cluster_labels  # shape: (num_items,)
            count_cluster_sizes(self.item_cluster_labels, num_clusters=self.num_clusters)

            # Assign cluster labels to users based on item centroids
            self.user_cluster_labels = assign_users_to_centroids(self.init_user_cf_embeds, self.item_cf_centroids)

            # Optional: print how many users per cluster
            print('users:')
            count_cluster_sizes(self.user_cluster_labels, num_clusters=self.num_clusters)

        self.is_embeds_learnable = False
        if self.is_embeds_learnable:
            print('embeds are learnable')
            self.init_item_cf_embeds = nn.Parameter(self.init_item_cf_embeds)
            self.init_user_cf_embeds = nn.Parameter(self.init_user_cf_embeds)

        self.k = 8
        self.is_batch_ens = False
        if self.is_batch_ens:
            print(f'+ adapter; k = {self.k}')
            self.r = nn.Parameter(
                torch.empty(self.k, self.init_embed_shape, device=self.device))
            nn.init.xavier_normal_(self.r)

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
            if self.is_kmeans:
                self.mlp = self.create_cluster_mlps(self.num_clusters, multiplier)
            else:
                self.mlp = nn.Sequential(
                    nn.Linear(self.init_embed_shape, int(multiplier * self.init_embed_shape)),
                    nn.LeakyReLU(),
                    nn.Linear(int(multiplier * self.init_embed_shape), self.embed_size)
                )

                # self.mlp = Expert(d_in=self.init_embed_shape, d_inter=int(multiplier * self.init_embed_shape),
                #                   d_out=self.embed_size)
                # self.mlp = MoE(d_in=self.init_embed_shape,
                #            d_out=self.embed_size,
                #            n_blocks=1,
                #            d_block=8 * int(multiplier * self.init_embed_shape),
                #            dropout=None,
                #            activation='LeakyReLU',
                #            num_experts=8,
                #            gating_type='gumbel',
                #            default_num_samples=10,
                #            tau=1.0)
                # self.mlp = SparseMoE(d_in=self.init_embed_shape,
                #                  d_out=self.embed_size,
                #                  n_blocks=1,
                #                  d_block_per_expert=int(multiplier * self.init_embed_shape),
                #                  dropout=None,
                #                  activation='LeakyReLU',
                #                  num_experts=8,
                #                  tau=1.0)
            if self.random_user_emb:
                self.mlp_user = nn.Sequential(
                    nn.Linear(self.init_embed_shape, self.embed_size, bias=False)  # homo
                )
            print('mlp:')
            print(self.mlp)

    def create_cluster_mlps(self, num_clusters, multiplier):
        mlps = nn.ModuleList()

        for cluster_idx in range(num_clusters):
            mlp = nn.Sequential(
                nn.Linear(self.init_embed_shape, int(multiplier * self.init_embed_shape)),
                nn.LeakyReLU(),
                nn.Linear(int(multiplier * self.init_embed_shape), self.embed_size)
            )
            mlps.append(mlp)
        return mlps

    def init_embedding(self):
        pass

    def compute(self):
        if self.is_batch_ens:
            def run_mlp(x, n_elements):
                # users_cf_emb = self.mlp(self.init_user_cf_embeds) no need
                # Expand input: (E, B, D)
                x_expanded = x.unsqueeze(0).expand(self.k, n_elements,
                                                   self.init_embed_shape)  # (E, B, D)
                r_expanded = self.r.unsqueeze(1)  # (E, 1, D)

                # Element-wise multiply
                x_scaled = x_expanded * r_expanded  # (E, B, D)

                # Flatten for processing through shared MLP
                x_flat = x_scaled.reshape(self.k * n_elements, self.init_embed_shape)

                # Shared MLP
                emb = self.mlp(x_flat)

                # Reshape back
                return emb.view(self.k, n_elements, -1).mean(dim=0)  # (E, B, output_dim)

            users_emb = run_mlp(self.init_user_cf_embeds, self.data.n_users)
            items_emb = run_mlp(self.init_item_cf_embeds, self.data.n_items)
        elif self.is_kmeans:
            # -------- USERS --------
            if not self.random_user_emb:
                user_embeds = self.init_user_cf_embeds
            else:
                user_embeds = self.init_user_cf_embeds.weight  # nn.Embedding

            users_cf_emb = apply_cluster_mlps(
                user_embeds,
                self.user_cluster_labels,
                self.mlp,
                num_clusters=self.num_clusters
            )

            # -------- ITEMS --------
            items_cf_emb = apply_cluster_mlps(
                self.init_item_cf_embeds,
                self.item_cluster_labels,
                self.mlp,
                num_clusters=self.num_clusters
            )

            # -------- FINAL --------
            users_emb = users_cf_emb
            items_emb = items_cf_emb
        else:
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

    def forward(self, users, pos_items, neg_items, mask):

        all_users, all_items = self.compute()
        # if not self.data.is_sample_pos_items:
        #     # padding index = -1; -> Step 1: Append a padding embedding at the end of all_items
        #     # TODO: is that okay?
        #     padding_emb = torch.zeros((1, all_items.size(1)), device=all_items.device).detach()
        #     all_items = torch.cat([all_items, padding_emb], dim=0)  # now all_items[-1] = padding
        #     # n_real_elements = torch.sum(pos_items != -1)
        #     # n_pad_elements = torch.sum(pos_items == -1)
        #     n_items_per_user = torch.sum(mask, dim=-1)
        #     # print(n_items_per_user)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        if not self.data.is_sample_pos_items:
            return supcon_loss(users_emb, pos_emb, neg_emb, mask, self.tau, 0)

        if (self.train_norm):
            users_emb = F.normalize(users_emb, dim=-1)
            pos_emb = F.normalize(pos_emb, dim=-1)
            neg_emb = F.normalize(neg_emb, dim=-1)

        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1),
                                   neg_emb.permute(0, 2, 1))

        pos_ratings = torch.sum(users_emb * pos_emb, dim=-1)
        numerator = torch.exp(pos_ratings / self.tau)
        # if self.data.is_sample_pos_items:
        #     pos_ratings = torch.sum(users_emb * pos_emb, dim=-1)
        #     numerator = torch.exp(pos_ratings / self.tau)
        # else:
        #     pos_ratings = torch.sum(users_emb.unsqueeze(1) * pos_emb,
        #                             dim=-1)  # [B, L]
        #     numerator = torch.sum(torch.exp(pos_ratings / self.tau) * mask, dim=-1)  # [B]

        # if self.data.is_sample_pos_items:
        #     neg_ratings = neg_ratings.squeeze(dim=1)
        # else:
        #     pass
        #     # neg_ratings = neg_ratings.expand(-1, pos_emb.shape[1], -1)

        denominator = numerator + torch.sum(torch.exp(neg_ratings / self.tau), dim=2)

        ssm_loss = torch.mean(torch.negative(torch.log(numerator / denominator)))
        # if self.data.is_sample_pos_items:
        #     ssm_loss = torch.mean(torch.negative(torch.log(numerator / denominator)))
        # else:
        #     print((1.0/n_items_per_user).shape)
        # print((1.0/n_items_per_user))
        # print((1.0/n_items_per_user)*torch.negative(torch.log(numerator / denominator)))
        # ssm_loss = torch.sum(torch.negative(torch.log(numerator / denominator))) / len(numerator)
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
