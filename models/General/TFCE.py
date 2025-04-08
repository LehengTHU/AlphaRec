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

from scipy.sparse import csr_matrix
import os
import json


def convert_np(obj):
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


class TFCE_RS(AbstractRS):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)
        print(f'number of GCN layers:{self.n_layers}')

    def train(self) -> None:
        # no need to train, just
        self.eval_and_check_early_stop(0)
        # visualize_and_save_log(self.base_path + 'stats.txt', self.dataset_name)

    def save_eval_results(self, n_ret, eval_name):
        # Construct the directory path
        dir_path = f"{self.model_name}_{eval_name}_results/{self.dataset_name}"
        os.makedirs(dir_path, exist_ok=True)  # Create directory if it doesn't exist

        # Construct the file path
        file_path = os.path.join(dir_path, f"{self.n_layers}.json")

        # Save the dictionary as JSON
        with open(file_path, 'w') as f:
            json.dump(convert_np(n_ret), f, indent=2)

    def eval_and_check_early_stop(self, epoch):
        self.model.eval()
        for i, evaluator in enumerate(self.evaluators):
            tt1 = time.time()
            is_best, temp_flag, n_ret = evaluation(self.args, self.data, self.model, epoch, self.base_path, evaluator,
                                                   self.eval_names[i])
            tt2 = time.time()
            print("Evaluating %d [%.1fs]: %s" % (i, tt2 - tt1, self.eval_names[i]))
            if temp_flag:
                self.flag = True
            self.save_eval_results(n_ret, self.eval_names[i])
        # checkpoint_buffer=save_checkpoint(self.model, epoch, self.base_path, self.checkpoint_buffer, self.args.max2keep)

        self.model.train()

    def execute(self):

        self.save_args()  # save the args
        # write args
        perf_str = str(self.args)
        with open(self.base_path + 'stats.txt', 'a') as f:
            f.write(perf_str + "\n")

        self.model, self.start_epoch = self.restore_checkpoint(self.model, self.base_path,
                                                               self.device)  # restore the checkpoint

        start_time = time.time()
        # train the model if not test only
        if not self.test_only:
            print("start training")
            self.train()
            # test the model

        end_time = time.time()
        print(f'training time: {end_time - start_time}')



class TFCE_Data(AbstractData):
    def __init__(self, args):
        super().__init__(args)

    def add_special_model_attr(self, args):
        self.lm_model = args.lm_model
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
        self.user_cf_embeds = np.array(list(user_cf_embeds_dict.values()))


class TFCE(AbstractModel):
    def __init__(self, args, data) -> None:
        super().__init__(args, data)
        self.tau = args.tau
        self.embed_size = args.hidden_size
        self.lm_model = args.lm_model
        self.model_version = args.model_version

        self.init_item_cf_embeds = data.item_cf_embeds
        self.init_item_cf_embeds = torch.tensor(self.init_item_cf_embeds, dtype=torch.float32).to(self.device)
        self.init_embed_shape = self.init_item_cf_embeds.shape[1]

        self.init_user_cf_embeds = data.user_cf_embeds
        self.init_user_cf_embeds = torch.tensor(self.init_user_cf_embeds, dtype=torch.float32).to(self.device)

        self.set_graph_embeddings()

    def init_embedding(self):
        pass

    def set_graph_embeddings(self):
        print('applying GCN to LLM embeddings')

        all_emb = torch.cat([self.init_user_cf_embeds, self.init_item_cf_embeds])

        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        self.init_user_cf_embeds, self.init_item_cf_embeds = torch.split(light_out,
                                                                         [self.data.n_users, self.data.n_items])

    def compute(self):
        return self.init_user_cf_embeds, self.init_item_cf_embeds

    def forward(self, users, pos_items, neg_items, mask):
        # if self.training:
        #     return torch.tensor(1.0, requires_grad=True)
        all_users, all_items = self.compute()
        # if not self.data.is_one_pos_item:
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

        if not self.data.is_one_pos_item:
            return supcon_loss(users_emb, pos_emb, neg_emb, mask, self.tau, 0)

        if (self.train_norm):
            users_emb = F.normalize(users_emb, dim=-1)
            pos_emb = F.normalize(pos_emb, dim=-1)
            neg_emb = F.normalize(neg_emb, dim=-1)

        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1),
                                   neg_emb.permute(0, 2, 1))

        pos_ratings = torch.sum(users_emb * pos_emb, dim=-1)
        numerator = torch.exp(pos_ratings / self.tau)
        # if self.data.is_one_pos_item:
        #     pos_ratings = torch.sum(users_emb * pos_emb, dim=-1)
        #     numerator = torch.exp(pos_ratings / self.tau)
        # else:
        #     pos_ratings = torch.sum(users_emb.unsqueeze(1) * pos_emb,
        #                             dim=-1)  # [B, L]
        #     numerator = torch.sum(torch.exp(pos_ratings / self.tau) * mask, dim=-1)  # [B]

        # if self.data.is_one_pos_item:
        #     neg_ratings = neg_ratings.squeeze(dim=1)
        # else:
        #     pass
        #     # neg_ratings = neg_ratings.expand(-1, pos_emb.shape[1], -1)

        denominator = numerator + torch.sum(torch.exp(neg_ratings / self.tau), dim=2)

        ssm_loss = torch.mean(torch.negative(torch.log(numerator / denominator)))
        # if self.data.is_one_pos_item:
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
