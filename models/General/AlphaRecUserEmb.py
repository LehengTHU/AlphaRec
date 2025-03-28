import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from .base.abstract_model import AbstractModel
from .base.abstract_RS import AbstractRS
from .AlphaRec import AlphaRec_Data, AlphaRec_RS, AlphaRec
from .base.abstract_data import AbstractData, helper_load, helper_load_train
from tqdm import tqdm

from .base.utils import *

from .MoE import MoE
from .SparseMoE import SparseMoE


class AlphaRecUserEmb_RS(AlphaRec_RS):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)

    # def train_one_epoch(self, epoch):
    #     running_loss, num_batches = 0, 0
    #
    #     pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total=len(self.data.train_loader))
    #     for batch_i, batch in pbar:
    #
    #         batch = [x.to(self.device) for x in batch]
    #         users, pos_items, users_pop, pos_items_pop = batch[0], batch[1], batch[2], batch[3]
    #
    #         if self.args.infonce == 0 or self.args.neg_sample != -1:
    #             neg_items = batch[4]
    #             neg_items_pop = batch[5]
    #
    #         self.model.train()
    #
    #         loss = self.model(users, pos_items, neg_items)
    #
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    #
    #         running_loss += loss.detach().item()
    #         num_batches += 1
    #
    #     return [running_loss / num_batches]


class AlphaRecUserEmb_Data(AlphaRec_Data):
    def __init__(self, args):
        super().__init__(args)
        self.user_embedding = None

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

        # w/o averaging for users
        # def group_agg(group_data, embedding_dict, key='item_id'):
        #     ids = group_data[key].values
        #     embeds = [embedding_dict[id] for id in ids]
        #     embeds = np.array(embeds)
        #     return embeds.mean(axis=0)
        #
        # # self.train_user_list
        # pairs = []
        # for u, v in self.train_user_list.items():
        #     for i in v:
        #         pairs.append((u, i))
        # pairs = pd.DataFrame(pairs, columns=['user_id', 'item_id'])
        #
        # # User CF Embedding: the average of item embeddings
        # groups = pairs.groupby('user_id')
        # item_cf_embeds_dict = {i: self.item_cf_embeds[i] for i in range(len(self.item_cf_embeds))}
        # user_cf_embeds = groups.apply(group_agg, embedding_dict=item_cf_embeds_dict, key='item_id')
        # user_cf_embeds_dict = user_cf_embeds.to_dict()
        # user_cf_embeds_dict = dict(sorted(user_cf_embeds_dict.items(), key=lambda item: item[0]))
        #
        # self.user_cf_embeds = np.array(list(user_cf_embeds_dict.values()))

    def getSparseGraph(self):

        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("finish loading adjacency matrix")
                norm_adj = pre_adj_mat
            # If there is no preprocessed adjacency matrix, generate one.
            except:
                print("generating adjacency matrix")
                s = time.time()
                adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                self.trainItem = np.array(self.trainItem)
                self.trainUser = np.array(self.trainUser)
                self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                              shape=(self.n_users, self.n_items))
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.tocsr()
                sp.save_npz(self.path + '/adj_mat.npz', adj_mat)
                print("successfully saved adj_mat...")

                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time.time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(self.device)

        return self.Graph


class AlphaRecUserEmb(AbstractModel):
    def __init__(self, args, data) -> None:
        self.multiplier_user_embed_dim = 1
        self.user_model_version = args.user_model_version
        super().__init__(args, data)
        print(self.user_model_version)
        self.tau = args.tau
        self.embed_size = args.hidden_size
        self.lm_model = args.lm_model
        self.model_version = args.model_version
        # self.init_user_cf_embeds = data.user_cf_embeds
        self.init_item_cf_embeds = data.item_cf_embeds

        # self.init_user_cf_embeds = torch.tensor(self.init_user_cf_embeds, dtype=torch.float32).to(self.device)
        self.init_item_cf_embeds = torch.tensor(self.init_item_cf_embeds, dtype=torch.float32).to(self.device)

        # self.init_embed_shape = self.init_user_cf_embeds.shape[1]
        self.init_embed_shape = self.init_item_cf_embeds.shape[1]  # TODO: is it okay?

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

        else:  # MLP
            # self.mlp = nn.Sequential(
            #     nn.Linear(self.init_embed_shape, int(multiplier * self.init_embed_shape)),
            #     nn.LeakyReLU(),
            #     nn.Linear(int(multiplier * self.init_embed_shape), self.embed_size)
            # )
            # self.mlp = MoE(d_in=self.init_embed_shape,
            #                d_out=self.embed_size,
            #                n_blocks=1,
            #                d_block=8*int(multiplier * self.init_embed_shape),
            #                dropout=None,
            #                activation='LeakyReLU',
            #                num_experts=32,
            #                gating_type='gumbel',
            #                default_num_samples=10,
            #                tau=0.5)
            self.mlp = SparseMoE(d_in=self.init_embed_shape,
                                 d_out=self.embed_size,
                                 n_blocks=1,
                                 d_block_per_expert=int(multiplier * self.init_embed_shape),
                                 dropout=None,
                                 activation='LeakyReLU',
                                 num_experts=8,
                                 tau=1.0)
        if self.user_model_version == 'homo':
            self.mlp_user = nn.Sequential(
                nn.Linear(self.multiplier_user_embed_dim * self.emb_dim, self.embed_size, bias=False)  # homo
            )
        elif self.user_model_version == 'mlp':
            self.mlp_user = nn.Sequential(
                nn.Linear(self.multiplier_user_embed_dim * self.emb_dim, self.multiplier_user_embed_dim * self.emb_dim),
                nn.LeakyReLU(),
                # nn.Dropout(p=0.2),
                nn.Linear(self.multiplier_user_embed_dim * self.emb_dim, self.embed_size)
            )
        elif self.user_model_version == 'emb':
            self.mlp_user = None
        else:
            assert False, 'only mlp, homo and emb are supported for user mapping'

        print('mlp:')
        print(self.mlp)

        if self.user_model_version != 'emb':
            print('mlp user:')
            print(self.mlp_user)
        else:
            print('no mlp for user')
        self.k = 32
        self.is_batch_ens = False
        if self.is_batch_ens:
            print('+ adapter')
            self.r = nn.Parameter(
                torch.empty(self.k, self.multiplier_user_embed_dim * self.emb_dim, device=self.device))
            nn.init.xavier_normal_(self.r)

    def init_embedding(self):
        # UserItemNet = csr_matrix((np.ones(len(self.data.trainUser)), (self.data.trainUser, self.data.trainItem)),
        #                               shape=(self.data.n_users, self.data.n_items))
        #
        # # only for users
        # # --- Truncated SVD embedding initialization --- #
        # print("initializing user embeddings with TruncatedSVD")  # <-- NEW
        # svd = TruncatedSVD(n_components=self.multiplier_user_embed_dim * self.emb_dim)  # <-- NEW
        # user_svd_embeddings = svd.fit_transform(UserItemNet)  # <-- NEW
        # user_svd_embeddings = normalize(user_svd_embeddings, norm='l2', axis=1) #sklearn
        # # Convert to PyTorch nn.Embedding
        # self.embed_user = nn.Embedding(self.data.n_users, self.multiplier_user_embed_dim * self.emb_dim).to(self.device)  # <-- NEW
        # with torch.no_grad():  # Optional: avoid tracking gradients for init
        #     self.embed_user.weight.data.copy_(
        #         torch.tensor(user_svd_embeddings, dtype=torch.float32).to(self.device)
        #     )  # <-- NEW
        # print("user embedding initialized")  # <-- NEW

        self.user_emb_dim = self.multiplier_user_embed_dim * self.emb_dim \
            if self.user_model_version != 'emd' else self.embed_size

        self.embed_user = nn.Embedding(self.data.n_users, self.user_emb_dim).to(self.device)
        nn.init.xavier_normal_(self.embed_user.weight)

    def compute(self):
        if self.is_batch_ens:
            # users_cf_emb = self.mlp(self.init_user_cf_embeds) no need
            # Expand input: (E, B, D)
            x_expanded = self.embed_user.weight.unsqueeze(0).expand(self.k, self.data.n_users,
                                                                    self.user_emb_dim)  # (E, B, D)
            r_expanded = self.r.unsqueeze(1)  # (E, 1, D)

            # Element-wise multiply
            x_scaled = x_expanded * r_expanded  # (E, B, D)

            # Flatten for processing through shared MLP
            x_flat = x_scaled.reshape(self.k * self.data.n_users, self.user_emb_dim)

            # Shared MLP
            users_emb = self.mlp_user(x_flat)

            # Reshape back
            users_emb = users_emb.view(self.k, self.data.n_users, -1).mean(dim=0)  # (E, B, output_dim)
        else:
            if self.mlp_user is not None:
                users_emb = self.mlp_user(self.embed_user.weight)
            else:
                users_emb = self.embed_user.weight
        items_cf_emb = self.mlp(self.init_item_cf_embeds)
        # users_emb = users_cf_emb

        items_emb = items_cf_emb

        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        # users, items = torch.split(embs, [self.data.n_users, self.data.n_items])
        # items = items[:,0,:]
        # users = torch.mean(users, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.data.n_users, self.data.n_items])

        return users, items

    def forward(self, users, pos_items, neg_items):

        all_users, all_items = self.compute()

        users_emb = all_users[users]
        # userEmb0 = self.embed_user(users)
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

        # regularizer = 0.5 * torch.norm(userEmb0) ** 2
        # regularizer = regularizer / self.batch_size
        # reg_loss = self.decay * regularizer

        return ssm_loss  # + reg_loss

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
