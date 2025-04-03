from torch import nn
import torch
import torch.nn.functional as F


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, d_in: int, d_inter: int, d_out: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = nn.Linear(d_in, d_inter)
        self.w2 = nn.Linear(d_inter, d_out)
        self.w3 = nn.Linear(d_in, d_inter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def kmeans_dot_product(x, num_clusters, num_iters=1000):
    """
    KMeans clustering using dot product similarity, with centroid change reporting.
    x: Tensor of shape (n_samples, embed_dim), unnormalized
    """
    n_samples, dim = x.shape

    # Randomly choose initial centroids from the dataset
    indices = torch.randperm(n_samples)[:num_clusters]
    centroids = x[indices]
    print('start k means')
    for i in range(num_iters):
        # Compute dot product similarity
        sim = torch.matmul(x, centroids.T)  # (n_samples, num_clusters)

        # Assign each point to the nearest centroid
        labels = torch.argmax(sim, dim=1)

        # Recompute centroids
        new_centroids = torch.stack([
            x[labels == k].mean(dim=0) if (labels == k).sum() > 0 else centroids[k]
            for k in range(num_clusters)
        ])

        # Compute and report centroid shift (mean L2 diff)
        diff = torch.norm(centroids - new_centroids, dim=1).mean()
        print(f"Iteration {i + 1}: mean centroid shift = {diff.item():.6f}")

        # Check for convergence
        if diff < 1e-5:
            print("Converged.")
            break

        centroids = new_centroids

    return labels, centroids


def count_cluster_sizes(labels, num_clusters):
    counts = torch.bincount(labels, minlength=num_clusters)
    for i, count in enumerate(counts):
        print(f"Cluster {i}: {count.item()} items")
    return counts


import torch
import torch.nn.functional as F

def assign_users_to_centroids(user_embeds, centroids, hard=True):
    """
    Assign users to item centroids using dot product similarity.

    Args:
        user_embeds (Tensor): User embeddings, shape (num_users, embed_dim)
        centroids (Tensor): Centroid embeddings, shape (num_clusters, embed_dim)
        hard (bool): If True, return hard assignments (argmax);
                     if False, return soft assignments (probabilities)

    Returns:
        Tensor:
            If hard=True: shape (num_users,), containing cluster indices.
            If hard=False: shape (num_users, num_clusters), containing probabilities.
    """
    # Compute dot product similarity
    sim = torch.matmul(user_embeds, centroids.T)  # shape: (num_users, num_clusters)

    if hard:
        # Hard assignment: return cluster index with highest similarity
        user_labels = torch.argmax(sim, dim=1)  # shape: (num_users,)
    else:
        # Soft assignment: return softmax probabilities over clusters
        user_labels = F.softmax(sim, dim=1)  # shape: (num_users, num_clusters)

    return user_labels


# def apply_cluster_mlps(embeds, cluster_labels, mlps, num_clusters):
#     """
#     Apply cluster-specific MLPs to embeddings in batch per cluster.
#
#     embeds: (N, embed_dim)
#     cluster_labels: (N,) with values in [0, num_clusters-1]
#     mlps: list or ModuleList of MLPs
#     """
#     output = torch.zeros((embeds.size(0), mlps[0][-1].out_features), device=embeds.device)
#
#     for cluster_id in range(num_clusters):
#         indices = (cluster_labels == cluster_id).nonzero(as_tuple=True)[0]
#         if indices.numel() == 0:
#             continue
#         cluster_embeds = embeds[indices]
#         output[indices] = mlps[cluster_id](cluster_embeds)
#
#     return output
def apply_cluster_mlps(embeds, cluster_labels, mlps, num_clusters, hard=True):
    """
    Apply cluster-specific MLPs to embeddings using hard or soft assignment.

    Args:
        embeds: Tensor of shape (N, embed_dim)
        cluster_labels:
            If hard=True: LongTensor of shape (N,) with values in [0, num_clusters-1]
            If hard=False: FloatTensor of shape (N, num_clusters) with soft probabilities
        mlps: list or ModuleList of MLPs (one per cluster)
        num_clusters: int
        hard: bool

    Returns:
        output: Tensor of shape (N, output_dim)
    """
    N = embeds.size(0)
    output_dim = mlps[0][-1].out_features
    device = embeds.device

    if hard:
        output = torch.zeros((N, output_dim), device=device)
        for cluster_id in range(num_clusters):
            indices = (cluster_labels == cluster_id).nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                continue
            cluster_embeds = embeds[indices]
            output[indices] = mlps[cluster_id](cluster_embeds)
    else:
        # Process all MLPs in one go for soft labels
        # Compute (N, output_dim) for each cluster and stack them: (num_clusters, N, output_dim)
        outputs = torch.stack([mlp(embeds) for mlp in mlps], dim=0)  # shape: (num_clusters, N, output_dim)

        # Transpose to (N, num_clusters, output_dim)
        outputs = outputs.permute(1, 0, 2)

        # cluster_labels: (N, num_clusters) -> unsqueeze to (N, num_clusters, 1)
        probs = cluster_labels.unsqueeze(2)

        # Weighted sum over clusters -> (N, output_dim)
        output = torch.sum(outputs * probs, dim=1)

    return output

def supcon_loss(user_emb, pos_item_embs, neg_item_embs, mask, tau, neg_sample):
    """
    Unified SupCon loss for both external and in-batch negative sampling modes.

    Args:
        user_emb:        [B, D]        - anchor user embeddings
        pos_item_embs:   [B, P, D]     - positive item embeddings (padded)
        neg_item_embs:   [B, N, D] or [B, P, D] - either sampled negatives or reused positives (for in-batch)
        mask:            [B, P]        - binary mask for valid positives
        tau:             float         - temperature
        neg_sample:      int           - - neg_sample == -1 → full in-batch negatives, neg_sample == -2 → one negative per other user, neg_sample > 0   → external negatives
    Returns:
        Scalar SupCon loss
    """
    # Normalize all embeddings
    user_emb = F.normalize(user_emb, dim=-1)  # [B, D]
    pos_item_embs = F.normalize(pos_item_embs, dim=-1)  # [B, P, D]
    neg_item_embs = F.normalize(neg_item_embs, dim=-1)  # shape depends on mode

    B, P, D = pos_item_embs.shape
    user_exp = user_emb.unsqueeze(1).expand(-1, P, -1)  # [B, P, D]

    # Positive similarities: [B, P]
    pos_sim = torch.exp(torch.sum(user_exp * pos_item_embs, dim=-1) / tau)

    if neg_sample == -1:
        assert False
        # ---------- IN-BATCH NEGATIVE SAMPLING ----------
        # Flatten all positive items across batch
        all_items_flat = neg_item_embs.view(B * P, D)  # [B*P, D]
        # all_items_flat = all_items_flat.detach()                  # optional: prevent gradients through negs

        # Similarities: [B, B*P]
        sim_matrix = torch.matmul(user_emb, all_items_flat.T) / tau
        sim_matrix = torch.exp(sim_matrix)

        # Create mask to exclude each user's own positives from denominator
        neg_mask = torch.ones((B, B * P), device=user_emb.device)
        for i in range(B):
            valid_p = int(mask[i].sum().item())
            neg_mask[i, i * P: i * P + valid_p] = 0  # zero out self-positives

        # Denominator: sum over other users' positives
        neg_denom = (sim_matrix * neg_mask).sum(dim=1, keepdim=True)  # [B, 1]
        denom = neg_denom + pos_sim  # [B, P] - broadcast adds back self-positives

    elif neg_sample == -2:
        assert False
        # 1 negative from each other user
        # Input: neg_item_embs [B-1, D] → already pre-sampled in forward
        sim = torch.exp(torch.sum(user_emb.unsqueeze(1) * neg_item_embs, dim=-1) / tau)  # [B, B-1]
        neg_sum = sim.sum(dim=1, keepdim=True)  # [B, 1]
        denom = pos_sim + neg_sum  # [B, P]

    else:
        # ---------- EXTERNAL NEGATIVE SAMPLING ----------
        # Compute [B, N] similarities between users and their negatives
        neg_sim = torch.exp(
            torch.bmm(user_emb.unsqueeze(1), neg_item_embs.transpose(1, 2)).squeeze(1) / tau
        )  # [B, N]

        neg_sum = neg_sim.sum(dim=1, keepdim=True)  # [B, 1]
        denom = pos_sim + neg_sum.expand(-1, P)  # [B, P]

    # Compute log-probabilities for positives
    log_prob = torch.log(pos_sim / (denom + 1e-8))  # [B, P]

    # Apply mask to ignore padding
    masked_log_prob = log_prob * mask  # [B, P]
    user_loss = -masked_log_prob.sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # [B]

    return user_loss.mean()


from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np
from collections import defaultdict


def find_user_connected_components(user_item_matrix):
    """
    Given a user-item binary interaction matrix, returns connected components of users.

    Parameters:
        user_item_matrix (array-like or csr_matrix): shape (n_users, n_items)
            Binary matrix where 1 indicates interaction.

    Returns:
        clusters_dict (dict): mapping of cluster_id -> list of user indices
        labels (np.ndarray): cluster ID assigned to each user
        n_components (int): number of connected components
    """
    # Ensure matrix is in CSR sparse format
    X = csr_matrix(user_item_matrix)
    # Compute user-user binary co-interaction matrix
    user_user_shared = X @ X.T
    user_user_shared.data = np.where(user_user_shared.data > 0, 1, 0)  # binarize connections

    # Compute connected components in the graph
    n_components, labels = connected_components(csgraph=user_user_shared, directed=False, connection='weak')
    print(f'n components:{n_components}')
    # Organize users into clusters
    clusters_dict = defaultdict(list)
    for user_id, cluster_id in enumerate(labels):
        clusters_dict[cluster_id].append(user_id)

    print('number of users in every cluster:')
    [print(len(v)) for v in clusters_dict.values()]
    return clusters_dict, labels, n_components


import numpy as np
from scipy.sparse import csr_matrix, vstack
from typing import Dict, Tuple


def get_user_neighbors_and_union_items(
        interaction_matrix: csr_matrix
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Efficient version using matrix ops:
    1. Finds user-user intersections via m @ m.T
    2. Collects union of items per user and their neighbors

    Returns:
        Dict[user_id] = (neighbor_user_ids, union_item_ids)
    """
    result = {}
    n_users = interaction_matrix.shape[0]

    # User-user intersection: shared items count
    user_user_shared = interaction_matrix @ interaction_matrix.T  # shape (n_users, n_users)
    user_user_shared.setdiag(0)  # remove self-intersections

    for user_id in range(n_users):
        neighbors = user_user_shared[user_id].nonzero()[1]  # users with shared items

        if neighbors.size > 0:
            rows_to_union = vstack([interaction_matrix[user_id], interaction_matrix[neighbors]])
        else:
            rows_to_union = interaction_matrix[user_id]

        # Get all unique items interacted with by user or neighbors
        union_items = rows_to_union.sum(axis=0).nonzero()[1]

        num_union_items = union_items.size

        result[user_id] = (neighbors, union_items, num_union_items)

    return result
