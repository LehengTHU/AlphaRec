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



def kmeans_dot_product(x, num_clusters, num_iters=100):
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


def assign_users_to_centroids(user_embeds, centroids):
    """
    Assign each user to the nearest item centroid using dot product similarity.
    """
    # user_embeds: (num_users, embed_dim)
    # centroids: (num_clusters, embed_dim)

    # Compute dot product similarity
    sim = torch.matmul(user_embeds, centroids.T)  # shape: (num_users, num_clusters)

    # Assign each user to the most similar centroid
    user_labels = torch.argmax(sim, dim=1)

    return user_labels


def apply_cluster_mlps(embeds, cluster_labels, mlps, num_clusters):
    """
    Apply cluster-specific MLPs to embeddings in batch per cluster.

    embeds: (N, embed_dim)
    cluster_labels: (N,) with values in [0, num_clusters-1]
    mlps: list or ModuleList of MLPs
    """
    output = torch.zeros((embeds.size(0), mlps[0][-1].out_features), device=embeds.device)

    for cluster_id in range(num_clusters):
        indices = (cluster_labels == cluster_id).nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            continue
        cluster_embeds = embeds[indices]
        output[indices] = mlps[cluster_id](cluster_embeds)

    return output