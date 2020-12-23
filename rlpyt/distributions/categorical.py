
import torch

from rlpyt.distributions.base import Distribution
from rlpyt.distributions.discrete import DiscreteMixin
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import valid_mean, select_at_indexes, batch_pairwise_joint_entropy_mean_T_B, batch_pairwise_cross_entropy_mean_T_B

EPS = 1e-8

DistInfo = namedarraytuple("DistInfo", ["prob"])


class Categorical(DiscreteMixin, Distribution):
    """Multinomial distribution over a discrete domain."""

    def kl(self, old_dist_info, new_dist_info):
        p = old_dist_info.prob
        q = new_dist_info.prob
        return torch.sum(p * (torch.log(p + EPS) - torch.log(q + EPS)), dim=-1)

    def mean_kl(self, old_dist_info, new_dist_info, valid=None):
        return valid_mean(self.kl(old_dist_info, new_dist_info), valid)

    def sample(self, dist_info):
        """Sample from ``torch.multiomial`` over trailing dimension of
        ``dist_info.prob``."""
        p = dist_info.prob
        sample = torch.multinomial(p.view(-1, self.dim), num_samples=1)
        return sample.view(p.shape[:-1]).type(self.dtype)  # Returns indexes.

    def entropy(self, dist_info):
        p = dist_info.prob
        return -torch.sum(p * torch.log(p + EPS), dim=-1)

    def joint_entropy(self, dist_info_1, dist_info_2):
        p1, p2 = dist_info_1.prob, dist_info_2.prob  # Assumed to already be softmax
        joint = p1 * p2
        return -torch.sum(joint * torch.log(joint), dim=-1)  # Sum over action dimension

    def pairwise_joint_entropy(self, all_dist_info, index=2):
        return batch_pairwise_joint_entropy_mean_T_B(all_dist_info.prob, pair_dim=index)

    def pairwise_cross_entropy(self, all_dist_info, index=2):
        return batch_pairwise_cross_entropy_mean_T_B(all_dist_info.prob, pair_dim=index)

    def log_likelihood(self, indexes, dist_info):
        selected_likelihood = select_at_indexes(indexes, dist_info.prob)
        return torch.log(selected_likelihood + EPS)

    def likelihood_ratio(self, indexes, old_dist_info, new_dist_info):
        num = select_at_indexes(indexes, new_dist_info.prob)
        den = select_at_indexes(indexes, old_dist_info.prob)
        return (num + EPS) / (den + EPS)
