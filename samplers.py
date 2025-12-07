import numpy as np
import sys
import torch
from heapq import heapify, heappop

# ===========================================================
# Classe base
# ===========================================================
class Sampler:
    def __init__(self, ratio=0.1):
        self.ratio = ratio

    def select(self, *args, **kwargs):
        raise NotImplementedError


# ===========================================================
# Aleatória
# ===========================================================
class RandomSampler(Sampler):
    def select(self, size):
        sample_size = int(self.ratio * size)
        selected_indices = np.random.choice(size, size=sample_size, replace=False)
        return selected_indices


# ===========================================================
# Incerteza
# ===========================================================
class UncertaintySampler(Sampler):
    def select(self, uncertainty_scores):
        size = len(uncertainty_scores)
        sample_size = int(self.ratio * size)

        # Seleciona maiores incertezas
        topk = torch.topk(torch.tensor(uncertainty_scores), k=sample_size)
        return topk.indices.cpu().numpy()


# ===========================================================
# Uniforme (baseada nos scores)
# ===========================================================
class UniformSampler(Sampler):
    def select(self, scores):
        """
        Seleciona amostras de forma uniforme ao longo da distribuição de scores.
        """
        size = len(scores)
        sample_size = int(self.ratio * size)

        minval, maxval = scores.min(), scores.max()
        intervals = np.arange(minval, maxval, (maxval - minval) / sample_size)
        bins = np.digitize(scores, intervals)

        unique_bins = np.unique(bins)
        pq_intra = {i: [] for i in unique_bins}

        for i in unique_bins:
            bin_indices = np.where(bins == i)[0]
            bin_scores = scores[bin_indices]
            pq_intra[i].extend(zip(bin_scores, bin_indices))
            heapify(pq_intra[i])

        selected_indices = []
        while len(selected_indices) < sample_size:
            pq_inter = []
            for i in unique_bins:
                if len(pq_intra[i]) > 0:
                    pq_inter.append(heappop(pq_intra[i]))

            if len(selected_indices) + len(pq_inter) < sample_size:
                selected_indices.extend(list(list(zip(*pq_inter))[1]))
            else:
                heapify(pq_inter)
                while len(selected_indices) < sample_size:
                    _, index = heappop(pq_inter)
                    selected_indices.append(index)

        return np.array(selected_indices)
