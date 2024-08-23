from typing import List, Tuple, Dict, FrozenSet, Set, Union
import random
import faiss
import torch
import numpy as np
import os
import logging

from collections import Counter, defaultdict
LOGGER = logging.getLogger(__name__)


class ClusterData:
    number_iterations = 25
    kmeans_num_tries = 2

    cluster_size_headroom = 0.2
    docs_per_author = 2
    kmeans_centroid_multiplier = 2.5
    
    
    def using_ddp(self):
        return 'LOCAL_RANK' in os.environ


    def get_device(self) -> Union[int, str]:
        if 'USE_CPU' in os.environ:
            LOGGER.info('using cpu')
            return 'cpu'
        elif torch.cuda.is_available():
            if self.using_ddp():
                local_rank = os.environ['LOCAL_RANK']
            # Note: a single ordinal like this is treated as a cuda device
                device = f"cuda:{int(local_rank)}"
            else:
                device = 'cuda'
            LOGGER.info(f'using device: {device}: {torch.cuda.get_device_name()}')
            return device
        else:
            raise RuntimeError('No GPU found. Set USE_CPU in env to use cpu.')
        
    def __init__(self, batch_count: int, shuffle: bool, seed: int, batch_size: int):
        self.batch_size = batch_size
        self.batch_count = batch_count
        self.batch_author_count = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.device = self.get_device()
        self.model_outputs: List[torch.Tensor] = []
        self.target_cluster_size = None
        self.max_cluster_size = None
    
    def clear(self) -> None:
        self.model_outputs = []
        
    def add(self, model_output):
        self.model_outputs.append(model_output)
        
    def empty(self):
        return len(self.model_outputs) == 0
    
    def cluster_random(self, author_id_int):
        random.seed(self.seed)
        self.seed += 1
        l = list(range(author_id_int))
        random.shuffle(l)
        return l
    
    def _get_author_ids_and_inds(self, author_ids: torch.Tensor) -> Tuple[List[int], np.ndarray]:
        author_inds = defaultdict(list)
        for i, a_id in enumerate(author_ids.tolist()):
            author_inds[a_id].append(i)
        author_ids = sorted(author_inds.keys())


        # author_inds = \
        #     np.array(
        #         [author_inds[a_id]
        #          for a_id in author_ids]
        #     )

        LOGGER.info("*** Implementing Sadiri benchmark dataset ***")
        author_inds = \
            np.array(
                [author_inds[a_id]
                for a_id in author_ids]
            )
        LOGGER.info(f"*** The size of author_inds = {author_inds.shape} ***")


        # NB: The preceding command will fail if the number of indices per author varies.
        if author_inds.shape[1] != self.docs_per_author:
            raise ValueError(f"Got an unexpected number of documents.")
        LOGGER.info(f"number of author_ids: {len(author_ids)}")
        return author_ids, author_inds
    
    @staticmethod
    def _select_centroid(distances: List[float], c_ids: List[int], centroid_sizes: Dict[int, int]) -> List[int]:
        size_diff = centroid_sizes[c_ids[0]] - centroid_sizes[c_ids[1]]
        if size_diff > 0:
            best_ind = 0
        elif size_diff < 0:
            best_ind = 1
        elif distances[0] < distances[1]:
            best_ind = 0
        else:
            best_ind = 1
        centroid_sizes[
            c_ids[
                1 if best_ind == 0 else 0
            ]
        ] -= 1
        return [c_ids[best_ind]]

    @staticmethod
    def _get_sizes(centroids: Dict[FrozenSet[str], Set[int]]):
        return sorted([(k, len(v))
                       for k, v in centroids.items()], key=lambda x: (x[1], x[0]))
    
    
    def _get_targets_and_sizes(self, centroids: Dict[FrozenSet[str], Set[int]]):
        sizes = self._get_sizes(centroids)
        over_cnt = 0
        over_total = 0
        under_total = 0
        for k, v in sizes:
            if v >= self.target_cluster_size:
                over_cnt += 1
                over_total += v
            else:
                under_total += v
        if self.batch_count > over_cnt:
            local_target_cluster_size = min(self.target_cluster_size,
                                            int(round(under_total / int(self.batch_count - over_cnt))))
        else:
            local_target_cluster_size = self.target_cluster_size

        target_ind = int(-1 - self.batch_count)
        target_size = sizes[target_ind][1]
        max_add_c = None
        target_add_c = None
        if len(sizes) > 1:
            for k, v in sizes[1 + target_ind:]:
                if target_size + v <= local_target_cluster_size:
                    target_add_c = k
                if target_size + v <= self.max_cluster_size:
                    max_add_c = k
                else:
                    break
        return sizes[target_ind][0], target_add_c if target_add_c is not None else max_add_c, sizes
    
    def _get_centroids(self,
                       author_ids: List[int],
                       init_centroid_ids: np.ndarray):
        LOGGER.info(f"get_centroids batch_count: {self.batch_count} target_cluster_size: {self.target_cluster_size} "
                    f"max_cluster_size: {self.max_cluster_size}")

        centroids = defaultdict(set)
        for a_id, c_id in zip(author_ids, init_centroid_ids):
            centroids[
                frozenset(c_id)
            ].add(a_id)

        # break up over-size clusters
        for k, loc_cnt in self._get_sizes(centroids):
            if loc_cnt > self.max_cluster_size:
                v = list(centroids[k])
                random.shuffle(v)
                i = 0
                while i * self.batch_author_count < len(v):
                    centroids[
                        frozenset(
                            set(k).union([f"split_{i}"])
                        )
                    ] = set(
                        v[(i * self.batch_author_count):((1 + i) * self.batch_author_count)]
                    )
                    i += 1
                del centroids[k]

        while len(centroids) > self.batch_count:
            # Eliminate the smallest cluster.
            smallest_c, add_c, sizes = self._get_targets_and_sizes(centroids)
            if add_c is None:
                raise ValueError(f"Failed to merge smallest cluster.")
            new_c = frozenset(list(smallest_c) + list(add_c))
            new_vals = centroids[smallest_c].union(centroids[add_c])
            del centroids[smallest_c]
            del centroids[add_c]
            centroids[new_c] = new_vals

        if len(centroids) != self.batch_count:
            raise ValueError(f"output batch count does not match target.")

        assigned_ids = Counter()
        for v in centroids.values():
            assigned_ids.update(list(v))

        if max(list(assigned_ids.values())) > 1:
            LOGGER.info(f"Some authors were assigned to multiple clusters...")
            raise ValueError

        missed_ids = [a_id for a_id in author_ids
                      if assigned_ids[a_id] != 1]
        if len(missed_ids) > 0:
            LOGGER.info(f"Missed {len(missed_ids)} author_ids: {author_ids}")
            by_size = self._get_sizes(centroids)
            targ_ind = 0
            while len(missed_ids) > 0:
                add_id = missed_ids.pop()
                while len(centroids[by_size[targ_ind]]) > self.max_cluster_size:
                    targ_ind += 1
                LOGGER.info(f"Add extra id: {add_id} to centroid: {by_size[targ_ind][0]}")
                centroids[
                    by_size[targ_ind][0]
                ].add(add_id)

        LOGGER.info(f"data_size: {self.data_size} author_ids: {len(author_ids)} "
                    f"ratio: {self.data_size / float(len(author_ids))}")

        return centroids
    
    def cluster_hard_negative(self):
        # get model output and author id
        self.seed += 1
        author_ids = []
        reps = []
        start = 0
        for model_output in self.model_outputs:
            reps.append(model_output)
            labels = torch.arange(start, (len(model_output) / 2) + start).long().to(self.device)
            labels = torch.cat([labels, labels], dim=0)
            author_ids.append(labels)
            start += self.batch_size
            
        author_ids = torch.cat(author_ids)
        reps = torch.cat(reps)
        
        reps = reps.to(torch.float32)
        reps = torch.nn.functional.normalize(reps, dim=-1)
        reps = reps.cpu().detach().numpy()
        
        self.data_size = reps.shape[0]
        
        author_ids, author_inds = self._get_author_ids_and_inds(author_ids)
        self.clear()
        input_dimension = reps.shape[-1]

        LOGGER.info(f"batch_size: {self.batch_size} batch_count: {self.batch_count} reps.shape: {reps.shape}")

        num_centroids = int(round(self.batch_count * self.kmeans_centroid_multiplier))
        
        kmeans = \
            faiss.Kmeans(input_dimension,
                         num_centroids,
                         niter=self.number_iterations,
                         verbose=False,
                         nredo=self.kmeans_num_tries,
                         spherical=True,
                         seed=self.seed,
                         gpu=(str(self.device) != 'cpu')
                         )
        author_reps = reps[author_inds.flatten()]
        kmeans.train(author_reps)
        
        
        distances, init_centroid_ids = kmeans.index.search(author_reps, 1)
        LOGGER.info(f"clustering distances mean: {distances.mean():.4f} std: {distances.std():.4f} "
                    f"min: {distances.min():.4f} max: {distances.max():.4f} median: {np.median(distances):.4f}")
        centroid_sizes: Dict[int, int] = Counter([c[0] for c in init_centroid_ids])
        centroid_size_np = np.array([n for c, n in
                                     sorted(list(centroid_sizes.items()))
                                     ])
        LOGGER.info(f"init centroid count: {centroid_size_np.shape[0]} mean: {centroid_size_np.mean():.4f} "
                    f"std: {centroid_size_np.std():.4f} min: {centroid_size_np.min(initial=None):.4f} "
                    f"max: {centroid_size_np.max(initial=None):.4f} median: {np.median(centroid_size_np):.4f}")

        distances = distances.squeeze(-1).reshape(-1, author_inds.shape[-1])
        init_centroid_ids = init_centroid_ids.squeeze(-1).reshape(-1, author_inds.shape[-1])
        LOGGER.info(f"got distances and init_centroid_ids.")
        
        selected_centroids = np.array([self._select_centroid(a_d.tolist(), a_c.tolist(), centroid_sizes)
                                       for a_d, a_c in zip(distances, init_centroid_ids)])
        LOGGER.info(f"got selected_centroids.")

        LOGGER.info(f"batch_count: {self.batch_count} target_cluster_size: {self.batch_author_count} "
                    f"max_cluster_size: {round(self.batch_author_count * (1 + self.cluster_size_headroom))}")
        
        self.target_cluster_size = self.batch_author_count
        self.max_cluster_size = round(self.batch_author_count * (1 + self.cluster_size_headroom))
        centroids = self._get_centroids(author_ids=author_ids, init_centroid_ids=selected_centroids)
        centroids = [list(v) for v in centroids.values()]
        batch_sizes = np.array([len(c) for c in centroids])
        LOGGER.info(f"batch sizes count: {len(centroids)} mean: {batch_sizes.mean()} "
                    f"min: {batch_sizes.min(initial=None)} max: {batch_sizes.max(initial=None)} "
                    f"std: {batch_sizes.std()}")
        if batch_sizes.min(initial=None) < 1:
            raise ValueError

        return centroids
    
    
