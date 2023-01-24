import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


def simple_random_sampling(
    cur_entity: Tuple[int, int],
    num_neg_sample: int,
    true_triplet: Dict,
    num_entity: int = None,
    go_terms: List = None,
    **kwargs
) -> List:
    """
    A simple strategy of negative sampling.

    Args:
        cur_entity: Fixed entity (id) with relation (id) which used to construct negative triplet samples with another entity 
                    sampled from a specific entity set.
        true_triplet: dict containing postive triplets. Key is (head,relation) tail is set of positive triplets
        num_neg_sample: the number of negative sampling.
        num_entity: the size of sampling set. If the type of fixed entity is `Go`, the parameter could be ignored.
        go_terms: List containing all GO with the same type as the current go tail type. This allows sampling from the same
                    go type.
    """

    is_go = False if go_terms is None else True
    
    negative_sample_list = []
    negative_sample_size = 0

    while negative_sample_size < num_neg_sample:
        if not is_go:
            negative_sample = np.random.randint(num_entity, size=num_neg_sample)
        else:
            negative_sample = np.random.choice(go_terms, size=num_neg_sample, replace=False)

        # mask is True when negative tail is not in the list of true tails given (head,relation)
        mask = np.in1d(
            negative_sample,
            true_triplet[cur_entity],
            assume_unique=True,
            invert=True
        )
        negative_sample = negative_sample[mask]
        negative_sample_list.append(negative_sample)
        negative_sample_size += negative_sample.size
    
    negative_sample = np.concatenate(negative_sample_list)[:num_neg_sample]
    return negative_sample


negative_sampling_strategy = {
    'simple_random': simple_random_sampling,
}