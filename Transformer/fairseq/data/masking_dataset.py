import math

import numpy as np
import torch

from typing import Dict, List, Tuple

from . import FairseqDataset, data_utils

from fairseq.data import Dictionary
from fairseq.data.masking import RandomWordMasking, RandomTokenMasking, SpanTokenMasking


class MaskingDataset(FairseqDataset):
    """
    masked language model
      * random token-level masking (bert-style)
      * random word-level masking  (bert-style)
      * span token-level masking   (span bert) 
      
    support masking ratio update at beginning of epoch
    """
    def __init__(
        self, src_dataset, src_sizes, src_dict, mask_idx,
        seed=1,
        shuffle=True,
        masking_ratio=0.15,
        masking_prob=0.8,
        random_token_prob=0.1,
        bpe_cont_marker=None,
        span_mask=False,
        span_len_lambda=3,
        static_noising=False,
        token_range=None,
    ):
        self.src_dataset = src_dataset
        self.src_sizes = np.array(src_sizes)
        self.src_dict = src_dict
        self.pad_idx = self.src_dict.pad()
        self.eos_idx = self.src_dict.pad()
        self.mask_idx = mask_idx
        self.seed = seed
        self.shuffle = shuffle

        self.masking_ratio = masking_ratio
        self.masking_prob = masking_prob
        self.random_token_prob = random_token_prob
        self.bpe_cont_marker = bpe_cont_marker
        self.span_mask = span_mask
        self.span_len_lambda = span_len_lambda
        self.static_noising = static_noising
        self.token_range = token_range

        if span_mask:
            self.token_mask = SpanTokenMasking(
                dictionary=src_dict,
                mask_idx=self.mask_idx,
                masking_ratio=masking_ratio,
                span_len_lambda=span_len_lambda,
                masking_prob=masking_prob,
                random_token_prob=random_token_prob,
                token_range=token_range,
            )
        else:
            self.token_mask = RandomWordMasking(
                dictionary=src_dict,
                mask_idx=mask_idx,
                bpe_cont_marker=bpe_cont_marker,
                token_range=token_range,
                masking_ratio=masking_ratio,
                masking_prob=masking_prob,
                random_token_prob=random_token_prob
            )

    def __getitem__(self, index):
        src_item = self.src_dataset[index]

        if self.static_noising:
            with data_utils.numpy_seed(self.seed + index):
                masked_x, target, _ = self.token_mask.masking(
                    x=src_item, 
                    masking_ratio=self.masking_ratio,
                    span_len_lambda=self.span_len_lambda
                )
        else:
            masked_x, target, _ = self.token_mask.masking(
                x=src_item,
                masking_ratio=self.masking_ratio,
                span_len_lambda=self.span_len_lambda
            )

        return {
            'id': index,
            'source': torch.LongTensor(masked_x),
            'target': torch.LongTensor(target),
        }
    
    def __len__(self):
        return len(self.src_dataset)
    
    def collater(self, samples):
        """
        Merge a list of samples to form a mini-batch.
        """
        if len(samples) == 0:
            return {}

        def merge(key):
            return data_utils.collate_tokens(
                [s[key] for s in samples], 
                self.pad_idx, self.eos_idx, left_pad=False
            )
        
        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens = merge('source')
        # sort by descending source length
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)

        target = merge('target')
        target = target.index_select(0, sort_order)

        return {
            "id": id,
            "nsentences": len(samples),
            "ntokens": sum(len(s["source"]) for s in samples),
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
            },
            "target": target,
        }

    def num_tokens(self, index):
        return self.src_sizes[index]
    
    def size(self, index):
        return self.src_sizes[index]

    @property
    def sizes(self):
        return self.src_sizes

    def ordered_indices(self):
        """
        Return an ordered list of indices. Batches will be constructed based
        on this order.
        """
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.src_sizes)
        return np.lexsort(order)
    
    @property
    def supports_prefetch(self):
        return getattr(self.src_dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        self.src_dataset.prefetch(indices)
    
    def set_epoch(self, epoch, **kwargs):
        super().set_epoch(epoch, **kwargs)

        if hasattr(self.src_dataset, 'set_epoch'):
            self.src_dataset.set_epoch(epoch, **kwargs)
        
        # reset masking hparams
        if kwargs.get('masking_ratio') is not None:
            self.masking_ratio = kwargs['masking_ratio']
        if kwargs.get('span_len_lambda') is not None:
            self.span_len_lambda = kwargs['span_len_lambda']