# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from . import FairseqDataset
from torch.utils.data.dataloader import default_collate


class SortDataset(FairseqDataset):

    def __init__(self, dataset, sort_order):
        super().__init__()
        self.dataset = dataset
        if not isinstance(sort_order, (list, tuple)):
            sort_order = [sort_order]
        self.sort_order = sort_order

        assert all(len(so) == len(dataset) for so in sort_order)

    def ordered_indices(self):
        return np.lexsort(self.sort_order)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if hasattr(self.dataset, 'collater'):
            return self.dataset.collater(samples)
        else:
            return default_collate(samples)
    
    @property
    def sizes(self):
        return self.dataset.sizes

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)
    
    @property
    def supports_prefetch(self):
        return getattr(self.dataset, 'supports_prefetch', False)

    def prefetch(self, indices):
        self.dataset.prefetch(indices)

    def set_epoch(self, epoch, **kwargs):
        super().set_epoch(epoch, **kwargs)
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch, **kwargs)

    def set_sort_order(self, sort_order):
        self.sort_order = sort_order
        assert all(len(so) == len(self.dataset) for so in sort_order)