# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from . import data_utils, FairseqDataset
from .noising import WordDropout, WordNoising, WordShuffle
from .masking import TextInfilling, TokenTextInfilling
from .language_pair_dataset import collate


def _lang_token(lang: str):
    return '__{}__'.format(lang)


def _lang_token_index(dic, lang):
    """Return language token index."""
    idx = dic.index(_lang_token(lang))
    assert idx != dic.unk_index, \
        'cannot find language token for lang {}'.format(lang)
    return idx


class DaeDataset(FairseqDataset):
    """
    A wrapper around torch.utils.data.Dataset for noising data for DAE.
    support multilingual setting (append langid to end of source or beginning of target)
    """

    def __init__(
        self, src_dataset, src_sizes, src_dict,
        src_langs=None, # to support multilingual
        mask_idx=None,  # use <mask> or <unk>
        max_word_shuffle_distance=3,
        word_dropout_prob=0.1,
        word_blanking_prob=0.1,
        text_infilling_ratio=0.2,
        text_infilling_lambda=3,
        bpe_cont_marker=None,
        bpe_end_marker=None,
        append_langid_encoder=False,
        append_langid_decoder=False,
        left_pad_source=True, 
        left_pad_target=False,
        max_source_positions=1024, 
        max_target_positions=1024,
        seed=1,
        shuffle=True, 
        input_feeding=True, 
        remove_eos_from_source=False, 
        append_eos_to_target=False,
        static_noising=False,
    ):
        self.src_dataset = src_dataset
        self.src_sizes = np.array(src_sizes)
        self.src_dict = src_dict
        self.src_langs = np.array(src_langs) if src_langs is not None else None
        self.append_langid_encoder = append_langid_encoder
        self.append_langid_decoder = append_langid_decoder
        self.mask_idx = mask_idx if mask_idx is not None else src_dict.unk()
        self.static_noising = static_noising

        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.seed = seed
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target

        self.max_word_shuffle_distance = max_word_shuffle_distance
        self.word_dropout_prob = word_dropout_prob
        self.word_blanking_prob = word_blanking_prob
        self.text_infilling_ratio = text_infilling_ratio
        self.text_infilling_lambda = text_infilling_lambda

        self.word_dropout = WordDropout(
            dictionary=src_dict,
            bpe_cont_marker=bpe_cont_marker,
            bpe_end_marker=bpe_end_marker,
        )
        self.word_shuffle = WordShuffle(
            dictionary=src_dict,
            default_max_shuffle_distance=self.max_word_shuffle_distance,
            bpe_cont_marker=bpe_cont_marker,
            bpe_end_marker=bpe_end_marker,
        )
        self.text_infilling = TextInfilling(
            dictionary=src_dict,
            mask_idx=self.mask_idx,
            masking_ratio=text_infilling_ratio,
            span_len_lambda=text_infilling_lambda,
            bpe_cont_marker=bpe_cont_marker,
            bpe_end_marker=bpe_end_marker
        )
    
    def __getitem__(self, index):
        src_item = self.src_dataset[index]
        src_lengths = torch.LongTensor([len(src_item)])
        src_lang = self.src_langs[index] if self.src_langs is not None else None
        lang_id = _lang_token_index(self.src_dict, src_lang)

        # add noise
        if self.static_noising:
            with data_utils.numpy_seed(self.seed + index):
                noisy_src_item, noisy_src_lengths = self.noising(src_item, src_lengths)
        else:
            noisy_src_item, noisy_src_lengths = self.noising(src_item, src_lengths)

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if src_item[-1] == eos:
                src_item = src_item[:-1]
        
        if self.append_langid_encoder:
            noisy_src_item = torch.cat([noisy_src_item, torch.LongTensor([lang_id])])

        if self.append_langid_decoder:
            src_item = torch.cat([torch.LongTensor(lang_id), src_item])

        return {
            'id': index,
            'source': noisy_src_item,  # noising seq
            'target': src_item,        # original seq       
        }
    
    def noising(self, x, lengths):
        # Word Shuffle
        noisy_src_tokens, noisy_src_lengths = self.word_shuffle.noising(
            x=torch.t(x.unsqueeze(0)),
            lengths=lengths,
            max_shuffle_distance=self.max_word_shuffle_distance,
        )

        # Text Infilling
        noisy_src_tokens, _, noisy_src_lengths = self.text_infilling.masking(
            x=noisy_src_tokens.squeeze(),
            lengths=noisy_src_lengths,
            masking_ratio=self.text_infilling_ratio,
            span_len_lambda=self.text_infilling_lambda
        )
        noisy_src_tokens = torch.t(torch.LongTensor(noisy_src_tokens).unsqueeze(0))
        noisy_src_lengths = torch.LongTensor(noisy_src_lengths)

        # Word Dropout
        noisy_src_tokens, noisy_src_lengths = self.word_dropout.noising(
            x=noisy_src_tokens,
            lengths=noisy_src_lengths,
            dropout_prob=self.word_dropout_prob,
        )

        # Word Blanking (equiv to masking, no replacing yet)
        noisy_src_tokens, noisy_src_lengths = self.word_dropout.noising(
            x=noisy_src_tokens,
            lengths=noisy_src_lengths,
            dropout_prob=self.word_blanking_prob,
            blank_idx=self.mask_idx,
        )

        noisy_src_tokens = noisy_src_tokens.squeeze()
        return noisy_src_tokens, noisy_src_lengths

    def collater(self, samples):
        return collate(
            samples, 
            pad_idx=self.src_dict.pad(), 
            eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, 
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def __len__(self):
        return len(self.src_dataset)
    
    def num_tokens(self, index):
        return self.src_sizes[index]
    
    def size(self, index):
        return self.src_sizes[index]
    
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
        
        # reset noising hparams
        if kwargs.get('word_dropout_prob') is not None:
            self.word_dropout_prob = kwargs['word_dropout_prob']
        if kwargs.get('word_blanking_prob') is not None:
            self.word_dropout_prob = kwargs['word_blanking_prob']
        if kwargs.get('max_word_shuffle_distance') is not None:
            self.max_word_shuffle_distance = kwargs['max_word_shuffle_distance']
        if kwargs.get('text_infilling_ratio') is not None:
            self.text_infilling_ratio = kwargs['text_infilling_ratio']
        if kwargs.get('text_infilling_lambda') is not None:
            self.text_infilling_lambda = kwargs['text_infilling_lambda']