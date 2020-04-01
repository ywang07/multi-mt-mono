import torch
import numpy as np
import math

from fairseq.data import data_utils

class MaskingScheme(object):
    def __init__(
        self, 
        dictionary, 
        mask_idx,
        bpe_cont_marker="@@", 
        bpe_end_marker=None,
        token_range=None
    ):
        self.dictionary = dictionary
        self.mask_idx = mask_idx
        self.pad_idx = dictionary.pad()
        self.token_range = token_range if token_range is not None else (self.dictionary.nspecial, len(self.dictionary))

        # word boundary
        self.bpe_start, self.bpe_end = None, None
        if bpe_cont_marker is not None and bpe_cont_marker == "sentencepiece":
            self.bpe_start = np.array([
                self.dictionary[i].startswith('\u2581')
                for i in range(len(self.dictionary))
            ])
        elif bpe_cont_marker is not None:
            self.bpe_end = np.array([
                not self.dictionary[i].endswith(bpe_cont_marker)
                for i in range(len(self.dictionary))
            ])
        elif bpe_end_marker is not None:
            self.bpe_end = np.array([
                self.dictionary[i].endswith(bpe_end_marker)
                for i in range(len(self.dictionary))
            ])

        if self.bpe_start is not None:
            self.get_word_idx = self._get_spm_word_idx
        elif self.bpe_end is not None:
            self.get_word_idx = self._get_bpe_word_idx
        else:
            self.get_word_idx = self._get_token_idx

    def masking(self, x, **kwargs):
        raise NotImplementedError()

    def bert_masking(self, x, word_idx, mask_pos, masking_prob, random_token_prob):
        """
        replacing each mask position to a special token MASK
        """
        masked_x, target = np.copy(x), np.copy(x)
        rand = np.random.rand(max(word_idx)+1)

        for i in range(len(x)):
            widx = word_idx[i]
            if widx in mask_pos:
                # replace with mask if probability is less than masking_prob
                # (Eg: 0.8)
                if rand[widx] < masking_prob:
                    masked_x[i] = self.mask_idx

                # replace with random token if probability is less than
                # masking_prob + random_token_prob (Eg: 0.9)
                elif rand[widx] < (masking_prob + random_token_prob):
                    # sample random token from dictionary
                    masked_x[i] = (
                        np.random.randint(
                            self.token_range[0], self.token_range[1]
                        )
                    )
            else:
                target[i] = self.pad_idx
        return masked_x, target

    def span_masking(self, x, x_len, spans):
        """
        replacing a token span to a special token MASK
        """
        keep = []
        idx = 0
        for start, end in spans:
            keep.append(x[idx:start])
            keep.append(np.array([self.mask_idx]))
            idx = end
        if idx < x_len:
            keep.append(x[idx:])
        else:
            keep = keep[:-1]
        return np.concatenate(keep)

    def _get_spm_word_idx(self, x):
        """
        return: 
            word_idx: [0, 1, 1, 1, 2, 2, 3]
            word_start_idx: {0: 0, 1: 1, 2: 4, 3: 6, 4: 7}
        """
        bpe_start = self.bpe_start[x]

        if len(x) == 1:
            return np.array([0]), {0: 0, 1: 1}

        start_pos = np.argwhere(bpe_start).squeeze()
        word_start_idx = {i: start_pos[i] for i in range(len(start_pos))}
        word_start_idx[len(start_pos)] = len(x)

        word_idx = bpe_start.cumsum(0)
        word_idx -= 1 if min(word_idx) > 0 else 0 
        return word_idx, word_start_idx
    
    def _get_bpe_word_idx(self, x):
        """
        return: 
            word_idx: [0, 1, 1, 1, 2, 2, 3]
            word_start_idx: {0: 0, 1: 1, 2: 4, 3: 6, 4: 7}
        """
        bpe_end = self.bpe_end[x]

        if len(x) == 1:
            return np.array([0]), {0: 0}
        
        end_pos = np.argwhere(bpe_end).squeeze()
        word_start_idx = {i+1: end_pos[i]+1 for i in range(len(end_pos)-1)}
        word_start_idx[0] = 0
        word_start_idx[len(end_pos)] = len(x)

        # do a reduce front sum to generate word ids
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0) - word_idx
        return word_idx, word_start_idx

    def _get_token_idx(self, x):
        return np.array(range(len(x))), None


class RandomTokenMasking(MaskingScheme):
    def __init__(
        self, 
        dictionary, 
        bpe_cont_marker=None, 
        bpe_end_marker=None,
        token_range=None,
        mask_idx=-1,
        masking_ratio=0.15,
        masking_prob=0.8,
        random_token_prob=0.1,
    ):
        super().__init__(dictionary, mask_idx, bpe_cont_marker, bpe_end_marker, token_range)
        self.masking_ratio = masking_ratio
        self.masking_prob = masking_prob
        self.random_token_prob = random_token_prob

    def masking(self, x, **kwargs):
        masking_ratio = kwargs.get('masking_ratio', self.masking_ratio)

        x_len = len(x)
        x_len_no_eos = x_len - 1 if x[-1] == self.dictionary.eos() else 0
        mask_num = math.ceil(x_len_no_eos * masking_ratio)
        mask_pos = np.random.choice(x_len_no_eos, mask_num, replace=False)
        
        masked_x, target = self.bert_masking(
            x=x,
            word_idx=np.array(range(x_len)),
            mask_pos=mask_pos,
            masking_prob=self.masking_prob,
            random_token_prob=self.random_token_prob,
        )
        return masked_x, target, x_len


class RandomWordMasking(MaskingScheme):
    def __init__(
        self, 
        dictionary, 
        bpe_cont_marker="@@", 
        bpe_end_marker=None,
        token_range=None,
        mask_idx=-1,
        masking_ratio=0.35,
        masking_prob=0.8, 
        random_token_prob=0.1
    ):
        super().__init__(dictionary, mask_idx, bpe_cont_marker, bpe_end_marker, token_range)
        self.masking_ratio = masking_ratio
        self.masking_prob = masking_prob
        self.random_token_prob = random_token_prob
        

    def masking(self, x, **kwargs):
        masking_ratio = kwargs.get('masking_ratio', self.masking_ratio)
        x_len = len(x)

        # mask entire word
        word_idx, _ = self.get_word_idx(x)
        num_words = max(word_idx) + 1
        num_words_no_eos = num_words - 1 if x[-1] == self.dictionary.eos() else 0

        mask_num = math.ceil(num_words_no_eos * masking_ratio)
        mask_pos = np.random.choice(num_words_no_eos, mask_num, replace=False)
        
        masked_x, target = self.bert_masking(
            x=x,
            word_idx=word_idx,
            mask_pos=mask_pos,
            masking_prob=self.masking_prob,
            random_token_prob=self.random_token_prob,
        )
        return masked_x, target, x_len


class SpanTokenMasking(MaskingScheme):
    """
    mask a span of tokens instead of single tokens
    span BERT: https://arxiv.org/abs/1907.10529
    """
    def __init__(
        self, dictionary, mask_idx,
        bpe_cont_marker=None, 
        bpe_end_marker=None,
        token_range=None,
        masking_ratio=0.3, 
        span_len_lambda=3,
        masking_prob=0.8, 
        random_token_prob=0.1,
    ):
        super().__init__(dictionary, mask_idx, bpe_cont_marker, bpe_end_marker, token_range)
        self.masking_ratio = masking_ratio
        self.span_len_lambda = span_len_lambda
        self.masking_prob = masking_prob
        self.random_token_prob = random_token_prob

    def masking(self, x, **kwargs):
        masking_ratio = kwargs.get('masking_ratio', self.masking_ratio)
        span_len_lambda = kwargs.get('span_len_lambda', self.span_len_lambda)

        if masking_ratio == 0:
            return x, x, len(x)
        
        assert 0 < masking_ratio < 1
        assert span_len_lambda > 0

        x_len = len(x)
        x_len_no_eos = x_len - 1 if x[-1] == self.dictionary.eos() else 0
        mask_num = math.ceil(x_len_no_eos * masking_ratio)
        mask_pos = set()

        while len(mask_pos) < mask_num:
            span_len = min(np.random.poisson(lam=span_len_lambda), mask_num)
            start = np.random.choice(x_len_no_eos-span_len)
            if start in mask_pos:
                continue
            for i in range(start, start+span_len):
                if len(mask_pos) >= mask_num:
                    break
                mask_pos.add(i)
        
        masked_x, target = self.bert_masking(
            x=x,
            word_idx=np.array(range(x_len)),
            mask_pos=mask_pos,
            masking_prob=self.masking_prob,
            random_token_prob=self.random_token_prob,
        )
        return masked_x, target, x_len


class TokenTextInfilling(MaskingScheme):
    """
    replace a span of tokens into a single MASK token
    
    text infilling task in BART
    https://arxiv.org/pdf/1910.13461.pdf
    """
    def __init__(
        self, dictionary, mask_idx,
        masking_ratio=0.3, 
        span_len_lambda=3,
        bpe_cont_marker=None, 
        bpe_end_marker=None
    ):
        super().__init__(dictionary, mask_idx, bpe_cont_marker, bpe_end_marker)
        self.masking_ratio = masking_ratio
        self.span_len_lambda = span_len_lambda

    def masking(self, x, length, **kwargs):
        masking_ratio = kwargs.get('masking_ratio', self.masking_ratio)
        span_len_lambda = kwargs.get('span_len_lambda', self.span_len_lambda)

        if masking_ratio == 0:
            return x, x, length
        
        assert 0 < masking_ratio < 1
        assert span_len_lambda > 0

        x_len = len(x)
        x_len_no_eos = x_len - 1 if x[-1] == self.dictionary.eos() else 0
        mask_num = math.ceil(x_len_no_eos * masking_ratio)
        mask = set()
        spans = []    # list of start/end token idx (end not included)

        while len(mask) < mask_num:
            span_len = min(np.random.poisson(lam=span_len_lambda), mask_num)
            start = np.random.choice(x_len_no_eos-span_len)
            if start in mask:
                continue
            spans.append([start, start])
            for i in range(start, start+span_len):
                if len(mask) >= mask_num:
                    break
                mask.add(i)
                spans[-1][-1] = i+1
        
        spans = merge_intervals(spans)
        masked_x = self.span_masking(x, x_len, spans)
    
        return masked_x, x, torch.LongTensor([len(masked_x)])


class TextInfilling(MaskingScheme):
    """
    replace a span of words into a single MASK token

    text infilling task in BART
    https://arxiv.org/pdf/1910.13461.pdf
    """
    def __init__(
        self, dictionary, mask_idx,
        masking_ratio=0.3, 
        span_len_lambda=3,
        bpe_cont_marker=None, 
        bpe_end_marker=None
    ):
        super().__init__(dictionary, mask_idx, bpe_cont_marker, bpe_end_marker)
        self.masking_ratio = masking_ratio
        self.span_len_lambda = span_len_lambda

    def masking(self, x, lengths, **kwargs):
        masking_ratio = kwargs.get('masking_ratio', self.masking_ratio)
        span_len_lambda = kwargs.get('span_len_lambda', self.span_len_lambda)

        if masking_ratio == 0:
            return x, x, lengths
        
        assert 0 < masking_ratio < 1
        assert span_len_lambda > 0

        x_len = len(x)

        # word-level masking
        word_idx, word_start_idx = self.get_word_idx(x)
        num_words = max(word_idx) + 1

        num_words_no_eos = num_words - 1 if x[-1] == self.dictionary.eos() else 0
        mask_num = math.ceil(num_words_no_eos * masking_ratio)
        mask = set() 
        spans = []    # list of start/end word idx (end not included)

        while len(mask) < mask_num:
            span_len = min(np.random.poisson(lam=span_len_lambda), mask_num)
            start = np.random.choice(num_words_no_eos-span_len)
            if start in mask:
                continue
            spans.append([start, start])
            for i in range(start, start+span_len):
                if len(mask) >= mask_num:
                    break
                mask.add(i)
                spans[-1][-1] = i+1
        
        if word_start_idx is not None:
            spans = [[word_start_idx[s1], word_start_idx[s2]] for s1, s2 in spans]
        spans = merge_intervals(spans)
        masked_x = self.span_masking(x, x_len, spans)

        return masked_x, x, torch.LongTensor([len(masked_x)])


def merge_intervals(intervals):
    intervals = sorted(intervals, key=lambda x : x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged
