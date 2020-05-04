import itertools
import os
import numpy as np

from fairseq.data import (
    Dictionary,
    ConcatDataset,
    data_utils,
    TokenBlockDataset,
    indexed_dataset,
)
from fairseq.data.language_pair_langid_dataset import LanguagePairLangidDataset
from fairseq.data.resampling_dataset import ResamplingDataset
from fairseq.data.sort_dataset import SortDataset
from fairseq.data.masking_dataset import MaskingDataset
# from fairseq.data.masked_seq_dataset import MaskedSeqDataset
from fairseq.data.noising_dataset import DaeDataset


def get_sample_prob(dataset_lens, temp):
    """
    Temperature based sampling
    https://arxiv.org/abs/1907.05019
    
    p_l \\prop (D_l / \\sum D_k) ^ 1/T
    """
    prob = dataset_lens / dataset_lens.sum()
    smoothed_prob = prob ** (1./temp)
    smoothed_prob = smoothed_prob / smoothed_prob.sum()
    return smoothed_prob


def get_size_ratio(sample_probs, dataset_lengths, language_upsample_max=False):
    if language_upsample_max:
        max_id = dataset_lengths.argmax()
        max_size, max_probs = dataset_lengths[max_id], sample_probs[max_id]
        size_ratios = sample_probs / max_probs * max_size / dataset_lengths
    else:
        size_ratios = (sample_probs * dataset_lengths.sum()) / dataset_lengths
    return size_ratios


def _mask_index(dic: Dictionary, allow_use_unk=False):
    idx = dic.index("<mask>")
    if not allow_use_unk:
        assert idx != dic.unk_index, \
            'cannot find special token <mask>'
    return idx


class LangpairDatasetLoader(object):
    """
    language pair dataset loader, supports:
        * multilingual bitext dataset
        * temperature based data resampling
        * + BT dataset
    """

    def __init__(
        self, data_path, split, lang_pairs, src_dict, tgt_dict,
        combine, dataset_impl, 
        upsample_primary=False,
        left_pad_source=True, 
        left_pad_target=False, 
        max_source_positions=1024, 
        max_target_positions=1024,
        encoder_langtok='tgt', 
        decoder_langtok=False, 
        seed=1, 
        epoch=0, 
        resample=False,
        language_upsample_max=False, 
        language_sample_temperature=1.0, 
        bt_data_path=None, 
        is_train=True,
        downsample_bt=False,
    ):
        self.data_path = data_path
        self.split = split
        self.lang_pairs = lang_pairs
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.combine = combine
        self.dataset_impl = dataset_impl
        self.upsample_primary = upsample_primary
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.encoder_langtok = encoder_langtok
        self.decoder_langtok = decoder_langtok
        self.seed = seed
        self.epoch = epoch
        self.resample = resample
        self.language_upsample_max = language_upsample_max
        self.language_sample_temperature = language_sample_temperature
        self.bt_data_path = bt_data_path
        self.is_train = is_train
        self.downsample_bt = downsample_bt

    def load_all_langpair_dataset(self):
        # load bt corpus (skip dev)
        bt_dataset = self.load_langpair_langid_dataset(
            data_path=self.bt_data_path,
            lang_pairs=self.lang_pairs,
            is_bt=True
        ) if (self.bt_data_path is not None and self.is_train) else None
        bt_length = len(bt_dataset) if bt_dataset is not None else None

        # load bitext corpus (with resampling)
        if self.resample and self.is_train:
            resampled_lang_pair_datasets, dataset_lengths = self.load_langpair_resample_dataset(
                data_path=self.data_path,
                lang_pairs=self.lang_pairs,
                is_bt=False
            )
            # bitext + bt
            if bt_dataset is not None:
                bt_dataset = self.load_bt_dataset(bt_dataset, bitext_lengths=sum(dataset_lengths))
                resampled_lang_pair_datasets.append(bt_dataset)
            dataset = ConcatDataset(resampled_lang_pair_datasets)

            # shuffle dataset
            with data_utils.numpy_seed(self.seed + self.epoch):
                shuffle = np.random.permutation(len(dataset))
            # sort primarily by source length, then by shuffle seeds
            return SortDataset(dataset, sort_order=[shuffle, dataset.sizes]), dataset_lengths, bt_length

        # load bitext corpus (no resampling)
        bitext_dataset = self.load_langpair_langid_dataset(
            data_path=self.data_path,
            lang_pairs=self.lang_pairs,
            is_bt=False
        )
        # bitext + bt
        if bt_dataset is not None:
            bt_dataset = self.load_bt_dataset(bt_dataset, bitext_lengths=len(bitext_dataset))
            dataset = ConcatDataset([bitext_dataset, bt_dataset])
            with data_utils.numpy_seed(self.seed + self.epoch):
                shuffle = np.random.permutation(len(dataset))
            return SortDataset(dataset, sort_order=[shuffle, dataset.sizes]), None, bt_length
        # bitext only
        return bitext_dataset, None, bt_length

    def load_bt_dataset(self, bt_dataset, bitext_lengths=None):
        # downsample to match #bitext
        if self.downsample_bt:
            size_ratio = min(bitext_lengths / float(len(bt_dataset)), 1.0)
            print("| [resample]  downsampling BT with ratio: {}".format(size_ratio))
            bt_dataset = ResamplingDataset(
                dataset=bt_dataset,
                replace=False,
                size_ratio=size_ratio,
                seed=self.seed,
                epoch=self.epoch,
            )
        return bt_dataset

    def load_langpair_langid_dataset(self, data_path, lang_pairs, is_bt=False):
        src_datasets, tgt_datasets = [], []
        src_langs, tgt_langs = [], []

        for lang_pair in lang_pairs:
            src, tgt = lang_pair.split("-")
            src_datasets, tgt_datasets, src_langs, tgt_langs = self._load_dataset_from_file(
                src, tgt, data_path, 
                src_datasets, tgt_datasets, src_langs, tgt_langs, is_bt=is_bt)
            
        assert len(src_datasets) == len(tgt_datasets)
        assert len(src_langs) == len(tgt_langs)

        if len(src_datasets) == 1:
            src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        else:
            src_dataset = ConcatDataset(src_datasets)
            tgt_dataset = ConcatDataset(tgt_datasets)

        return LanguagePairLangidDataset(
            src_dataset, src_dataset.sizes, self.src_dict, src_langs,
            tgt_dataset, tgt_dataset.sizes, self.tgt_dict, tgt_langs,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            shuffle=True,
            max_source_positions=self.max_source_positions,
            max_target_positions=self.max_target_positions,
            encoder_langtok=self.encoder_langtok,
            decoder_langtok=self.decoder_langtok
        )

    def load_langpair_resample_dataset(self, data_path, lang_pairs, is_bt=False):
        lang_pair_datasets = []
        
        # load langpair dataset
        for lang_pair in lang_pairs:
            lang_pair_datasets.append(self.load_langpair_langid_dataset(
                data_path, lang_pairs=[lang_pair], is_bt=is_bt))

        # resampling
        dataset_lengths = np.array([len(d) for d in lang_pair_datasets], dtype=float)
        print("| loaded {} language pairs".format(len(dataset_lengths)))
        print("| [resample]  epoch {:03d}, sampling temperature: T = {}".format(
            self.epoch, self.language_sample_temperature))
        sample_probs = get_sample_prob(dataset_lengths, temp=self.language_sample_temperature)
        print("| [resample]  epoch {:03d}, sampling probability by language: {}".format(
            self.epoch, ", ".join(["{}: {:0.4f}".format(_lang, sample_probs[_i])
            for _i, _lang in enumerate(lang_pairs)])))
        size_ratios = get_size_ratio(sample_probs, dataset_lengths, self.language_upsample_max)
        print("| [resample]  epoch {:03d}, up/down sampling ratio by language: {}".format(
            self.epoch, ", ".join(["{}: {:0.2f}".format(_lang, size_ratios[_i])
            for _i, _lang in enumerate(lang_pairs)])))
        
        resampled_lang_pair_datasets = [
            ResamplingDataset(
                lang_pair_datasets[i],
                size_ratio=size_ratios[i],
                seed=self.seed,
                epoch=self.epoch,
                replace=size_ratios[i] > 1.0
            )
            for i, d in enumerate(lang_pair_datasets)
        ]
        return resampled_lang_pair_datasets, dataset_lengths

    def _split_exists(self, split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=self.dataset_impl)

    def _load_dataset_from_file(
        self, src, tgt, data_path, 
        src_datasets, tgt_datasets, src_langs, tgt_langs, is_bt=False
    ):
        for k in itertools.count():
            split_k = self.split + (str(k) if k > 0 else '')

            # infer langcode
            if self._split_exists(split_k, src, tgt, src, data_path):
                prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
            elif self._split_exists(split_k, tgt, src, src, data_path) and not is_bt:
                prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
            else:
                if k > 0:
                    break
                elif is_bt:
                    print("| [warning] skipped, BT {}-{} dataset not found".format(src, tgt))
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(self.split, data_path))

            src_datasets.append(indexed_dataset.make_dataset(
                prefix + src, impl=self.dataset_impl, fix_lua_indexing=True, dictionary=self.src_dict))
            tgt_datasets.append(indexed_dataset.make_dataset(
                prefix + tgt, impl=self.dataset_impl, fix_lua_indexing=True, dictionary=self.tgt_dict))

            print('| {} {} {}-{} {} examples'.format(data_path, split_k, src, tgt, len(src_datasets[-1])))
            src_langs += [src] * len(src_datasets[-1])
            tgt_langs += [tgt] * len(src_datasets[-1])

            if not self.combine:
                break
        return src_datasets, tgt_datasets, src_langs, tgt_langs


class MonoDatasetLoader(object):
    """
    monolingual dataset loader, supports:
        * for MLM
        * for DAE
    """

    def __init__(
        self, data_path, split, langs, src_dict,
        combine, dataset_impl, 
        left_pad_source=True, 
        left_pad_target=False,
        max_source_positions=1024, 
        max_target_positions=1024,
        seed=1, 
        epoch=0, 
        static_noising=False,
        token_range=None,
        is_train=True,
        append_langid_encoder=True,
        append_langid_decoder=False,
        max_word_shuffle_distance=3, 
        word_dropout_prob=0.1, 
        word_blanking_prob=0.2,
        text_infilling_ratio=0.,
        text_infilling_lambda=3.5,
        blank_mask_token=False, 
        bpe_cont_marker="sentencepiece",
        masking_ratio=0.15,
        masking_prob=0.8,
        random_token_prob=0.1,
        word_mask=False,
        span_mask=False,
        span_len_lambda=3.5,
        max_dataset_length=-1,
    ):
        self.data_path = data_path
        self.split = split
        self.langs = langs
        self.src_dict = src_dict
        self.combine = combine
        self.dataset_impl = dataset_impl
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.seed = seed
        self.epoch = epoch
        self.static_noising = static_noising
        self.is_train = is_train
        self.token_range = token_range
        self.max_dataset_length = max_dataset_length

        # mlm settings
        self.masking_ratio = masking_ratio
        self.masking_prob = masking_prob
        self.random_token_prob = random_token_prob
        self.word_mask = word_mask
        self.span_mask = span_mask
        self.span_len_lambda = span_len_lambda

        # dae settings
        self.max_word_shuffle_distance = max_word_shuffle_distance
        self.word_dropout_prob = word_dropout_prob
        self.word_blanking_prob = word_blanking_prob
        self.text_infilling_ratio = text_infilling_ratio
        self.text_infilling_lambda = text_infilling_lambda
        self.blank_mask_token = blank_mask_token
        self.bpe_cont_marker = bpe_cont_marker
        self.append_langid_encoder = append_langid_encoder
        self.append_langid_decoder = append_langid_decoder
    

    def load_mlm_dataset(self, data_path, langs):
        datasets, src_langs = [], []

        for lang in langs:
            datasets, src_langs = self._load_dataset_from_file(
                lang, data_path, datasets, src_langs, task="mlm")

        if len(datasets) == 1:
            datasets = datasets[0]
        else:
            datasets = ConcatDataset(datasets)
        
        mlm_dataset = MaskingDataset(
            datasets, datasets.sizes, self.src_dict,
            mask_idx=_mask_index(self.src_dict),
            seed=self.seed,
            shuffle=True,
            masking_ratio=self.masking_ratio,
            masking_prob=self.masking_prob,
            random_token_prob=self.random_token_prob,
            bpe_cont_marker=self.bpe_cont_marker if self.word_mask else None,
            span_mask=self.span_mask,
            span_len_lambda=self.span_len_lambda,
            static_noising=self.static_noising,
            token_range=self.token_range
        )
        if self.max_dataset_length > 0:
            size_ratio = min(float(self.max_dataset_length) / len(mlm_dataset), 1.)
            print("|  [resample] downsample MLM-mono data with ratio: {}".format(size_ratio))
            mlm_dataset = ResamplingDataset(
                mlm_dataset,
                size_ratio=size_ratio,
                seed=self.seed,
                epoch=self.epoch,
                replace=False
            )
        return mlm_dataset

    """
    # deprecated
    def load_mlm_dataset(self, data_path, langs):
        datasets, src_langs = [], []

        for lang in langs:
            datasets, src_langs = self._load_dataset_from_file(
                lang, data_path, datasets, src_langs, task="mlm")
        
        datasets = [
            TokenBlockDataset(
                ds, ds.sizes, self.tokens_per_sample,
                pad=self.src_dict.pad(),
                eos=self.src_dict.eos(),
                break_mode='eos'
            ) 
            for ds in datasets
        ]

        if len(datasets) == 1:
            datasets = datasets[0]
        else:
            datasets = ConcatDataset(datasets)
        
        return MaskedSeqDataset(
            datasets, datasets.sizes, self.src_dict,
            pad_idx=self.src_dict.pad(),
            mask_idx=_mask_index(self.src_dict),
            sep_token_idx=self.src_dict.eos(),
            shuffle=True,
            has_pairs=False,
            masking_ratio=self.masking_ratio,
            masking_prob=self.masking_prob,
            random_token_prob=self.random_token_prob,
            static_noising=self.static_noising
        )
    """

    def load_dae_dataset(self, data_path, langs):
        datasets, src_langs = [], []

        for lang in langs:
            datasets, src_langs = self._load_dataset_from_file(
                lang, data_path, datasets, src_langs, task="dae")

        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = ConcatDataset(datasets)
        
        dae_dataset = DaeDataset(
            dataset, dataset.sizes, self.src_dict,
            src_langs=src_langs,
            mask_idx=_mask_index(self.src_dict) if self.blank_mask_token else self.src_dict.unk(),
            max_word_shuffle_distance=self.max_word_shuffle_distance,
            word_dropout_prob=self.word_dropout_prob,
            word_blanking_prob=self.word_blanking_prob,
            text_infilling_ratio=self.text_infilling_ratio,
            text_infilling_lambda=self.text_infilling_lambda,
            bpe_cont_marker=self.bpe_cont_marker,
            append_langid_encoder=self.append_langid_encoder,
            append_langid_decoder=self.append_langid_decoder,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            max_source_positions=self.max_source_positions,
            max_target_positions=self.max_target_positions,
            shuffle=True,
            static_noising=self.static_noising
        )

        if self.max_dataset_length > 0:
            size_ratio = min(float(self.max_dataset_length) / len(dae_dataset), 1.)
            print("|  [resample] downsample DAE-mono data with ratio: {}".format(size_ratio))
            dae_dataset = ResamplingDataset(
                dae_dataset,
                size_ratio=size_ratio,
                seed=self.seed,
                epoch=self.epoch,
                replace=False
            )
        return dae_dataset

    def _load_dataset_from_file(self, src, data_path, src_datasets, src_langs, task="dae"):
        for k in itertools.count():
            split_k = self.split + (str(k) if k > 0 else '')
            path = os.path.join(data_path, '{}.{}'.format(split_k, src))
            ds = indexed_dataset.make_dataset(
                path, 
                impl=self.dataset_impl,
                fix_lua_indexing=True, 
                dictionary=self.src_dict
            )
            if ds is None:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(self.split, path))

            src_datasets.append(ds)
            src_langs += [src] * len(src_datasets[-1])

            print('| {path} {split} {task}-{lang} {num} examples'.format(
                path=data_path, split=split_k, task=task.upper(), lang=src, num=len(src_datasets[-1])))

            if not self.combine:
                break
        return src_datasets, src_langs

    def set_mlm_hparams(
        self, src_dict, token_range,
        masking_ratio, masking_prob, random_token_prob, word_mask,
        span_mask, span_len_lambda,
        bpe_cont_marker
    ):
        self.src_dict = src_dict
        self.token_range = token_range
        self.masking_ratio = masking_ratio
        self.masking_prob = masking_prob
        self.random_token_prob = random_token_prob
        self.word_mask = word_mask
        self.span_mask = span_mask
        self.span_len_lambda = span_len_lambda
        self.bpe_cont_marker = bpe_cont_marker
    
    def set_dae_hparams(
        self,
        src_dict,
        token_range,
        max_word_shuffle_distance, 
        word_dropout_prob, 
        word_blanking_prob, 
        text_infilling_ratio,
        text_infilling_lambda,
        blank_mask_token, 
        bpe_cont_marker,
        append_langid_encoder, 
        append_langid_decoder
    ):
        self.src_dict = src_dict
        self.token_range = token_range
        self.max_word_shuffle_distance = max_word_shuffle_distance
        self.word_dropout_prob = word_dropout_prob
        self.word_blanking_prob = word_blanking_prob
        self.text_infilling_ratio = text_infilling_ratio
        self.text_infilling_lambda = text_infilling_lambda
        self.blank_mask_token = blank_mask_token
        self.bpe_cont_marker = bpe_cont_marker
        self.append_langid_encoder = append_langid_encoder
        self.append_langid_decoder = append_langid_decoder