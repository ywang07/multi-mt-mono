# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import OrderedDict
import copy
import itertools
import os

import torch
import numpy as np

from fairseq import options, utils
from fairseq.data import (
    iterators,
    FairseqDataset,
    Dictionary,
    data_utils,
    RoundRobinZipDatasets,
)
from . import FairseqTask, register_task

from fairseq.models.transformer_mlm import TransformerMlmModel 
from fairseq.criterions.mlm_loss import MlmLoss
from fairseq.data.language_pair_langid_dataset import LanguagePairLangidDataset
from fairseq.data.sort_dataset import SortDataset
from fairseq.data.resampling_dataset import ResamplingDataset
from fairseq.data.langpair_dataset_loader import (
    LangpairDatasetLoader,
    MonoDatasetLoader,
    get_size_ratio,
    get_sample_prob
)


def _lang_token(lang: str):
    return '__{}__'.format(lang)


def _lang_token_index(dic: Dictionary, lang: str):
    """Return language token index."""
    idx = dic.index(_lang_token(lang))
    assert idx != dic.unk_index, \
        'cannot find language token for lang {}'.format(lang)
    return idx


def _get_criterion_key(key):
    if "mlm" in key:
        return "mlm"
    return "seq2seq"


@register_task('translation_mtl_multitask_curr')
class TranslationMtlMultitaskCurrTask(FairseqTask):
    """A task for training multiple translation models simultaneously.

    Different from multilingual_translation task that iterate over different languages
    Each batch consists of randomly sampled sentences from different language pairs

    Add multitask objectives
        * Masked LM
        * Denoising Auto Encoder
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', metavar='DIR', help='path to data directory')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs: en-de,en-fr,de-fr')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language (only needed for inference)')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language (only needed for inference)')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left (default: True)')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left (default: False)')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--encoder-langtok', default='tgt', type=str, choices=['src', 'tgt'],
                            metavar='SRCTGT',
                            help='replace beginning-of-sentence in source sentence with source or target '
                                 'language token. (src/tgt) default to be target lan_id')
        parser.add_argument('--decoder-langtok', action='store_true',
                            help='replace beginning-of-sentence in target sentence with target language token')
        
        # BT / mono data
        parser.add_argument('--data-bt', metavar='DIR', default=None, help='path to back translation data directory')
        parser.add_argument('--data-mono', metavar='DIR', default=None, help='path to mono data directory')
        parser.add_argument('--downsample-bt', action='store_true',
                            help="downsample bt to match the length of bitext data")
        parser.add_argument('--downsample-mono', action='store_true',
                            help="downsample mono to match the length of bitext data")
        
        # MLM
        parser.add_argument('--multitask-mlm', action='store_true',
                            help='use MaskedLM objective together with MT cross-entropy')
        parser.add_argument('--lang-mlm', default=None,
                            help='comma-separated list of languages with mono data: en,de,fr (for MLM)')
        parser.add_argument('--mlm-masking-ratio', default=0.15, type=float,
                            help='masking ratio for MaskedLM')
        parser.add_argument('--mlm-masking-prob', default=0.8, type=float,
                            help='masking ratio for MaskedLM')
        parser.add_argument('--mlm-random-token-prob', default=0.1, type=float,
                            help='masking ratio for MaskedLM')
        parser.add_argument('--mlm-word-mask', action='store_true',
                            help='use word-level random masking for MaskedLM')
        parser.add_argument('--mlm-span-mask', action='store_true',
                            help='use span masking for MaskedLM')
        parser.add_argument('--mlm-span-lambda', default=3.5, type=float,
                            help='lambda of poisson distribution for span length sampling')

        # DAE
        parser.add_argument('--multitask-dae', action='store_true',
                            help='use DAE objective together with MT cross-entropy')
        parser.add_argument('--lang-dae', default=None,
                            help='comma-separated list of languages with mono data: en,de,fr (for DAE)')
        parser.add_argument('--dae-max-shuffle-distance', default=3.0, type=float,
                            help='maximum shuffle distance for DAE')
        parser.add_argument('--dae-dropout-prob', default=0., type=float,
                            help='token dropout probability for DAE')
        parser.add_argument('--dae-blanking-prob', default=0., type=float,
                            help='token blanking probability for DAE')
        parser.add_argument('--dae-blanking-with-mask', action='store_true',
                            help='token blanking with <mask> token instead of <unk> for DAE')
        parser.add_argument('--dae-span-masking-ratio', default=0.35, type=float,
                            help='span masking ratio for DAE')
        parser.add_argument('--dae-span-lambda', default=3.5, type=float,
                            help='lambda of poisson distribution for span length sampling')
        parser.add_argument('--bpe-cont-marker', default="sentencepiece", type=str,
                            help='word level (if sentencepiece or bpe) or token level (others) noising')       
        parser.add_argument('--static-noising', action='store_true',
                            help='use same noising for same example in each epoch (both mlm and dae)')       
        
        # data schedule
        parser.add_argument('--language-sample-temperature', default=1.0, type=float, 
                            help='sampling temperature for multi-languages')
        parser.add_argument('--language-upsample-max', action='store_true',
                            help='upsample to make the max-capacity language a full set '
                                 '(default: upsample and downsample to maintain the same total corpus size)')
        parser.add_argument('--language-temperature-scheduler', default='static', type=str, 
                            help='sampling temperature scheduler [static, linear]')
        parser.add_argument('--min-language-sample-temperature', default=1.0, type=float, 
                            help='min (starting) sampling temperature')
        parser.add_argument('--language-sample-warmup-epochs', default=0, type=int, 
                            help='warmup epochs for language sampling scheduler')

        # task schedule
        parser.add_argument('--multitask-scheduler', default='static', type=str, 
                            help='multitask weight scheduler [static, linear]')
        parser.add_argument('--mlm-alpha', default=1, type=float, help='weight for mlm objective')
        parser.add_argument('--dae-alpha', default=1, type=float, help='weight for dae objective')
        parser.add_argument('--mlm-alpha-min', default=1, type=float, help='minimum weight for mlm objective')
        parser.add_argument('--dae-alpha-min', default=1, type=float, help='minimum weight for dae objective')
        parser.add_argument('--mlm-alpha-warmup', default=1, type=float, help='warmup epochs for mlm objective')
        parser.add_argument('--dae-alpha-warmup', default=1, type=float, help='warmup epochs for dae objective')

        # noising ratio schedule
        parser.add_argument('--mlm-masking-ratio-min', default=0.15, type=float,
                            help='minimal (starting) masking ratio for MaskedLM')
        parser.add_argument('--mlm-masking-ratio-warmup-epochs', default=1, type=int,
                            help='warmup epochs for masking ratio scheduler')
        parser.add_argument('--mlm-masking-ratio-scheduler', default='static', type=str, 
                            help='MaskedLM masking ratio scheduler [static, linear]')

        parser.add_argument('--dae-span-masking-ratio-min', default=0., type=float,
                            help='minimal (starting) span masking ratio for DAE')
        parser.add_argument('--dae-span-masking-ratio-warmup-epochs', default=1, type=int,
                            help='warmup epochs for span masking ratio scheduler')
        parser.add_argument('--dae-span-masking-ratio-scheduler', default='static', type=str, 
                            help='DAE span masking ratio scheduler [static, linear]')

    def __init__(self, args, src_dict, tgt_dict, training, src_token_range=None, tgt_token_range=None):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.src_token_range = src_token_range
        self.tgt_token_range = tgt_token_range

        self.lang_pairs = args.lang_pairs
        self.eval_lang_pairs = args.lang_pairs
        self.model_lang_pairs = copy.copy(args.lang_pairs)
        
        self.langs = sorted(list({x for lang_pair in args.lang_pairs for x in lang_pair.split('-')}))
        self.training = training
        self.criterions = {}
        self.langs_mlm = sorted(list(set(args.lang_mlm.split(",")))) if args.lang_mlm is not None else []
        self.langs_dae = sorted(list(set(args.lang_dae.split(",")))) if args.lang_dae is not None else []
        self.langs_mono = sorted(list(set(self.langs_mlm + self.langs_dae)))
        self.multitask_mlm = args.multitask_mlm
        self.multitask_dae = args.multitask_dae
        self.use_bt = False

        self.data_lengths = None
        self.data_bt_lengths = None
        self.dataset_to_epoch_iter = {}
        self.mlm_alpha = float(self.args.mlm_alpha) if self.multitask_mlm else 0.
        self.dae_alpha = float(self.args.dae_alpha) if self.multitask_dae else 0.

        self.update_iterator = (self.args.language_sample_temperature != 1. \
            or self.args.language_temperature_scheduler != "static" \
            or self.args.downsample_bt \
            or self.args.downsample_mono
            )

    @classmethod
    def setup_task(cls, args, **kwargs):
        src_dict, tgt_dict, training, src_token_range, tgt_token_range = cls.prepare(args, **kwargs)
        return cls(args, src_dict, tgt_dict, training, src_token_range, tgt_token_range)

    @classmethod
    def prepare(cls, args, **kargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'
        if args.lang_pairs is None:
            raise ValueError('--lang-pairs is required. List all the language pairs in the training objective.')
        args.lang_pairs = args.lang_pairs.split(',')
        if args.multitask_mlm and args.lang_mlm is None:
            raise ValueError('--lang-mlm is required for mlm objective. List all the language with monolingual data.')
        if args.multitask_dae and args.lang_dae is None:
            raise ValueError('--lang-dae is required for dae objective. List all the language with monolingual data.')

        sorted_langs = sorted(list({x for lang_pair in args.lang_pairs for x in lang_pair.split('-')}))
        if args.lang_mlm is not None:
            sorted_langs = sorted(list(set(sorted_langs + args.lang_mlm.split(","))))
        if args.lang_dae is not None:
            sorted_langs = sorted(list(set(sorted_langs + args.lang_dae.split(","))))
        
        if args.source_lang is not None or args.target_lang is not None:
            training = False
            args.lang_pairs = ['{}-{}'.format(args.source_lang, args.target_lang)]
        else:
            training = True
            args.source_lang, args.target_lang = "src", "tgt"

        # load dictionaries
        paths = args.data.split(':')
        assert len(paths) > 0
        src_dict = Dictionary.load(os.path.join(paths[0], 'dict.src.txt'))
        tgt_dict = Dictionary.load(os.path.join(paths[0], 'dict.tgt.txt'))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        src_token_range = ((src_dict.nspecial, len(src_dict)))
        tgt_token_range = ((tgt_dict.nspecial, len(tgt_dict)))

        # add mask token
        _mask_idx = src_dict.add_symbol("<mask>")
        src_dict.add_special_symbol(_mask_idx)
        _mask_idx = tgt_dict.add_symbol("<mask>")
        tgt_dict.add_special_symbol(_mask_idx)

        # add language token to dictionaries
        if args.encoder_langtok is not None or args.decoder_langtok:
            for lang_to_add in sorted_langs:
                _land_idx = src_dict.add_symbol(_lang_token(lang_to_add))
                src_dict.add_special_symbol(_land_idx)
                _land_idx = tgt_dict.add_symbol(_lang_token(lang_to_add))
                tgt_dict.add_special_symbol(_land_idx)
        print('| [src] dictionary: {} types, token range: {}'.format(len(src_dict), src_token_range))
        print('| [tgt] dictionary: {} types, token range: {}'.format(len(tgt_dict), tgt_token_range))
        return src_dict, tgt_dict, training, src_token_range, tgt_token_range

    def build_criterion(self, args):
        """
        Build the :class:`~fairseq.criterions.FairseqCriterion` instance for
        this task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.criterions.FairseqCriterion` instance
        """
        from fairseq import criterions
        criterion = criterions.build_criterion(args, self)
        self.criterions['seq2seq'] = criterion
        print('| criterion (seq2seq) {}'.format(self.criterions['seq2seq'].__class__.__name__))
        if self.multitask_mlm:
            self.criterions['mlm'] = MlmLoss(args, self)
            print('| criterion (mlm)     {}'.format(self.criterions['mlm'].__class__.__name__))
        return criterion

    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=0,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.
        """
        
        # set multitask weights for current epoch
        if 'mlm' in dataset.datasets or 'dae' in dataset.datasets:
            self.set_multitask_alpha(epoch)
            print("| [multitask] epoch {:03d}, alpha_mlm: {}, alpha_dae: {}".format(
                epoch, self.mlm_alpha, self.dae_alpha))
        
        # only rebuild iterator when online resampling meeded
        if not self.update_iterator and dataset in self.dataset_to_epoch_iter:
            return self.dataset_to_epoch_iter[dataset]        
        
        seed = seed + epoch
        assert isinstance(dataset, FairseqDataset)

        # set epoch for online resampling
        if isinstance(dataset, RoundRobinZipDatasets):
            for kk, vv in dataset.datasets.items():
                if "translation" in kk:
                    dataset.datasets[kk] = self.set_translation_epoch(vv, seed=seed, epoch=epoch)
                else:
                    # set ratios
                    noising_ratios = self.set_noising_ratio(epoch)
                    dataset.datasets[kk].set_epoch(epoch, **noising_ratios)
                if "mlm" in kk:
                    if isinstance(vv, ResamplingDataset):
                        masking_ratio = vv.dataset.masking_ratio
                    else:
                        masking_ratio = vv.masking_ratio
                    print("| [multitask] epoch {:03d}, MLM masking ratio: {}".format(epoch, masking_ratio))
                if "dae" in kk:
                    if isinstance(vv, ResamplingDataset):
                        text_infilling_ratio = vv.dataset.text_infilling_ratio
                    else:
                        text_infilling_ratio = vv.text_infilling_ratio
                    if text_infilling_ratio > 0:
                        print("| [multitask] epoch {:03d}, DAE text infilling ratio: {}".format(
                            epoch, text_infilling_ratio))
            dataset.set_epoch(epoch)
        else:
            dataset = self.set_translation_epoch(dataset, seed=seed, epoch=epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        indices = data_utils.filter_by_size(
            indices, dataset.size, max_positions, raise_exception=(not ignore_invalid_inputs),
        )

        # create mini-batches with given size constraints
        batch_sampler = data_utils.batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )
        self.dataset_to_epoch_iter[dataset] = epoch_iter
        return epoch_iter

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # parallel data for translation
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        bt_data_path = None
        if self.args.data_bt is not None:
            paths_bt = self.args.data_bt.split(':')
            bt_data_path = paths_bt[epoch % len(paths_bt)]

        language_sample_temperature = getattr(
            self.args, "min_language_sample_temperature", self.args.language_sample_temperature)
        if self.args.language_temperature_scheduler == "static":
            language_sample_temperature = self.args.language_sample_temperature
        resample = language_sample_temperature != 1.0 or self.args.language_temperature_scheduler != "static"
        is_train = (split == self.args.train_subset)

        from fairseq import meters
        load_timer = meters.StopwatchMeter()
        load_timer.start()

        # parallel data
        dataset_loader = LangpairDatasetLoader(
            data_path, split, self.lang_pairs, self.src_dict, self.tgt_dict,
            combine=combine, 
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            encoder_langtok=self.args.encoder_langtok,
            decoder_langtok=self.args.decoder_langtok,
            seed=self.args.seed,
            epoch=epoch,
            resample=(resample and is_train),
            language_sample_temperature=language_sample_temperature,
            language_upsample_max=self.args.language_upsample_max,
            bt_data_path=bt_data_path,
            is_train=is_train,
            downsample_bt=self.args.downsample_bt
        ) 
     
        dataset_mt, data_lengths, bt_lengths = dataset_loader.load_all_langpair_dataset()
        self.data_lengths = data_lengths if is_train else self.data_lengths
        self.data_bt_lengths = bt_lengths
        all_datasets = [("translation", dataset_mt)]

        # mono data
        mono_dataset_loader = None
        if is_train and (self.multitask_mlm or self.multitask_dae):
            paths_mono = self.args.data_mono.split(':') if self.args.data_mono is not None else paths
            assert len(paths_mono) > 0
            data_mono_path = paths_mono[epoch % len(paths_mono)]
            mono_dataset_loader = MonoDatasetLoader(
                data_mono_path, split, self.langs_mono, self.src_dict,
                combine=combine, 
                dataset_impl=self.args.dataset_impl,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                static_noising=self.args.static_noising,
                max_dataset_length=len(dataset_mt) if self.args.downsample_mono else -1,
            )
            noising_ratios = self.set_noising_ratio(epoch=0)

        # mono data for mlm
        if self.multitask_mlm and is_train:
            bpe_cont_marker_map = {"sentencepiece": "sentencepiece", "bpe": "@@", "token": None}
            mono_dataset_loader.set_mlm_hparams(
                src_dict=self.src_dict,
                token_range=self.src_token_range,
                masking_ratio=noising_ratios['masking_ratio'],
                masking_prob=self.args.mlm_masking_prob,
                random_token_prob=self.args.mlm_random_token_prob,
                word_mask=self.args.mlm_word_mask,
                span_mask=self.args.mlm_span_mask,
                span_len_lambda=self.args.mlm_span_lambda,
                bpe_cont_marker=bpe_cont_marker_map.get(self.args.bpe_cont_marker),
            )
            dataset_mlm = mono_dataset_loader.load_mlm_dataset(data_mono_path, self.langs_mlm)
            all_datasets.append(("mlm", dataset_mlm))

        # mono data for dae
        if self.multitask_dae and is_train:
            bpe_cont_marker_map = {"sentencepiece": "sentencepiece", "bpe": "@@", "token": None}
            mono_dataset_loader.set_dae_hparams(
                src_dict=self.tgt_dict,
                token_range=self.tgt_token_range,
                max_word_shuffle_distance=self.args.dae_max_shuffle_distance,
                word_dropout_prob=self.args.dae_dropout_prob,
                word_blanking_prob=self.args.dae_blanking_prob,
                text_infilling_ratio=noising_ratios['text_infilling_ratio'],
                text_infilling_lambda=self.args.dae_span_lambda,
                blank_mask_token=self.args.dae_blanking_with_mask,
                bpe_cont_marker=bpe_cont_marker_map.get(self.args.bpe_cont_marker),
                append_langid_encoder=self.args.encoder_langtok is not None,
                append_langid_decoder=self.args.decoder_langtok,
            )
            dataset_dae = mono_dataset_loader.load_dae_dataset(data_mono_path, self.langs_dae)
            all_datasets.append(("dae", dataset_dae))

        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict(all_datasets), 
            eval_key=None if self.training else "translation")
        
        load_timer.stop()
        print('| loading dataset took total {} seconds'.format(load_timer.sum))

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        src_langs = [self.args.source_lang] * len(src_lengths)
        tgt_langs = [self.args.target_lang] * len(src_lengths)
        dataset_mt = LanguagePairLangidDataset(
            src_tokens, src_lengths, self.source_dictionary,
            src_langs, tgt_langs=tgt_langs,
            encoder_langtok=self.args.encoder_langtok,
            decoder_langtok=self.args.decoder_langtok
        )
        return RoundRobinZipDatasets(
             OrderedDict([("translation", dataset_mt)]),
             eval_key="translation")

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        if not isinstance(model, TransformerMlmModel):
            raise ValueError('TranslationMtlMultitask Task requires a TransformerMlmModel architecture')
        return model

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        for kk in self.criterions.keys():
            self.criterions[kk].train()
        agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}

        """
        torch.set_printoptions(profile="full")
        print("[debug] gpu_id={}, mt : {} ({}), mlm: {} ({}), dae: {} ({})".format(
            self.args.distributed_rank,
            sample['translation']['net_input']['src_tokens'].size(), sample['translation']['net_input']['src_tokens'].numel(),
            sample['mlm']['net_input']['src_tokens'].size(), sample['mlm']['net_input']['src_tokens'].numel(),
            sample['dae']['net_input']['src_tokens'].size(), sample['dae']['net_input']['src_tokens'].numel()),
            flush=True, force=True)
        
        print("\n[debug] gpu_id={}, sample['dae']: {}".format(
            self.args.distributed_rank, sample['dae']['net_input']['src_tokens']),
            flush=True, force=True)
        """

        try:
            # seq2seq for MT
            loss_seq2seq, sample_size, logging_output = self.criterions['seq2seq'](
                model.models['seq2seq'], sample['translation'])
            if ignore_grad:
                loss_seq2seq *= 0
            optimizer.backward(loss_seq2seq)
            agg_loss += loss_seq2seq.detach().item()
            agg_sample_size += sample_size
            agg_logging_output['translation'] = logging_output

            # mlm
            if self.multitask_mlm and self.mlm_alpha > 0:
                loss_mlm, sample_size_mlm, logging_output = self.criterions['mlm'](
                    model.models['mlm'], sample['mlm'])
                if ignore_grad:
                    loss_mlm *= 0
                loss_mlm *= self.mlm_alpha
                optimizer.backward(loss_mlm)
                agg_loss += loss_mlm.detach().item()
                agg_sample_size += sample_size_mlm
                agg_logging_output['mlm'] = logging_output

            # dae
            if self.multitask_dae and self.dae_alpha > 0:
                loss_dae, sample_size_dae, logging_output = self.criterions['seq2seq'](
                    model.models['seq2seq'], sample['dae'])
                if ignore_grad:
                    loss_dae *= 0
                loss_dae *= self.dae_alpha
                optimizer.backward(loss_dae)
                agg_loss += loss_dae.detach().item()
                agg_sample_size += sample_size_dae
                agg_logging_output['dae'] = logging_output

            """
            print("[debug] gpu_id: {}, loss_mt: {}, loss_mlm: {}, loss_dae: {}".format(
                self.args.distributed_rank, loss_seq2seq, loss_mlm, loss_dae), flush=True, force=True)
            """
            
        except RuntimeError as e:
            print("| WARNING: exception: {} on gpu-{}, skipping batch: [mt] {}, [mlm] {}, [dae] {}. logging_output={}".format(
                    e, self.args.distributed_rank,
                    sample['translation']['net_input']['src_tokens'].size(),
                    sample['mlm']['net_input']['src_tokens'].size() if self.multitask_mlm else 'None',
                    sample['dae']['net_input']['src_tokens'].size() if self.multitask_dae else 'None',
                    agg_logging_output), flush=True, force=True)

            """
            torch.set_printoptions(profile="full")
            print("[debug] gpu_id: {}, samples: \n\t{}".format(
                self.args.distributed_rank, 
                "\n\t".join(["{}: {}".format(kk, vv) for kk, vv in sample.items()])), flush=True, force=True)
            
            for i, s in enumerate(sample['translation']['net_input']['src_tokens']):
                print('[debug] [mt]  gpu{}-#{} src     : {}\n[mt]  gpu{}-#{} tgt     : {}'.format(
                    self.args.distributed_rank, i, self.src_dict.string(s),
                    self.args.distributed_rank, i, self.src_dict.string(sample['translation']['target'][i])), flush=True, force=True)
            
            for i, s in enumerate(sample['mlm']['net_input']['src_tokens']):
                print('[debug] [mlm] gpu{}-#{} masked  : {}\n[mlm] gpu{}-#{} padded  : {}'.format(
                    self.args.distributed_rank, i, self.src_dict.string(s, ),
                    self.args.distributed_rank, i, self.src_dict.string(sample['mlm']['target'][i])), flush=True, force=True)

            for i, s in enumerate(sample['dae']['net_input']['src_tokens']):
                print('[debug] [dae] gpu{}-#{} noised  : {}\n[dae] gpu{}-#{} orig    : {}'.format(
                    self.args.distributed_rank, i, self.src_dict.string(s),
                    self.args.distributed_rank, i, self.src_dict.string(sample['dae']['target'][i])), flush=True, force=True)
            
            print("[debug] gpu_id: {}, agg_logging_output: {}".format(
                self.args.distributed_rank, agg_logging_output), flush=True, force=True)
            print("[debug] gpu_id: {}, loss_mt:  {}".format(
                self.args.distributed_rank, loss_seq2seq), flush=True, force=True)
            print("[debug] gpu_id: {}, loss_mlm: {}".format(
                self.args.distributed_rank, loss_mlm), flush=True, force=True)
            print("[debug] gpu_id: {}, loss_dae: {}".format(
                self.args.distributed_rank, loss_dae), flush=True, force=True)
            """

        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        for kk in self.criterions.keys():
            self.criterions[kk].eval()
        with torch.no_grad():
            loss, sample_size, logging_output = self.criterions['seq2seq'](
                model.models['seq2seq'], sample['translation'])
        return loss, sample_size, {'translation': logging_output}

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        seq2seq_models = [m.models['seq2seq'] for m in models]
        with torch.no_grad():
            return generator.generate(seq2seq_models, sample, prefix_tokens=prefix_tokens)

    def grad_denom(self, sample_sizes, criterion):
        return criterion.__class__.grad_denom(sample_sizes)

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        logging_output_keys = []
        for _logging_output in logging_outputs:
            logging_output_keys = _logging_output.keys()
            if len(logging_output_keys) > 0:
                break
        
        if len(logging_output_keys) == 0:
            return {}

        agg_logging_outputs = {
            key: self.criterions[_get_criterion_key(key)].__class__.aggregate_logging_outputs(
                [_logging_output.get(key, {}) for _logging_output in logging_outputs])
            for key in logging_output_keys
        }

        def sum_over_tasks(key):
            return sum(
                _logging_output[key] for _logging_output in agg_logging_outputs.values()
                if key in _logging_output
            )
       
        # flatten logging outputs
        flat_logging_output  = {
            '{}: {}'.format(task_name, k): v
            for task_name, _agg_logging_output in agg_logging_outputs.items()
            for k, v in _agg_logging_output.items()
        }
        flat_logging_output['loss'] = sum_over_tasks('loss')
        if any('nll_loss' in logging_output for logging_output in agg_logging_outputs.values()):
            flat_logging_output['nll_loss'] = sum_over_tasks('nll_loss')
        flat_logging_output['sample_size'] = sum_over_tasks('sample_size')
        flat_logging_output['nsentences'] = sum_over_tasks('nsentences')
        flat_logging_output['ntokens'] = sum_over_tasks('ntokens')
        return flat_logging_output

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        _max_pos = {"translation": (self.args.max_source_positions, self.args.max_target_positions)}
        if self.multitask_mlm:
            _max_pos["mlm"] = (self.args.max_source_positions, )
        if self.multitask_dae:
            _max_pos["dae"] = (self.args.max_source_positions, self.args.max_target_positions)
        return _max_pos

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
    
    def get_sampling_temperature(self, epoch):
        if self.args.language_temperature_scheduler == "linear":
            # epoch * (T-T_0)/warmup_epochs + T_0
            t = self.args.language_sample_temperature - self.args.min_language_sample_temperature
            t *= float(epoch) / self.args.language_sample_warmup_epochs
            return t + self.args.min_language_sample_temperature
        raise NotImplementedError

    def get_resampling_size_ratio(self, epoch):
        new_temp = self.get_sampling_temperature(epoch)
        sample_probs = get_sample_prob(self.data_lengths, new_temp)
        size_ratios = get_size_ratio(
            sample_probs, self.data_lengths, 
            language_upsample_max=self.args.language_upsample_max)
        print("| [resample]  epoch {:03d}, sampling temperature: T = {}".format(epoch, new_temp))
        print("| [resample]  epoch {:03d}, sampling probability by language: {}".format(
            epoch, ", ".join(["{}: {:0.4f}".format(_lang, sample_probs[_i])
            for _i, _lang in enumerate(self.lang_pairs)])))
        print("| [resample]  epoch {:03d}, up/down sampling ratio by language: {}".format(
            epoch, ", ".join(["{}: {:0.2f}".format(_lang, size_ratios[_i])
            for _i, _lang in enumerate(self.lang_pairs)])))
        return size_ratios

    def set_translation_epoch(self, dataset, seed=1, epoch=0):
        """
        set epoch for translation dataset resampling
        update size ratio if changing sampling temperature
        """
        # initialize the dataset with the correct starting epoch
        # has to do so for online resampling
        if epoch > 0 and self.args.language_temperature_scheduler != "static" and \
            epoch <= self.args.language_sample_warmup_epochs:
            # update epoch and size ratios
            size_ratios = self.get_resampling_size_ratio(epoch)
            if self.args.language_upsample_max and self.args.downsample_bt:
                bt_ratio = min(sum(self.data_lengths) / float(self.data_bt_lengths), 1.0)
                print("[resample] epoch {:03d}, downsampling ratio for bt: {}".format(epoch, bt_ratio))
                size_ratios = np.concatenate([size_ratios, np.array([bt_ratio])])
            dataset.set_epoch(epoch, size_ratios=size_ratios)

            # reset sort order
            if isinstance(dataset, SortDataset):
                with data_utils.numpy_seed(seed + epoch):
                    shuffle = np.random.permutation(len(dataset))
                dataset.set_sort_order(sort_order=[shuffle, dataset.sizes])
        else:
            dataset.set_epoch(epoch)
        return dataset
    
    def set_multitask_alpha(self, epoch):
        if self.multitask_mlm:
            self.mlm_alpha = self.get_multitask_alpha(
                epoch=epoch,
                alpha_max=self.args.mlm_alpha,
                alpha_min=self.args.mlm_alpha_min,
                warmup_epochs=self.args.mlm_alpha_warmup,
            )

        if self.multitask_dae: 
            self.dae_alpha = self.get_multitask_alpha(
                epoch=epoch,
                alpha_max=self.args.dae_alpha,
                alpha_min=self.args.dae_alpha_min,
                warmup_epochs=self.args.dae_alpha_warmup,
            )

    def get_multitask_alpha(self, epoch, alpha_max, alpha_min, warmup_epochs):
        if self.args.multitask_scheduler == "static":
            return alpha_max
        if self.args.multitask_scheduler == "linear":
            if epoch >= warmup_epochs:
                return alpha_min
            return alpha_max - (alpha_max - alpha_min) / warmup_epochs * epoch
        raise NotImplementedError

    def set_noising_ratio(self, epoch):
        ratios = {}
        ratios["masking_ratio"] = self.get_noising_ratio(
            epoch=epoch,
            ratio_max=self.args.mlm_masking_ratio,
            ratio_min=self.args.mlm_masking_ratio_min,
            warmup_epochs=self.args.mlm_masking_ratio_warmup_epochs,
            scheduler=self.args.mlm_masking_ratio_scheduler,
        )
        ratios["text_infilling_ratio"] = self.get_noising_ratio(
            epoch=epoch,
            ratio_max=self.args.dae_span_masking_ratio,
            ratio_min=self.args.dae_span_masking_ratio_min,
            warmup_epochs=self.args.dae_span_masking_ratio_warmup_epochs,
            scheduler=self.args.dae_span_masking_ratio_scheduler,
        )
        return ratios

    def get_noising_ratio(self, epoch, ratio_max, ratio_min, warmup_epochs, scheduler):
        if scheduler == "static":
            return ratio_max
        if scheduler == "linear":
            if epoch > warmup_epochs:
                return ratio_max
            return ratio_min + (ratio_max - ratio_min) / warmup_epochs * epoch
        raise NotImplementedError
