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

from fairseq import options, utils
from fairseq.data import (
    Dictionary,
    iterators,
    FairseqDataset,
    data_utils,
)
from . import FairseqTask, register_task

from fairseq.data.language_pair_langid_dataset import LanguagePairLangidDataset
from fairseq.data.langpair_dataset_loader import LangpairDatasetLoader


def _lang_token(lang: str):
    return '__{}__'.format(lang)


def _lang_token_index(dic: Dictionary, lang: str):
    """Return language token index."""
    idx = dic.index(_lang_token(lang))
    assert idx != dic.unk_index, \
        'cannot find language token for lang {}'.format(lang)
    return idx


@register_task('translation_mtl')
class TranslationMtlTask(FairseqTask):
    """A task for training multiple translation models simultaneously.

    Different from multilingual_translation task that iterate over different languages
    Each batch consists of randomly sampled sentences from different language pairs
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
    
        # BT
        parser.add_argument('--data-bt', metavar='DIR', default=None, help='path to back translation data directory')
        parser.add_argument('--downsample-bt', action='store_true',
                            help="downsample bt to match the length of bitext data")

    def __init__(self, args, src_dict, tgt_dict, training):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.lang_pairs = args.lang_pairs
        # eval_lang_pairs for multilingual translation is usually all of the
        # lang_pairs. However for other multitask settings or when we want to
        # optimize for certain languages we want to use a different subset. Thus
        # the eval_lang_pairs class variable is provided for classes that extend
        # this class.
        self.eval_lang_pairs = args.lang_pairs
        # model_lang_pairs will be used to build encoder-decoder model pairs in
        # models.build_model(). This allows multitask type of sub-class can
        # build models other than the input lang_pairs
        self.model_lang_pairs = copy.copy(args.lang_pairs)
        self.langs = sorted(list({x for lang_pair in args.lang_pairs for x in lang_pair.split('-')}))
        self.training = training

        self.use_bt = False
        self.data_bt_lengths = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        src_dict, tgt_dict, training = cls.prepare(args, **kwargs)
        return cls(args, src_dict, tgt_dict, training)
    
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
        sorted_langs = sorted(list({x for lang_pair in args.lang_pairs for x in lang_pair.split('-')}))
        if args.source_lang is not None or args.target_lang is not None:
            training = False
            args.lang_pairs = ['{}-{}'.format(args.source_lang, args.target_lang)]
        else:
            training = True
            args.source_lang, args.target_lang = "src", "tgt"

        # load dictionaries
        paths = args.data.split(':')
        assert len(paths) > 0
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.src.txt'))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.tgt.txt'))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        # add language token
        if args.encoder_langtok is not None or args.decoder_langtok:
            for lang_to_add in sorted_langs:
                src_dict.add_symbol(_lang_token(lang_to_add))
                tgt_dict.add_symbol(_lang_token(lang_to_add))
        print('| [src] dictionary: {} types'.format(len(src_dict)))
        print('| [tgt] dictionary: {} types'.format(len(tgt_dict)))
        return src_dict, tgt_dict, training
    
    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=0,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 0).

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        seed = seed + epoch
        assert isinstance(dataset, FairseqDataset)

        dataset.set_epoch(epoch)

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
        return iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )
    
    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]
        
        bt_data_path = None
        if self.args.data_bt is not None:
            paths_bt = self.args.data_bt.split(':')
            bt_data_path = paths_bt[epoch % len(paths_bt)]

        from fairseq import meters
        load_timer = meters.StopwatchMeter()
        load_timer.start()

        is_train = (split == self.args.train_subset)

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
            resample=False,
            bt_data_path=bt_data_path,
            is_train=is_train,
            downsample_bt=self.args.downsample_bt
        )
        self.datasets[split], _, bt_lengths = dataset_loader.load_all_langpair_dataset()
        self.data_bt_lengths = bt_lengths if is_train else self.data_bt_lengths

        load_timer.stop()
        print('(loading dataset took total {} seconds)'.format(load_timer.sum))
    
    def build_dataset_for_inference(self, src_tokens, src_lengths):
        src_langs = [self.args.source_lang] * len(src_lengths)
        tgt_langs = [self.args.target_lang] * len(src_lengths)
        return LanguagePairLangidDataset(
            src_tokens, src_lengths, self.source_dictionary,
            src_langs, tgt_langs=tgt_langs,
            encoder_langtok=self.args.encoder_langtok,
            decoder_langtok=self.args.decoder_langtok
        )

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

        """
        for i, s in enumerate(sample['net_input']['src_tokens']):
            print('[debug] gpu{}-#{} src     : {}'.format(
                self.args.distributed_rank, i, self.src_dict.string(s)), 
                flush=True)
        """

        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

