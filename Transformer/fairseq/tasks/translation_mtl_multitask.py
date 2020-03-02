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
    ConcatDataset,
    data_utils,
    RoundRobinZipDatasets,
    TokenBlockDataset,
    indexed_dataset,
)
from fairseq.models.transformer_mlm import TransformerMlmModel 
from fairseq.data.language_pair_langid_dataset import LanguagePairLangidDataset
from fairseq.data.masked_seq_dataset import MaskedSeqDataset
from fairseq.criterions.mlm_loss import MlmLoss
from . import FairseqTask, register_task
from .translation_mtl import load_langpair_langid_dataset

def _lang_token(lang: str):
    return '__{}__'.format(lang)


def _lang_token_index(dic: Dictionary, lang: str):
    """Return language token index."""
    idx = dic.index(_lang_token(lang))
    assert idx != dic.unk_index, \
        'cannot find language token for lang {}'.format(lang)
    return idx


def _mask_index(dic: Dictionary):
    idx = dic.index("<mask>")
    assert idx != dic.unk_index, \
        'cannot find special token <mask>'
    return idx


def load_mlm_dataset(data_path, split, langs, src_dict, combine, dataset_impl, tokens_per_sample, masking_ratio):
    datasets = []

    for lang in langs:
        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')
            path = os.path.join(data_path, '{}.{}'.format(split_k, lang))
            ds = indexed_dataset.make_dataset(
                path,
                impl=dataset_impl,
                fix_lua_indexing=True,
                dictionary=src_dict,
            )

            if ds is None:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, path))
            
            datasets.append(TokenBlockDataset(
                ds, ds.sizes, tokens_per_sample, 
                pad=src_dict.pad(),
                eos=src_dict.eos(),
                break_mode='eos',
            ))
            
            print('| {} {} mlm-{} {} examples'.format(data_path, split_k, lang, len(datasets[-1])))
            
            if not combine:
                break
    
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = ConcatDataset(datasets)

    return MaskedSeqDataset(
        dataset, dataset.sizes, src_dict,
        pad_idx=src_dict.pad(),
        mask_idx=_mask_index(src_dict),
        sep_token_idx=src_dict.eos(),
        shuffle=True,
        has_pairs=False,
        masking_ratio=masking_ratio
    )


@register_task('translation_mtl_multitask')
class TranslationMtlMultitaskTask(FairseqTask):
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
        parser.add_argument('--data-mono', metavar='DIR', default=None, help='path to mono data directory')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs: en-de,en-fr,de-fr')
        parser.add_argument('--lang-mlm', default=None,
                            help='comma-separated list of languages with mono data: en,de,fr (for MLM)')
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
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of total tokens over all segments per sample for BERT dataset')
        parser.add_argument('--multitask-mlm', action='store_true',
                            help='use MaskedLM objective together with MT cross-entropy')
        parser.add_argument('--masking-ratio', default=0.15, type=int,
                            help='masking ratio for MaskedLM')

    def __init__(self, args, src_dict, tgt_dict, training):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.lang_pairs = args.lang_pairs
        self.langs_mlm = sorted(list(set(args.lang_mlm.split(","))))
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
        self.criterions = {}

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
        src_dict = Dictionary.load(os.path.join(paths[0], 'dict.src.txt'))
        tgt_dict = Dictionary.load(os.path.join(paths[0], 'dict.tgt.txt'))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        src_dict.add_symbol("<mask>")
        tgt_dict.add_symbol("<mask>")
        # add language token to dictionaries
        if args.encoder_langtok is not None or args.decoder_langtok:
            for lang_to_add in sorted_langs:
                src_dict.add_symbol(_lang_token(lang_to_add))
                tgt_dict.add_symbol(_lang_token(lang_to_add))
        print('| [src] dictionary: {} types'.format(len(src_dict)))
        print('| [tgt] dictionary: {} types'.format(len(tgt_dict)))
        return src_dict, tgt_dict, training
    
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
        self.criterions['mlm'] = MlmLoss(args, self)
        print('| criterion (seq2seq) {}'.format(self.criterions['seq2seq'].__class__.__name__))
        print('| criterion (mlm)     {}'.format(self.criterions['mlm'].__class__.__name__))
        return criterion

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # parallel data for translation
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]
        dataset_mt = load_langpair_langid_dataset(
            data_path, split, self.lang_pairs, self.src_dict, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            encoder_langtok=self.args.encoder_langtok,
            decoder_langtok=self.args.decoder_langtok
        )
        
        # mono data for mlm
        paths_mono = self.args.data_mono.split(':') if self.args.data_mono is not None else paths
        assert len(paths_mono) > 0
        data_mono_path = paths_mono[epoch % len(paths_mono)]
        dataset_mono = load_mlm_dataset(
            data_mono_path, split, self.langs_mlm, self.src_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            tokens_per_sample=self.args.tokens_per_sample,
            masking_ratio=self.args.masking_ratio
        )

        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict([("seq2seq", dataset_mt), ("mlm", dataset_mono)]), 
            eval_key=None if self.training else "seq2seq")

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
             OrderedDict([("seq2seq", dataset_mt)]),
             eval_key="seq2seq")

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        if not isinstance(model, TransformerMlmModel):
            raise ValueError('TranslationMtlMultitaskTask requires a TransformerMlmModel architecture')
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
        print("\n[debug][task:data]=======================")
        print("sample['seq2seq']: {}".format(sample['seq2seq'].keys()))
        print("sample['seq2seq']['net_input']: {}".format(sample['seq2seq']['net_input'].keys()))
        for s in sample['seq2seq']['net_input']['src_tokens']:
            # print(s)
            print(self.src_dict.string(s))
        print("\ntarget:")
        for s in sample['seq2seq']['target']:
            print(self.src_dict.string(s))
        print("[debug][task:data]==========================")
        """

        # seq2seq
        loss_seq2seq, sample_size, logging_output = self.criterions['seq2seq'](model.models['seq2seq'], sample['seq2seq'])
        if ignore_grad:
            loss_seq2seq *= 0
        optimizer.backward(loss_seq2seq)
        # agg_loss += loss_seq2seq.detach().item()
        agg_loss += loss_seq2seq
        agg_sample_size += sample_size
        agg_logging_output['seq2seq'] = logging_output

        """
        print("\n[debug][task:data]=======================")
        print("sample['mlm']: {}".format(sample['mlm'].keys()))
        print("sample['mlm']['net_input']: {}".format(sample['mlm']['net_input'].keys()))
        for i, s in enumerate(sample['mlm']['net_input']['src_tokens']):
            print('\n' + '-' * 30)
            print('truth  :', self.src_dict.string(sample['mlm']['truth'][i]))
            print('\nmasked :', self.src_dict.string(s))
            print('\npadded :', self.src_dict.string(sample['mlm']['target'][i]))
        print("[debug][task:data]==========================")
        """

        # mlm
        loss_mlm, sample_size_mlm, logging_output = self.criterions['mlm'](model.models['mlm'], sample['mlm'])
        if ignore_grad:
            loss_mlm *= 0
        optimizer.backward(loss_mlm)
        # agg_loss += loss_mlm.detach().item()
        agg_loss += loss_mlm
        agg_sample_size += sample_size_mlm
        agg_logging_output['mlm'] = logging_output

        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        for kk in self.criterions.keys():
            self.criterions[kk].eval()
        with torch.no_grad():
            loss, sample_size, logging_output = self.criterions['seq2seq'](model.models['seq2seq'], sample['seq2seq'])
        return loss, sample_size, {'seq2seq': logging_output}

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        seq2seq_models = [m.models['seq2seq'] for m in models]
        with torch.no_grad():
            return generator.generate(seq2seq_models, sample, prefix_tokens=prefix_tokens)

    def grad_denom(self, sample_sizes, criterion):
        return criterion.__class__.grad_denom(sample_sizes)

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        # return criterion.__class__.aggregate_logging_outputs(logging_outputs)
        # logging_output_keys = ["seq2seq", "mlm"]
        logging_output_keys = logging_outputs[0].keys()

        agg_logging_outputs = {
            key: self.criterions[key].__class__.aggregate_logging_outputs(
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
        # return (self.args.max_source_positions, self.args.max_target_positions)
        return {
            "seq2seq": (self.args.max_source_positions, self.args.max_target_positions),
            "mlm": (self.args.max_source_positions, )
        }

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict


    
