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
    data_utils,
    RoundRobinZipDatasets,
)
from . import FairseqTask, register_task

from fairseq.models.transformer_mlm import TransformerMlmModel 
from fairseq.criterions.mlm_loss import MlmLoss
from fairseq.data.language_pair_langid_dataset import LanguagePairLangidDataset
from fairseq.data.langpair_dataset_loader import (
    MonoDatasetLoader,
    LangpairDatasetLoader
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

        # MLM
        parser.add_argument('--multitask-mlm', action='store_true',
                            help='use MaskedLM objective together with MT cross-entropy')
        parser.add_argument('--lang-mlm', default=None,
                            help='comma-separated list of languages with mono data: en,de,fr (for MLM)')
        parser.add_argument('--tokens-per-sample', default=1024, type=int,
                            help='max number of total tokens over all segments per sample for BERT dataset')
        parser.add_argument('--mlm-masking-ratio', default=0.15, type=float,
                            help='masking ratio for MaskedLM')
        parser.add_argument('--mlm-masking-prob', default=0.8, type=float,
                            help='masking ratio for MaskedLM')
        parser.add_argument('--mlm-random-token-prob', default=0.1, type=float,
                            help='masking ratio for MaskedLM')
        
        # DAE
        parser.add_argument('--multitask-dae', action='store_true',
                            help='use DAE objective together with MT cross-entropy')
        parser.add_argument('--lang-dae', default=None,
                            help='comma-separated list of languages with mono data: en,de,fr (for DAE)')
        parser.add_argument('--dae-max-shuffle-distance', default=3.0, type=float,
                            help='maximum shuffle distance for DAE')
        parser.add_argument('--dae-dropout-prob', default=0.1, type=float,
                            help='token dropout probability for DAE')
        parser.add_argument('--dae-blanking-prob', default=0.2, type=float,
                            help='token blanking probability for DAE')
        parser.add_argument('--dae-blanking-with-mask', action='store_true',
                            help='token blanking with <mask> token instead of <unk> for DAE')
        parser.add_argument('--dae-bpe-cont-marker', default="sentencepiece", type=str,
                            help='word level (if sentencepiece or bpe) or token level (others) noising')       
        parser.add_argument('--static-noising', action='store_true',
                            help='use same noising for same example in each epoch (both mlm and dae)')            
        

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
        self.criterions = {}
        self.langs_mlm = sorted(list(set(args.lang_mlm.split(",")))) if args.lang_mlm is not None else []
        self.langs_dae = sorted(list(set(args.lang_dae.split(",")))) if args.lang_dae is not None else []
        self.langs_mono = sorted(list(set(self.langs_mlm + self.langs_dae)))
        self.multitask_mlm = args.multitask_mlm
        self.multitask_dae = args.multitask_dae

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
        print('| criterion (seq2seq) {}'.format(self.criterions['seq2seq'].__class__.__name__))
        if self.multitask_mlm:
            self.criterions['mlm'] = MlmLoss(args, self)
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

        bt_data_path = None
        if self.args.data_bt is not None:
            paths_bt = self.args.data_bt.split(':')
            bt_data_path = paths_bt[epoch % len(paths_bt)]

        is_train = (split == self.args.train_subset)

        from fairseq import meters
        load_timer = meters.StopwatchMeter()
        load_timer.start()

        langpair_dataset_loader = LangpairDatasetLoader(
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
            bt_data_path=bt_data_path,
            is_train=is_train
        ) 
        dataset_mt, _ = langpair_dataset_loader.load_all_langpair_dataset()
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
                static_noising=self.args.static_noising
            )

        # mono data for mlm
        if self.multitask_mlm and is_train:
            mono_dataset_loader.set_mlm_hparams(
                src_dict=self.src_dict,
                tokens_per_sample=self.args.tokens_per_sample,
                masking_ratio=self.args.mlm_masking_ratio,
                masking_prob=self.args.mlm_masking_prob,
                random_token_prob=self.args.mlm_random_token_prob,
            )
            dataset_mlm = mono_dataset_loader.load_mlm_dataset(data_mono_path, self.langs_mlm)
            all_datasets.append(("mlm", dataset_mlm))

        # mono data for dae
        if self.multitask_dae and is_train:
            bpe_cont_marker_map = {"sentencepiece": "sentencepiece", "bpe": "@@"}
            mono_dataset_loader.set_dae_hparams(
                src_dict=self.tgt_dict,
                max_word_shuffle_distance=self.args.dae_max_shuffle_distance,
                word_dropout_prob=self.args.dae_dropout_prob,
                word_blanking_prob=self.args.dae_blanking_prob,
                blank_mask_token=self.args.dae_blanking_with_mask,
                bpe_cont_marker=bpe_cont_marker_map.get(self.args.dae_bpe_cont_marker),
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
        print("\n[debug][task:data]=======================")
        print("sample['translation']: {}".format(sample['translation'].keys()))
        print("sample['translation']['net_input']: {}".format(sample['translation']['net_input'].keys()))
        for s in sample['translation']['net_input']['src_tokens']:
            # print(s)
            print(self.src_dict.string(s))
        print("\ntarget:")
        for s in sample['translation']['target']:
            print(self.src_dict.string(s))
        print("[debug][task:data]==========================")
        """

        # seq2seq for MT
        loss_seq2seq, sample_size, logging_output = self.criterions['seq2seq'](
            model.models['seq2seq'], sample['translation'])
        if ignore_grad:
            loss_seq2seq *= 0
        optimizer.backward(loss_seq2seq)
        agg_loss += loss_seq2seq.detach().item()
        # agg_loss += loss_seq2seq
        agg_sample_size += sample_size
        agg_logging_output['translation'] = logging_output

        """
        print("\n[debug][task:data]=======================")
        print("sample['mlm']: {}".format(sample['mlm'].keys()))
        print("sample['mlm']['net_input']: {}".format(sample['mlm']['net_input'].keys()))
        for i, s in enumerate(sample['mlm']['net_input']['src_tokens']):
            print('\n' + '-' * 30)
            # print('truth  :', self.src_dict.string(sample['mlm']['truth'][i]))
            print('\nmasked :', self.src_dict.string(s))
            print('\npadded :', self.src_dict.string(sample['mlm']['target'][i]))
        print("[debug][task:data]==========================")
        """

        # mlm
        if self.multitask_mlm:
            loss_mlm, sample_size_mlm, logging_output = self.criterions['mlm'](
                model.models['mlm'], sample['mlm'])
            if ignore_grad:
                loss_mlm *= 0
            optimizer.backward(loss_mlm)
            agg_loss += loss_mlm.detach().item()
            # agg_loss += loss_mlm
            agg_sample_size += sample_size_mlm
            agg_logging_output['mlm'] = logging_output

        """
        print("\n[debug][task:data]=======================")
        print("sample['dae']: {}".format(sample['dae'].keys()))
        print("sample['dae']['net_input']: {}".format(sample['dae']['net_input'].keys()))
        for i, s in enumerate(sample['dae']['net_input']['src_tokens']):
            print('\n' + '-' * 30)
            print('noising   :', self.src_dict.string(s))
            print('\noriginal:', self.src_dict.string(sample['dae']['target'][i]))
        print("[debug][task:data]==========================")
        """

        # dae
        if self.multitask_dae:
            loss_dae, sample_size_dae, logging_output = self.criterions['seq2seq'](
                model.models['seq2seq'], sample['dae'])
            if ignore_grad:
                loss_dae *= 0
            optimizer.backward(loss_dae)
            agg_loss += loss_dae.detach().item()
            agg_sample_size += sample_size_dae
            agg_logging_output['dae'] = logging_output

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
        logging_output_keys = logging_outputs[0].keys()

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



