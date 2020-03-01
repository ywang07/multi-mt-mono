# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import math
import torch
import torch.nn.functional as F

from fairseq import utils
from . import FairseqCriterion, register_criterion


def compute_cross_entropy_loss(logits, targets, ignore_index=-100):
    """
    Function to compute the cross entropy loss. The default value of
    ignore_index is the same as the default value for F.cross_entropy in
    pytorch.
    """
    assert logits.size(0) == targets.size(-1), \
        "Logits and Targets tensor shapes don't match up"

    loss = F.nll_loss(
        F.log_softmax(logits, -1, dtype=torch.float32),
        targets,
        reduction="sum",
        ignore_index=ignore_index,
    )
    return loss


@register_criterion('mlm_loss')
class MlmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    This optionally also computes the next sentence prediction (NSP) loss and
    adds it to the overall loss based on the specified args. There are three
    cases to consider:
        1) Generic MLM training without NSP loss. In this case sentence_targets
           and sentence_logits are both None.
        2) BERT training without NSP loss. In this case sentence_targets is
           not None but sentence_logits is None and we should not be computing
           a sentence level loss.
        3) BERT training with NSP loss. In this case both sentence_targets and
           sentence_logits are not None and we should be computing a sentence
           level loss. The weight of the sentence level loss is specified as
           an argument.
    """

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # compute MLM loss
        masked_tokens = sample['target'].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum().item()

        # (Rare case) When all tokens are masked, the model results in empty
        # tensor and gives CUDA error.
        if sample_size == 0:
            masked_tokens = None

        logits = model(**sample['net_input'], masked_tokens=masked_tokens)[0]
        targets = model.get_targets(sample, [logits])

        if sample_size != 0:
            targets = targets[masked_tokens]

        """
        print("\n[debug][mlm_loss]=======================")
        print("sample_size:   {}".format(sample_size))
        print("masked_tokens: {}".format(masked_tokens))
        print("targets:       {}".format(targets))
        print("logits:        {}".format(logits.size()))
        print("[debug][mlm_loss]=======================")
        """

        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        loss = compute_cross_entropy_loss(logits, targets, self.padding_idx)

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        """
        print("\n[debug][mlm_loss:aggregate]=======================")
        print("loss_sum:    {}".format(loss_sum))
        print("ntokens:     {}".format(ntokens))
        print("nsentences:  {}".format(nsentences))
        print("sample_size: {}".format(sample_size))
        print("[debug][mlm_loss:aggregate]=======================")
        """

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'nll_loss': nll_loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
