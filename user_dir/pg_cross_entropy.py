# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

# ================================================================
import os
import torch
import numpy as np
import torch.nn.functional as F
import copy
from torch.autograd import Variable
from fairseq.sequence_generator import SequenceGenerator
from fairseq import scoring
import time


def tensor_padding_to_fixed_length(input_tensor, max_len, pad):
    output_tensor = input_tensor.cpu()
    p1d = (0, max_len - input_tensor.shape[0])
    output_tensor = F.pad(input_tensor, p1d, "constant", pad)
    return output_tensor.cuda()


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def FindMaxLength(lst):
    maxList = max(lst, key=lambda i: len(i))
    maxLength = len(maxList)
    return maxLength


def train_discriminator(user_parameter, hypo_input, src_input, target_input):
    user_parameter["discriminator"].train()
    user_parameter["d_criterion"].train()
    fake_labels = Variable(torch.zeros(src_input.size(0)).float())
    fake_labels = fake_labels.to(src_input.device)

    disc_out = user_parameter["discriminator"](src_input, hypo_input)
    d_loss = user_parameter["d_criterion"](disc_out.squeeze(1), fake_labels)
    acc = torch.sum(torch.round(disc_out).squeeze(
        1) == fake_labels).float() / len(fake_labels)
    #print("Discriminator accuracy {:.3f}".format(acc))
    user_parameter["d_optimizer"].zero_grad()
    d_loss.backward()
    user_parameter["d_optimizer"].step()
    torch.cuda.empty_cache()


def get_token_translate_from_sample_no_bleu(network, user_parameter, sample, scorer, src_dict, tgt_dict):
    network.eval()

    translator = user_parameter["translator"]

    target_tokens = sample['target']
    src_tokens = sample['net_input']['src_tokens']

    with torch.no_grad():
        hypos = translator.generate(
            [network], sample=sample, prefix_tokens=None)

    tmp = []
    for i in range(len(hypos)):
        tmp.append(hypos[i][0]["tokens"])  # nbest = 1 so only one translate

    max_len = FindMaxLength(tmp)
    hypo_tokens_out = torch.empty(
        size=(len(tmp), max_len), dtype=torch.int64, device='cuda')
    for i in range(len(tmp)):
        hypo_tokens_out[i] = tensor_padding_to_fixed_length(
            tmp[i], max_len, tgt_dict.pad())

    return src_tokens, target_tokens, hypo_tokens_out


def get_token_translate_from_sample(network, user_parameter, sample, scorer, src_dict, tgt_dict):
    network.eval()

    translator = user_parameter["translator"]
    target_tokens = sample['target']
    src_tokens = sample['net_input']['src_tokens']

    with torch.no_grad():
        hypos = translator.generate([network], sample=sample)

    tmp = []
    bleus = []
    for i, sample_id in enumerate(sample["id"].tolist()):
        # print("==================")
        has_target = sample["target"] is not None

        # Remove padding
        if "src_tokens" in sample["net_input"]:
            src_token = utils.strip_pad(
                sample["net_input"]["src_tokens"][i, :], tgt_dict.pad()
            )
        else:
            src_token = None

        target_token = None

        if has_target:
            target_token = (
                utils.strip_pad(sample["target"][i, :],
                                tgt_dict.pad()).int().cpu()
            )

        src_str = src_dict.string(src_token, None)
        target_str = tgt_dict.string(
            target_token,
            None,
            escape_unk=True,
            extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                translator),
        )
        # Process top predictions

        for j, hypo in enumerate(hypos[i][: 1]):  # nbest = 1
            hypo_token, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo["tokens"].int().cpu(),
                src_str=src_str,
                alignment=None,
                align_dict=None,
                tgt_dict=tgt_dict,
                remove_bpe=None,
                extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                    translator),
            )
            tmp.append(hypo["tokens"])
            scorer.add(target_token, hypo_token)
            bleu_score = scorer.score()/100
            bleus.append(bleu_score)

    max_len = FindMaxLength(tmp)
    hypo_tokens_out = torch.empty(
        size=(len(tmp), max_len), dtype=torch.int64, device='cuda')
    for i in range(len(tmp)):
        hypo_tokens_out[i] = tensor_padding_to_fixed_length(
            tmp[i], max_len, tgt_dict.pad())

    torch.cuda.empty_cache()
    return src_tokens, target_tokens, hypo_tokens_out, bleus


@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("pg_cross_entropy", dataclass=CrossEntropyCriterionConfig)
class CrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

        self.tgt_dict = task.target_dictionary
        self.src_dict = task.source_dictionary
        self.vocab_size = len(task.target_dictionary)
        self.scorer = scoring.build_scorer("bleu", self.tgt_dict)
        self.entropy_coeff = 1
        self.gamma = 0.99

    def _returns_advantages(self, rewards, dones, values, next_value):
        """Returns the cumulative discounted rewards at each time step

        Parameters
        ----------
        rewards : array
            An array of shape (batch_size,) containing the rewards given by the env
        dones : array
            An array of shape (batch_size,) containing the done bool indicator given by the env
        values : array
            An array of shape (batch_size,) containing the values given by the value network
        next_value : float
            The value of the next state given by the value network

        Returns
        -------
        returns : array
            The cumulative discounted rewards
        advantages : array
            The advantages
        """

        returns = np.append(np.zeros_like(rewards), [next_value.cpu()], axis=0)
        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])

        returns = returns[:-1] #remove last element
        returns_tensor = torch.from_numpy(returns)
        returns_tensor = torch.divide(returns_tensor,returns_tensor.shape[0])
        advantages = torch.sub(returns_tensor,values.T[0].cpu())
        return returns_tensor.numpy(), advantages
    
    def forward(self, model, sample, user_parameter=None, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        
        #loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        bsz, src_len = sample['net_input']['src_tokens'].size()[:2]
        if user_parameter is not None:
            #start_time = time.time()
            observations, target_tokens, actions, bleus = get_token_translate_from_sample(model,
                                                                                            user_parameter,
                                                                                            sample,
                                                                                            self.scorer,
                                                                                            self.src_dict,
                                                                                            self.tgt_dict)
            #a = time.time() - start_time
            with torch.no_grad():
                values = user_parameter["discriminator"](observations, actions)
            dones = np.empty((bsz,), dtype=np.bool_)
            
            for i,bleu in enumerate(bleus):
                if bleu >=0.4:
                    dones[i] = True
                else:
                    dones[i] = False
                        
            # Update episode_count
            # If our epiosde didn't end on the last step we need to compute the value for the last state
            if dones[-1]:
                next_value = 0
            else:
                next_value = values[-1]
                
            #episode_count = sum(dones)
            
            # Compute returns and advantages
            returns, advantages = self._returns_advantages(bleus, dones, values, next_value)
            user_parameter["returns"] = returns
            # Learning step !
            lprobs, target = self.compute_lprob(model, net_output, sample)
            #a = lprobs.view(bsz, -1, self.vocab_size),
            #t = F.one_hot(actions, self.vocab_size)
            # indices_buf = torch.multinomial(
            #     lprobs.exp_().view(bsz, -1),
            #     1,
            #     replacement=True,
            # ).view(bsz, 1)
            
            # l = F.one_hot(indices_buf, self.vocab_size)
            
            loss_entropy = - (torch.exp(lprobs)* lprobs).sum(-1)
            lprobs = (lprobs.T*advantages.to(lprobs.device)).T
            
            lprobs = lprobs.view(-1, lprobs.size(-1))
            loss_reward = F.nll_loss(
                lprobs,
                target,
                ignore_index=self.padding_idx,
                reduction="sum" if reduce else "none",
                # reduction="none",
            )
            #average_reward = torch.mean(reward)
            loss = - loss_reward + self.entropy_coeff * loss_entropy
        else:
            loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)

        # real_random_number = int.from_bytes(os.urandom(1), byteorder="big")
        # if real_random_number > 127:
                
        sample_size = (
            sample["target"].size(
                0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        return loss, sample_size, logging_output

    def compute_lprob(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)        
        target = model.get_targets(sample, net_output).view(-1)
        return lprobs, target

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="mean" if reduce else "none",
            # reduction="none",
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(
                    meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
