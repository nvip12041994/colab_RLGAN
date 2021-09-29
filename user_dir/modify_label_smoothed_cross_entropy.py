# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from fairseq.sequence_generator import SequenceGenerator
from fairseq import scoring

import numpy as np
import torch.nn.functional as F
import copy
from torch.autograd import Variable

def tensor_padding_to_fixed_length(input_tensor,max_len,pad):
    output_tensor = input_tensor.cpu()
    p1d = (0,max_len - input_tensor.shape[1])
    output_tensor = F.pad(input_tensor,p1d,"constant",pad)
    return output_tensor

def numpy_padding_to_fixed_length(imput_list,max_len,pad):
    tmp_list = []
    for item in imput_list:
        len_tmp = len(item)
        number_to_pad = max_len - len_tmp
        np_tmp = np.pad(item, (0, number_to_pad), 'constant', constant_values=(pad, pad))
        tmp_list.append(np_tmp)
    np_array = np.asarray(tmp_list)
    return np_array

def FindMaxLength(lst):
    maxList = max(lst, key = lambda i: len(i))
    maxLength = len(maxList)
    return maxLength

def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}

def train_discriminator(user_parameter,hypo_input,src_input,target_input):
    user_parameter["discriminator"].train()
    fake_labels = Variable(torch.zeros(target_input.size(0)).float())
    fake_labels = fake_labels.to(src_input.device)
        
    disc_out = user_parameter["discriminator"](target_input, hypo_input)
    d_loss = user_parameter["d_criterion"](disc_out.squeeze(1), fake_labels)
    acc = torch.sum(torch.round(disc_out).squeeze(1) == fake_labels).float() / len(fake_labels)
    print("Discriminator accuracy {:.3f}".format(acc))
    user_parameter["d_optimizer"].zero_grad()
    d_loss.backward()
    user_parameter["d_optimizer"].step()

def no_padding_translate_from_sample(network,user_parameter,sample,scorer,src_dict,tgt_dict):
    network.eval()        
    tmp_samples = copy.deepcopy(sample)    
    translator = user_parameter["translator"]
    
    # tmp_target_tokens = tmp_samples['target']
    # target_tokens = tensor_padding_to_fixed_length(tmp_target_tokens,user_parameter["max_len_target"],tgt_dict.pad())
    # target_tokens = target_tokens.to(sample['target'].device)
    
    # tmp_src_tokens = tmp_samples['net_input']['src_tokens']               
    # src_tokens = tensor_padding_to_fixed_length(tmp_src_tokens,user_parameter["max_len_src"],src_dict.pad())        
    # src_tokens = src_tokens.to(sample['net_input']['src_tokens'].device)
    
    # print("padding src_tokens shape" + str(src_tokens.shape))
    # print("padding target_tokens shape" + str(target_tokens.shape))
    
    # tmp_samples['target'] = target_tokens
    # tmp_samples['net_input']['src_tokens'] = src_tokens
    target_tokens  = tmp_samples['target']
    src_tokens = tmp_samples['net_input']['src_tokens']
    
    with torch.no_grad():
        hypos = translator.generate([network],sample = tmp_samples)
    num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
    print(num_generated_tokens)
    tmp_hypo_tokens = []        
    #max_len_hypo = 0
    
    for i, sample_id in enumerate(tmp_samples["id"].tolist()):
        has_target = tmp_samples["target"] is not None

        # Remove padding
        if "src_tokens" in tmp_samples["net_input"]:
            src_token = utils.strip_pad(
                tmp_samples["net_input"]["src_tokens"][i, :], tgt_dict.pad()
            )
        else:
            src_token = None

        target_token = None
        
        if has_target:
            target_token = (
                utils.strip_pad(tmp_samples["target"][i, :], tgt_dict.pad()).int().cpu()
            )
        
        src_str = src_dict.string(src_token, None)
        target_str = tgt_dict.string(
                    target_token,
                    None,
                    escape_unk=True,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(translator),
                )
        # Process top predictions
        #prev_len_hypo = max_len_hypo
        for j, hypo in enumerate(hypos[i][: 1]): # nbest = 1
            hypo_token, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo["tokens"].int().cpu(),
                src_str=src_str,
                alignment=None,
                align_dict=None,
                tgt_dict=tgt_dict,
                remove_bpe=None,
                extra_symbols_to_ignore=get_symbols_to_strip_from_output(translator),
            )
            scorer.add(target_token, hypo_token)
        # if hypo_token.shape[0] > prev_len_hypo:
        #     max_len_hypo = hypo_token.shape[0]       
        tmp_hypo_tokens.append(hypo_token.cpu().tolist())
    
    #np_target_tokens = padding_to_fixed_length(tmp_target_tokens,max_len_target,self.tgt_dict.pad())
    #np_hypo_tokens = numpy_padding_to_fixed_length(tmp_hypo_tokens,user_parameter['max_len_hypo'],tgt_dict.pad())
    
    #target_tokens = torch.Tensor(np_target_tokens).to(src_tokens.dtype).to(src_tokens.device)
    hypo_tokens = torch.Tensor(tmp_hypo_tokens).to(src_tokens.dtype).to(src_tokens.device)        
    
    del tmp_hypo_tokens
    
    torch.cuda.empty_cache()
    print(scorer.result_string())
    return src_tokens,target_tokens,hypo_tokens


def translate_from_sample(network,user_parameter,sample,scorer,src_dict,tgt_dict):
    network.eval()        
    tmp_samples = copy.deepcopy(sample)    
    translator = user_parameter["translator"]
    
    tmp_target_tokens = tmp_samples['target']
    target_tokens = tensor_padding_to_fixed_length(tmp_target_tokens,user_parameter["max_len_target"],tgt_dict.pad())
    target_tokens = target_tokens.to(sample['target'].device)
    
    tmp_src_tokens = tmp_samples['net_input']['src_tokens']               
    src_tokens = tensor_padding_to_fixed_length(tmp_src_tokens,user_parameter["max_len_src"],src_dict.pad())        
    src_tokens = src_tokens.to(sample['net_input']['src_tokens'].device)
    
    # print("padding src_tokens shape" + str(src_tokens.shape))
    # print("padding target_tokens shape" + str(target_tokens.shape))
    
    tmp_samples['target'] = target_tokens
    tmp_samples['net_input']['src_tokens'] = src_tokens
    
    
    with torch.no_grad():
        hypos = translator.generate([network],sample = tmp_samples)
    num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
    print(num_generated_tokens)
    tmp_hypo_tokens = []        
    #max_len_hypo = 0
    
    for i, sample_id in enumerate(tmp_samples["id"].tolist()):
        has_target = tmp_samples["target"] is not None

        # Remove padding
        if "src_tokens" in tmp_samples["net_input"]:
            src_token = utils.strip_pad(
                tmp_samples["net_input"]["src_tokens"][i, :], tgt_dict.pad()
            )
        else:
            src_token = None

        target_token = None
        
        if has_target:
            target_token = (
                utils.strip_pad(tmp_samples["target"][i, :], tgt_dict.pad()).int().cpu()
            )
        
        src_str = src_dict.string(src_token, None)
        target_str = tgt_dict.string(
                    target_token,
                    None,
                    escape_unk=True,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(translator),
                )
        # Process top predictions
        #prev_len_hypo = max_len_hypo
        for j, hypo in enumerate(hypos[i][: 1]): # nbest = 1
            hypo_token, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo["tokens"].int().cpu(),
                src_str=src_str,
                alignment=None,
                align_dict=None,
                tgt_dict=tgt_dict,
                remove_bpe=None,
                extra_symbols_to_ignore=get_symbols_to_strip_from_output(translator),
            )
            scorer.add(target_token, hypo_token)
        # if hypo_token.shape[0] > prev_len_hypo:
        #     max_len_hypo = hypo_token.shape[0]       
        tmp_hypo_tokens.append(hypo_token.cpu().tolist())
    
    #np_target_tokens = padding_to_fixed_length(tmp_target_tokens,max_len_target,self.tgt_dict.pad())
    np_hypo_tokens = numpy_padding_to_fixed_length(tmp_hypo_tokens,user_parameter['max_len_hypo'],tgt_dict.pad())
    
    #target_tokens = torch.Tensor(np_target_tokens).to(src_tokens.dtype).to(src_tokens.device)
    hypo_tokens = torch.Tensor(np_hypo_tokens).to(src_tokens.dtype).to(src_tokens.device)        
    del tmp_samples
    del tmp_target_tokens
    del tmp_src_tokens
    del tmp_hypo_tokens
    del np_hypo_tokens
    torch.cuda.empty_cache()
    print(scorer.result_string())
    return src_tokens,target_tokens,hypo_tokens

@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "modify_label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        
        self.tgt_dict = task.target_dictionary
        self.src_dict = task.source_dictionary
        self.scorer = scoring.build_scorer("bleu", self.tgt_dict)
        

    def forward(self, model, sample, user_parameter = None,reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        
        net_output = model(**sample["net_input"])       
        #-------------------------MLE----------------------
        model.train()
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if user_parameter is not None:    
            # part II: train the discriminator
            no_padding_translate_from_sample
            src_tokens, target_tokens, hypo_tokens = no_padding_translate_from_sample(model,user_parameter,sample,self.scorer,self.src_dict,self.tgt_dict)
            #src_tokens, target_tokens, hypo_tokens = translate_from_sample(model,user_parameter,sample,self.scorer,self.src_dict,self.tgt_dict)
            output_parameter = {
                "src_tokens": src_tokens,
                "target_tokens": target_tokens,
                "hypo_tokens": hypo_tokens,
            }
        # train_discriminator(user_parameter,
        #                     hypo_input = hypo_tokens,
        #                     target_input=target_tokens,
        #                     src_input=src_tokens,
        #                    )
        
        # del target_tokens
        # del src_tokens
        # del hypo_tokens
        # torch.cuda.empty_cache()
        
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
