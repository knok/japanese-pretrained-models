# coding=utf-8
# Copyright 2019-2021, the HuggingFace Inc. team and Facebook, Inc.
# Copyright 2021 NOKUBI Takatsugu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
"""
distiller for gpt-2-japanese
use multiple files as train data
"""

import math
import os
import time
import sys
from typing import List

import psutil
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
sys.path.append('../src')

from grouped_batch_sampler import GroupedBatchSampler, create_lengths_groups
from lm_seqs_dataset import LmSeqsDataset
from transformers import get_linear_schedule_with_warmup
from task.pretrain_gpt2.data_source import DataSource
from task.pretrain_gpt2.train import load_docs_from_filepath
from utils import logger

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from distiller import Distiller

def collate_fn(batch_data):
    return batch_data

# data loader
def load_data(fname, tokenizer):
    bos = tokenizer.special_tokens_map["bos_token"]  # `<|endoftext|>`
    sep = tokenizer.special_tokens_map["eos_token"]  # `<|endoftext|>`
    rslt = []
    with open(fname, encoding='utf-8') as f:
        for line in f:
            text = f"{bos} {line.strip()} {sep}"
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            rslt.append(token_ids)
    rslt_ = [np.int32(d) for d in rslt]
    return rslt_

""" Dataset to distilled models
    adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import logger


class JpLmSeqsDataset(Dataset):
    """Custom Dataset wrapping language modeling sequences.

    Each sample will be retrieved by indexing the list of token_ids and their corresponding lengths.

    Input:
    ------
        params: `NameSpace` parameters
        data: `List[np.array[int]]
    """

    def __init__(self, config, tokenizer, docs, stage, randomize, params):
        # Attributes
        self.max_seq_len = config.max_seq_len
        # Other attributes
        self.tokenizer = tokenizer
        self.stage = stage
        self.randomize = randomize
        self.statistics = {"n_docs": 0, "n_sents": 0, "n_tokens": 0}

        self.params = params

        # Load dataset
        self.docs = docs

        self.token_ids = self.docs
        self.lengths = np.array([len(t) for t in docs])

        self.check()
        self.remove_long_sequences()
        self.remove_empty_sequences()
        self.remove_unknown_sequences()
        self.check()
        self.print_statistics()

    def __getitem__(self, index):
        return (self.token_ids[index], self.lengths[index])

    def __len__(self):
        return len(self.lengths)

    def check(self):
        """
        Some sanity checks
        """
        assert len(self.token_ids) == len(self.lengths)
        assert all(self.lengths[i] == len(self.token_ids[i]) for i in range(len(self.lengths)))

    def remove_long_sequences(self):
        """
        Sequences that are too long are split by chunk of max_model_input_size.
        """
        max_len = self.params.max_model_input_size
        indices = self.lengths > max_len
        logger.info(f"Splitting {sum(indices)} too long sequences.")

        def divide_chunks(l, n):
            return [l[i : i + n] for i in range(0, len(l), n)]

        new_tok_ids = []
        new_lengths = []
        if self.params.mlm:
            cls_id, sep_id = self.params.special_tok_ids["cls_token"], self.params.special_tok_ids["sep_token"]
        else:
            cls_id, sep_id = self.params.special_tok_ids["bos_token"], self.params.special_tok_ids["eos_token"]

        for seq_, len_ in zip(self.token_ids, self.lengths):
            assert (seq_[0] == cls_id) and (seq_[-1] == sep_id), seq_
            if len_ <= max_len:
                new_tok_ids.append(seq_)
                new_lengths.append(len_)
            else:
                sub_seqs = []
                for sub_s in divide_chunks(seq_, max_len - 2):
                    if sub_s[0] != cls_id:
                        sub_s = np.insert(sub_s, 0, cls_id)
                    if sub_s[-1] != sep_id:
                        sub_s = np.insert(sub_s, len(sub_s), sep_id)
                    assert len(sub_s) <= max_len
                    assert (sub_s[0] == cls_id) and (sub_s[-1] == sep_id), sub_s
                    sub_seqs.append(sub_s)

                new_tok_ids.extend(sub_seqs)
                new_lengths.extend([len(l) for l in sub_seqs])

        self.token_ids = np.array(new_tok_ids)
        self.lengths = np.array(new_lengths)

    def remove_empty_sequences(self):
        """
        Too short sequences are simply removed. This could be tuned.
        """
        init_size = len(self)
        indices = self.lengths > 11
        self.token_ids = self.token_ids[indices]
        self.lengths = self.lengths[indices]
        new_size = len(self)
        logger.info(f"Remove {init_size - new_size} too short (<=11 tokens) sequences.")

    def remove_unknown_sequences(self):
        """
        Remove sequences with a (too) high level of unknown tokens.
        """
        if "unk_token" not in self.params.special_tok_ids:
            return
        else:
            unk_token_id = self.params.special_tok_ids["unk_token"]
        init_size = len(self)
        unk_occs = np.array([np.count_nonzero(a == unk_token_id) for a in self.token_ids])
        indices = (unk_occs / self.lengths) < 0.5
        self.token_ids = self.token_ids[indices]
        self.lengths = self.lengths[indices]
        new_size = len(self)
        logger.info(f"Remove {init_size - new_size} sequences with a high level of unknown tokens (50%).")

    def print_statistics(self):
        """
        Print some statistics on the corpus. Only the master process.
        """
        if not self.params.is_master:
            return
        logger.info(f"{len(self)} sequences")
        # data_len = sum(self.lengths)
        # nb_unique_tokens = len(Counter(list(chain(*self.token_ids))))
        # logger.info(f'{data_len} tokens ({nb_unique_tokens} unique)')

        # unk_idx = self.params.special_tok_ids['unk_token']
        # nb_unknown = sum([(t==unk_idx).sum() for t in self.token_ids])
        # logger.info(f'{nb_unknown} unknown tokens (covering {100*nb_unknown/data_len:.2f}% of the data)')

    def batch_sequences(self, batch):
        """
        Do the padding and transform into torch.tensor.
        """
        token_ids = [t[0] for t in batch]
        lengths = [t[1] for t in batch]
        assert len(token_ids) == len(lengths)

        # Max for paddings
        max_seq_len_ = max(lengths)

        # Pad token ids
        if self.params.mlm:
            pad_idx = self.params.special_tok_ids["pad_token"]
        else:
            pad_idx = self.params.special_tok_ids["unk_token"]
        tk_ = [list(t.astype(int)) + [pad_idx] * (max_seq_len_ - len(t)) for t in token_ids]
        assert len(tk_) == len(token_ids)
        assert all(len(t) == max_seq_len_ for t in tk_)

        tk_t = torch.tensor(tk_)  # (bs, max_seq_len_)
        lg_t = torch.tensor(lengths)  # (bs)
        return tk_t, lg_t

class JpDistiller(Distiller):
    def __init__(self, params: dict, dataset_files: List[str] , token_probs: torch.tensor, \
            student: nn.Module, teacher: nn.Module, config, tokenizer):
        logger.info("Initializing Distiller")
        self.params = params
        self.dump_path = params.dump_path
        self.multi_gpu = params.multi_gpu
        self.fp16 = params.fp16

        self.student = student
        self.teacher = teacher

        self.student_config = student.config
        self.vocab_size = student.config.vocab_size
        self.config = config
        self.tokenizer = tokenizer

        if params.n_gpu <= 1:
            sampler = RandomSampler
        else:
            sampler = DistributedSampler
        self.sampler = sampler

        if params.group_by_size:
            # ignore
            groups = create_lengths_groups(lengths=dataset.lengths, k=params.max_model_input_size)
            sampler = GroupedBatchSampler(sampler=sampler, group_ids=groups, batch_size=params.batch_size)
        else:
            sampler = BatchSampler
        self.batch_sampler = sampler

        self.dataloader = DataLoader
        self.dataset_files = dataset_files

        self.temperature = params.temperature
        assert self.temperature > 0.0

        self.alpha_ce = params.alpha_ce
        self.alpha_mlm = params.alpha_mlm
        self.alpha_clm = params.alpha_clm
        self.alpha_mse = params.alpha_mse
        self.alpha_cos = params.alpha_cos

        self.mlm = params.mlm
        if self.mlm:
            logger.info("Using MLM loss for LM step.")
            self.mlm_mask_prop = params.mlm_mask_prop
            assert 0.0 <= self.mlm_mask_prop <= 1.0
            assert params.word_mask + params.word_keep + params.word_rand == 1.0
            self.pred_probs = torch.FloatTensor([params.word_mask, params.word_keep, params.word_rand])
            self.pred_probs = self.pred_probs.to(f"cuda:{params.local_rank}") if params.n_gpu > 0 else self.pred_probs
            self.token_probs = token_probs.to(f"cuda:{params.local_rank}") if params.n_gpu > 0 else token_probs
            if self.fp16:
                self.pred_probs = self.pred_probs.half()
                self.token_probs = self.token_probs.half()
        else:
            logger.info("Using CLM loss for LM step.")

        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.last_loss = 0
        self.last_loss_ce = 0
        self.last_loss_mlm = 0
        self.last_loss_clm = 0
        if self.alpha_mse > 0.0:
            self.last_loss_mse = 0
        if self.alpha_cos > 0.0:
            self.last_loss_cos = 0
        self.last_log = 0

        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        if self.alpha_mse > 0.0:
            self.mse_loss_fct = nn.MSELoss(reduction="sum")
        if self.alpha_cos > 0.0:
            self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

        logger.info("--- Initializing model optimizer")
        assert params.gradient_accumulation_steps >= 1
        self.num_steps_epoch = config.num_steps # len(self.dataloader)
        num_train_optimization_steps = (
            int(self.num_steps_epoch / params.gradient_accumulation_steps * params.n_epoch) + 1
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": params.weight_decay,
            },
            {
                "params": [
                    p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        logger.info(
            "------ Number of trainable parameters (student): %i"
            % sum([p.numel() for p in self.student.parameters() if p.requires_grad])
        )
        logger.info("------ Number of parameters (student): %i" % sum([p.numel() for p in self.student.parameters()]))
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=params.learning_rate, eps=params.adam_epsilon, betas=(0.9, 0.98)
        )

        warmup_steps = math.ceil(num_train_optimization_steps * params.warmup_prop)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
        )

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            logger.info(f"Using fp16 training: {self.params.fp16_opt_level} level")
            self.student, self.optimizer = amp.initialize(
                self.student, self.optimizer, opt_level=self.params.fp16_opt_level
            )
            self.teacher = self.teacher.half()

        if self.multi_gpu:
            if self.fp16:
                from apex.parallel import DistributedDataParallel

                logger.info("Using apex.parallel.DistributedDataParallel for distributed training.")
                self.student = DistributedDataParallel(self.student)
            else:
                from torch.nn.parallel import DistributedDataParallel

                logger.info("Using nn.parallel.DistributedDataParallel for distributed training.")
                self.student = DistributedDataParallel(
                    self.student,
                    device_ids=[params.local_rank],
                    output_device=params.local_rank,
                    find_unused_parameters=True,
                )

        self.is_master = params.is_master
        if self.is_master:
            logger.info("--- Initializing Tensorboard")
            self.tensorboard = SummaryWriter(log_dir=os.path.join(self.dump_path, "log", "train"))
            self.tensorboard.add_text(tag="config/training", text_string=str(self.params), global_step=0)
            self.tensorboard.add_text(tag="config/student", text_string=str(self.student_config), global_step=0)

    def train(self):
        """
        The real training loop.
        """
        if self.is_master:
            logger.info("Starting training")
        self.last_log = time.time()
        self.student.train()
        self.teacher.eval()

        for _ in range(self.params.n_epoch):
            if self.is_master:
                logger.info(f"--- Starting epoch {self.epoch}/{self.params.n_epoch-1}")
            if self.multi_gpu:
                torch.distributed.barrier()

            for train_file_idx in range(len(self.dataset_files)):
                train_docs = load_data(self.dataset_files[train_file_idx],
                tokenizer=self.tokenizer)
                train_data_source = JpLmSeqsDataset(self.config, self.tokenizer, train_docs, "train", randomize=True, params=self.params)
                train_data_sampler = self.sampler(train_data_source, replacement=False)
                train_data_loader = DataLoader(train_data_source, batch_size=self.config.batch_size,
                sampler=train_data_sampler, num_workers=0,
                collate_fn=train_data_source.batch_sequences, pin_memory=True)

                iter_bar = tqdm(train_data_loader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
                for batch in iter_bar:
                    if self.params.n_gpu > 0:
                        batch = tuple(t.to(f"cuda:{self.params.local_rank}") for t in batch)
                        # batch = tuple(torch.tensor(t).to(f"cuda:{self.params.local_rank}") for t in batch)
                        # batch: [[[seq], length], ...]

                    if self.mlm:
                        token_ids, attn_mask, lm_labels = self.prepare_batch_mlm(batch=batch)
                    else:
                        token_ids, attn_mask, lm_labels = self.prepare_batch_clm(batch=batch)
                    self.step(input_ids=token_ids, attention_mask=attn_mask, lm_labels=lm_labels)

                    iter_bar.update()
                    iter_bar.set_postfix(
                        {"Last_loss": f"{self.last_loss:.2f}", "Avg_cum_loss": f"{self.total_loss_epoch/self.n_iter:.2f}"}
                    )
                iter_bar.close()

            if self.is_master:
                logger.info(f"--- Ending epoch {self.epoch}/{self.params.n_epoch-1}")
            self.end_epoch()

        if self.is_master:
            logger.info("Save very last checkpoint as `pytorch_model.bin`.")
            self.save_checkpoint(checkpoint_name="pytorch_model.bin")
            logger.info("Training is finished")
