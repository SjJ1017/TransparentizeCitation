# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override
import re

# used in data processing
def get_tokens(input, tokenizer):
    text = tokenizer.decode(input)
    start = 0
    token_cache = []
    tokens = []
    token_span_lens = []
    for i in input:
        if not token_cache:
            token = tokenizer.decode(i)
        else:
            token = tokenizer.decode(token_cache + [i])
        len_token = len(token)
        text_token = text[start:start + len_token]
        if text_token[-1:] != token[-1:]:
            token_cache.append(i)
        else:
            token_span_lens.append(1 + len(token_cache))
            token_cache = []
            start += len_token
            tokens.append(token)
    return tokens, token_span_lens

def is_pattern(token_text, start, end, text, pattern):
    regex = re.compile(pattern)
    
    for match in regex.finditer(text):
        match_start, match_end = match.start(), match.end()
        if start >= match_start and end <= match_end:
            return True

    return False
# Helper functions to check patterns

def is_confidence(token_text, start, end, text):
    # (Confidence: 0.2) - (Confidence: 1.0)
    return is_pattern(token_text, start, end, text, r"\(Confidence: [01]\.\d+\)")

def is_citation(token_text, start, end, text):
    return is_pattern(token_text, start, end, text, r"\[\d+\]")

def get_weight_labels(input_ids, tokenizer, max_error_count=3):
    error_count = 0
    text = tokenizer.decode(input_ids, skip_special_tokens=False)
    start = 0
    offsets = []
    tokens, token_span_lens = get_tokens(input_ids, tokenizer=tokenizer)
    for i, (token, token_span_len) in enumerate(zip(tokens, token_span_lens)):
        len_token = len(token)
        offsets.append((start, start + len_token, token_span_len))
        if text[start:start + len_token][-1:] != token[-1:]:
            print(text[start:start + len_token], token, len_token, input_ids[i])
            error_count += 1
            if error_count > max_error_count:
                print(input_ids)
                raise ValueError("Error in tokenization")
        start += len_token
    # Define section ranges
    assistant_start = text.find("<|start_header_id|>assistant<|end_header_id|>")
    assistant_end = assistant_start + len("<|start_header_id|>assistant<|end_header_id|>")
    if assistant_start == -1:
        assistant_start = text.find("assistant")
        assistant_end = assistant_start + len("assistant")
    #find after assistant start:
    think_start = text.find("Think step by step:", assistant_start)
    references_start = text.find("References:",assistant_start)
    answer_start = text.find("Answer:", assistant_start)
    if think_start == -1:
        think_start = 100000
    if references_start == -1:
        references_start = 100000
    if answer_start == -1:
        answer_start = 100000
    # Initialize the label tensor
    labels = [0] * len(input_ids)
    # Assign labels based on token positions
    idx = 0
    for _ , (start, end, span_len) in enumerate(offsets):
        if start == end:
            continue
        token_text = text[start:end]
        for j in range(span_len):
            if think_start <= start < references_start:
                labels[idx] = 1
            elif assistant_end <= start < think_start:
                labels[idx] = -1
            elif references_start <= start < answer_start:
                if is_confidence(token_text, start, end, text):
                    labels[idx] = 4
                else:
                    labels[idx] = 2
            elif answer_start <= start:
                if is_citation(token_text, start, end, text):
                    labels[idx] = 5
                else:
                    labels[idx] = 3
            idx += 1
    return input_ids, labels


def weight_label2weight(labels, cot_weight, ref_weight, answer_weight, confidence_weight, citation_weight, punish_weight, dynamic = True):
    def _weight_mapping(labels):
        cot_count = (labels == 1).sum().item()
        ref_count = (labels == 2).sum().item()
        answer_count = (labels == 3).sum().item()
        confidence_count = (labels == 4).sum().item()
        citation_count = (labels == 5).sum().item()
        punish_count = (labels == -1).sum().item()
        answer_weight = 1.0
        cot_weight = 1.0

        # answer_weight * answer_count = citation_weight * citation_count
        citation_weight = answer_count * answer_weight / citation_count
        # ref_weight * ref_count = cot_weight * cot_count + answer_weight * answer_count
        ref_weight = (cot_count * cot_weight + answer_count * answer_weight) / ref_count
        # ref_weight * ref_count = confidence_weight * confidence_count
        confidence_weight = ref_count * ref_weight / confidence_count
        punish_count = punish_count

        return {'cot_weight': cot_weight, 'ref_weight': ref_weight, 'answer_weight': answer_weight, 'confidence_weight': confidence_weight, 'citation_weight': citation_weight, 'punish_weight': punish_weight}
    if not dynamic:
        weights = torch.zeros_like(labels, dtype=torch.float)  # 初始化权重为 0
        weights[labels == -100] = 0
        weights[labels == 0] = 0
        weights[labels == 1] = cot_weight
        weights[labels == 2] = ref_weight
        weights[labels == 3] = answer_weight
        weights[labels == 4] = confidence_weight
        weights[labels == 5] = citation_weight
        weights[labels == -1] = punish_weight
    else:
        mapping = _weight_mapping(labels)
        return weight_label2weight(labels, **mapping, dynamic = False)
    return weights


    
class WeightedSeq2SeqTrainer(Seq2SeqTrainer):
    remove_unused_columns = False

    def init_weights(self, **weights):
        print('Initializing weights:', weights.get("cot_weight", 1.0), weights.get("ref_weight", 1.0), weights.get("answer_weight", 1.0), weights.get("confidence_weight", 1.0), weights.get("citation_weight", 1.0))
        self.cot_weight = weights.get("cot_weight", 1.0)
        self.ref_weight = weights.get("ref_weight", 1.0)
        self.answer_weight = weights.get("answer_weight", 1.0)
        self.confidence_weight = weights.get("confidence_weight", 1.0)
        self.citation_weight = weights.get("citation_weight", 1.0)
        self.punish_weight = 50.0
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        weights = inputs.pop("weights") 

        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view_as(shift_labels) 
        if weights is not None:
            weights = weight_label2weight(shift_weights, self.cot_weight, self.ref_weight, self.answer_weight, self.confidence_weight, self.citation_weight, self.punish_weight)
        else:
            raise ValueError("Weight tensor not provided.")

        weighted_loss = loss * weights

        # print list loss
        active_loss = shift_labels != -100 
        normalized_weights = (weights * active_loss).sum(dim=1) 
        batch_loss = (weighted_loss * active_loss).sum(dim=1) / normalized_weights
        
        final_loss = batch_loss.mean()
        
        return (final_loss, outputs) if return_outputs else final_loss

