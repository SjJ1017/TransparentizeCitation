from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import random
import os

PIPELINE_OUTPUT = 'output'
PIPELINE_DOC_CACHE = 'doc_cache'

global autoais_model, autoais_tokenizer
autoais_model = None
autoais_tokenizer = None
get_docs_by_index = lambda i,docs: docs[i] if i < len(docs) else None 
ais_LLM = None

QA_MODEL = "gaotianyu1350/roberta-large-squad"

AUTOAIS_MODEL = os.environ.get('AUTOAIS_MODEL', None)
if not AUTOAIS_MODEL:
    AUTOAIS_MODEL = "google/t5_xxl_true_nli_mixture"



def load_auto_ais():
    global autoais_model, autoais_tokenizer
    print('Initializing eval model for citation precision and recall...') 
    autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)
    print('Done!')

def _run_nli_autoais(passage, claim, test = False):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    if not test:
        global autoais_model, autoais_tokenizer
        if not autoais_model:
            load_auto_ais()
        input_text = "premise: {} hypothesis: {}".format(passage, claim)
        input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
        with torch.inference_mode():
            outputs = autoais_model.generate(input_ids, max_new_tokens=10)
        result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
        inference = 1 if result == "1" else 0
        return inference
    else:
        res = random.choice([0,1])

    return res


