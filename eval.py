import random
import numpy as np
from utils import *
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
from nltk import sent_tokenize
import copy
import argparse
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default='DPO/datasets/refine_4o_res.json', help="input")
parser.add_argument("--output", type=str, default=None, help="output")
parser.add_argument("--output_type", type=str, default='dir', help="dir or file")
parser.add_argument("--dataset", type=str, default='DPO/crag_300_4o.json', help="dataset")
parser.add_argument("--start", type=int, default=0, help="start")
parser.add_argument("--total", type=int, default=5000, help="total number of data items")
parser.add_argument("--test", action='store_true', default=False, help="total number of data items")
parser.add_argument("--rich", action='store_true', help="[]{}")
parser.add_argument("--ex_docs", type=int, default=3, help="total number of data items")
parser.add_argument("--nonragfile", type=str, default='', help="input")
parser.add_argument("--full_answer", action='store_true', help="total number of data items")
parser.add_argument("--loose", action='store_true', help="1 correct in set answers => right")
parser.add_argument("--dpo", action='store_true')
parser.add_argument("--dpo_dataset",type=str, default='DPO/crag_dpo_test.json')
parser.add_argument("--adaptive_ex", action='store_true')
parser.add_argument("--full_document", action='store_true')
parser.add_argument("--process_vllm_infer", action='store_true')
parser.add_argument("--process_naive", action='store_true')
parser.add_argument("--process_RECITE", action='store_true')
parser.add_argument("--process_front", action='store_true')
parser.add_argument("--process_online", action='store_true')
parser.add_argument("--process_context", action='store_true')
parser.add_argument("--process_post", action='store_true')
parser.add_argument("--eval_dataset", action='store_true')
parser.add_argument("--absolute_model", type=str, default='google/t5_xxl_true_nli_mixture')
args = parser.parse_args()

def remove_quotes(text):

    text = text.strip()
    if text.endswith(','):
        text = text[:-1]
    return text.strip('\'"')

def _remove_confidence(ref):

    ref = re.sub(r' \(Confidence: \d+\.\d+\)', '', ref)
    return re.sub(r'\(Confidence: \d+\.\d+\)', '', ref)
def _remove_marks(ref):

    return re.sub(r'\[\d+\]', '', ref)
    
def extract_before_document_roman(text):

    pattern = r"(.*?)(Document [IVXLCDM]+)"
    match = re.search(pattern, text)
    
    if match:

        return match.group(1)
    else:

        return text

def extract_roman_numerals(text):
    # 匹配 Document + 罗马字母的模式
    pattern = r"Document ([IVXLCDM]+)"
    matches = re.findall(pattern, text)
    if not matches:

        return []
    return matches

def roman_to_int(s):
    roman_dict = {
        'I': 1,
        'V': 5,
        'X': 10,
    }
    res = 0
    for i in range(len(s)):
        if i > 0 and roman_dict[s[i]] > roman_dict[s[i-1]]:
            res += roman_dict[s[i]] - 2 * roman_dict[s[i-1]]
        else:
            res += roman_dict[s[i]]
    return res


def parse_output_vllm(output, datapoint = None):

    def _extract_sents(sents):
        # find last sentence with citation
        for i in range(len(sents)-1, -1, -1):
            if re.findall(r"\[\d+\]", sents[i]):
                return sents[:i+1]
        # find last abstain sentence
        for i in range(len(sents)-1, -1, -1):
            if is_response_abstained(sents[i]):
                return sents[:i+1]
        return sents[:1 if len(sents) > 1 else 0]
    def _remove_confidence(ref):
        # remove pattern like (Confidence: 1.0)
        ref = re.sub(r' \(Confidence: \d+\.\d+\)', '', ref)
        return re.sub(r'\(Confidence: \d+\.\d+\)', '', ref)
    def _remove_marks(ref):
        # remove pattern like [1]
        return re.sub(r'\[\d+\]', '', ref)
    
    pars = output.split('\n\n', 2)
    ex_idx = 0
    if len(pars) == 3:
        cot, refs, answer = pars
        if refs.startswith('References:\n'):
            refs = refs.replace('References:\n', '')
        else:
            if 'References:' in output:
                after_ref = output.split('References:')[1].strip()
                if 'Answer:' in after_ref:
                    refs, answer = after_ref.split('Answer:',1)
                    refs = refs.strip()
                    answer = answer.strip()
                else:
                    return [], output, 0
            else:
                return [], output, 0
        ref_list = refs.split('\n')
        ex_idx = len(ref_list)
        for i, ref in enumerate(ref_list):
            pattern = r"Document ([IVXLCDM]+)"
            if not re.findall(pattern, ref):
                ex_idx = i
                break
        if not datapoint or not args.full_document:
            ref_list = [remove_quotes(_remove_marks(_remove_confidence(extract_before_document_roman(ref)))) for ref in ref_list]
            ref_list = [remove_quotes(_remove_marks(_remove_confidence(ref)).split(' Internal Knowledge')[0]) for ref in ref_list]
        else:
            for k in range(ex_idx):
                print((_remove_confidence(ref_list[k])), ex_idx, k)
                romans = extract_roman_numerals(ref_list[k])
                if not romans:
                    continue
                roman = romans[0]
                number = roman_to_int(roman)
                ref_list[k] = datapoint[number - 1]['text']
            for k in range(ex_idx, len(ref_list)):
                if ', Internal Knowledge' in ref_list[k]:
                    ref_list[k] = remove_quotes(_remove_marks(_remove_confidence(ref_list[k])).split(' Internal Knowledge')[0])
                elif 'Internal Knowledge: ' in ref_list[k]:
                    ref_list[k] = remove_quotes(_remove_marks(_remove_confidence(ref_list[k])).split('Internal Knowledge: ', 1)[1])
                elif 'Internal Knowledge' in ref_list[k]:
                    ref_list[k] = remove_quotes(_remove_marks(_remove_confidence(ref)).split(' Internal Knowledge')[0])
                else:
                    ref_list[k] = remove_quotes(_remove_marks(_remove_confidence(ref_list[k])))


        answer = _remove_confidence(answer.replace('Final Answer: ', ''))
        answer = answer.replace('Answer: ', '')
        answer_sents = sent_tokenize(answer)[:3]
        answer_sents = _extract_sents(answer_sents)
        answer = ' '.join(answer_sents)
        return ref_list, answer, ex_idx
    else:
        if len(pars) == 2:
            cot, answer = pars
            answer = _remove_confidence(answer.replace('Final Answer: ', ''))
            answer = _remove_confidence(answer.replace('Answer: ', ''))
            answer_sents = sent_tokenize(answer)[:3]
            answer_sents = _extract_sents(answer_sents)
            answer = ' '.join(answer_sents)
            return [], answer, 0
        elif len(pars) == 1:
            return [], output, 0


def pre_process_vllm_infer(data, dataset, eval_dataset = args.eval_dataset):
    if eval_dataset:
        dataset = data
    return_data = []
    if isinstance(data, str):
        data = load_json(data)
    else:
        assert isinstance(data, list)
    if isinstance(dataset, str):
        dataset = load_json(dataset)
    else:
        assert isinstance(dataset, list)
    assert len(data) == len(dataset)
    if eval_dataset:
        data = list(filter(lambda x: not x['skip'], data))
        dataset = list(filter(lambda x: not x['skip'], dataset))
    count = -1
    for item, point in zip(data, dataset):
        count += 1
        new_item = {}
        new_item['data'] = {}
        new_item['data']['question'] = point['question']

        new_item['data']['answer'] = point['answer']
        if not eval_dataset:
            refs, answer, ex_idx = parse_output_vllm(item['predict'], point['docs'])
        else:
            if isinstance(item['oracle'], str):
                refs, answer, ex_idx = parse_output_vllm(item['oracle'])
            else:
                # multiple outputs
                num = len(item['oracle'])
                new_items = [copy.deepcopy(new_item) for _ in range(num)]
                for i, it in enumerate(new_items):
                    refs, answer, ex_idx = parse_output_vllm(item['oracle'][i])
                    it['data']['docs'] = refs
                    it['doc_cache'] = refs
                    it['output'] = answer
                    it['max_ex_idx'] = ex_idx
                    it['item_id'] = count
                    it['version'] = i
                    return_data.append(it)
                continue

        new_item['data']['docs'] = refs
        new_item['doc_cache'] = refs
        new_item['output'] = answer
        new_item['max_ex_idx'] = ex_idx
        
        return_data.append(new_item)
    if len(return_data) != len(dataset):
        print(len(return_data), len(dataset))
        # multiple outputs
        new_dataset = []
        for i, it in enumerate(return_data):
            item_id = it['item_id']
            point = dataset[item_id]
            num = len(item['oracle'])
            new_dataset.append(copy.deepcopy(point))
        dataset = new_dataset
    return return_data, dataset

def pre_process_naive(data, dataset):
    def _remove_confidence(ref):
        # remove pattern like (Confidence: 1.0)
        ref = re.sub(r' \(Confidence: \d+\.\d+\)', '', ref)
        return re.sub(r'\(Confidence: \d+\.\d+\)', '', ref)
    def _remove_marks(ref):
        # remove pattern like [1]
        return re.sub(r'\[\d+\]', '', ref)
    return_data = []
    if isinstance(data, str):
        data = load_json(data)
    else:
        assert isinstance(data, list)
    if isinstance(dataset, str):
        dataset = load_json(dataset)
    else:
        assert isinstance(dataset, list)
    assert len(data) == len(dataset)
    for item, point in zip(data, dataset):
        new_item = {}
        new_item['data'] = {}
        new_item['data']['question'] = point['question']

        new_item['data']['answer'] = point['answer']
        output = item['predict']
        pars = output.split('\n\n', 1)
        if len(pars) == 1:
            output = pars[0]
            in_docs = []
        else:
            output, refs = pars
            refs = refs.split('\n')
            in_docs = [remove_quotes(_remove_marks(_remove_confidence(ref))) for ref in refs]
        output = output.replace('Answer:', '').strip()
        out_put_first_three = sent_tokenize(output)[:3]
        output = ' '.join(out_put_first_three)
        new_item['data']['docs'] = [doc['text'] for doc in point['docs']] + in_docs
        new_item['doc_cache'] = [doc['text'] for doc in point['docs']] + in_docs
        new_item['output'] = output
        new_item['max_ex_idx'] = len(point['docs'])
        
        return_data.append(new_item)
    return  return_data, dataset

def pre_front(data, dataset):
    return_data = []
    if isinstance(data, str):
        data = load_json(data)
    else:
        assert isinstance(data, list)
    if isinstance(dataset, str):
        dataset = load_json(dataset)
    else:
        assert isinstance(dataset, list)
    assert len(data) == len(dataset)
    for item, point in zip(data, dataset):
        new_item = {}
        new_item['data'] = {}
        new_item['data']['question'] = point['question']

        new_item['data']['answer'] = point['answer']
        output = item['predict']
        if '<|eot_id|>' in output:
            output = output.split('<|eot_id|>')[0]
        if '[ANSWER]' in output:
            output = output.split('[ANSWER]')[1]
            if '[GROUNDING]' in output:
                output = output.split('[GROUNDING]')[0]
        first_three = sent_tokenize(output)[:3]
        output = ' '.join(first_three)
        new_item['data']['docs'] = [doc['text'] for doc in point['docs']]
        new_item['doc_cache'] = [doc['text'] for doc in point['docs']]
        new_item['output'] = output
        new_item['max_ex_idx'] = len(point['docs'])
        
        return_data.append(new_item)
    return return_data, dataset

def pre_online(data, dataset):
    """"""

    def _custom_sent_tokenize(text):

        footnotes = re.findall(r'\\footnote\{(.*?)\}', text)
        

        text_without_footnotes = re.sub(r'\\footnote\{.*?\}', ' FOOTNOTE_PLACEHOLDER ', text)
        print(text_without_footnotes)
        sentences = sent_tokenize(text_without_footnotes)

        result = []
        footnote_idx = 0  
        print(sentences)
        for sentence in sentences:
    
            parts = sentence.split('FOOTNOTE_PLACEHOLDER')
            

            if len(parts) > 1:
                for i in range(len(parts) - 1):
                    result.append(parts[i] + r'\footnote{' + footnotes[footnote_idx] + r'}')
                    footnote_idx += 1
                result.append(parts[-1])  
            else:
                result.append(parts[0])

        combined_results = []
        for results in result:
            if results.strip().startswith('\\footnote{') and combined_results:
                combined_results[-1] += results
            else:
                combined_results.append(results)
        combined_results = [x.strip() for x in combined_results]
        combined_results  = list(filter(lambda x:x!='', combined_results))

        new_combined = []
        for sent in combined_results:
            if len(sent) < 4 and new_combined:
                new_combined[-1] += sent
            else:
                new_combined.append(sent)
        return new_combined
        
        return result
    return_data = []
    if isinstance(data, str):
        data = load_json(data)
    else:
        assert isinstance(data, list)
    if isinstance(dataset, str):
        dataset = load_json(dataset)
    else:
        assert isinstance(dataset, list)
    assert len(data) == len(dataset)
    for item, point in zip(data, dataset):
        new_item = {}
        new_item['data'] = {}
        new_item['data']['question'] = point['question']

        new_item['data']['answer'] = point['answer']
        output = item['predict']
        if '<|eot_id|>' in output:
            output = output.split('<|eot_id|>')[0]

        first_three = _custom_sent_tokenize(output)[:3]
                
        notes = []
        for i, sent in enumerate(first_three):
            if '\\footnote' in sent:
                footnote = re.findall(r'\\footnote{(.*?)}', sent)
                if footnote:
                    first_three[i] = sent.split('\\footnote')[0] + '.'
                    notes.append(footnote[0])
                else:
                    first_three[i] = sent.split('\\footnote')[0] + '.'
                    notes.append('')
            else:
                first_three[i] = re.sub(r'\\footnote{(.*?)}', '', sent)
                notes.append('')
        #print(notes)
        ex_notets = []
        in_notes = []
        ex_cites = []
        in_cites = []
        for i, note in enumerate(notes):
            # If '[x]: ' in note, it is external note
            if re.findall(r'\[\d+\]: ', note):
                note = _remove_marks(_remove_confidence(note))
                ex_notets.append(note)
                ex_cites.append(i + 1)
            else:
                in_notes.append(_remove_confidence(_remove_marks(note)))
                in_cites.append(i + 1)
        cites = ex_cites + in_cites

        for i, sent in enumerate(first_three):
            cite = cites.index(i + 1) + 1
            if notes[i] != '':
                first_three[i] = add_citation_after(sent, cite)

        output = ' '.join(first_three)
        docs = ex_notets + in_notes
        docs = list(map(lambda x: x[2:] if x.startswith(': ') else x, docs))
        new_item['data']['docs'] = docs
        new_item['doc_cache'] = docs
        new_item['output'] = output
        new_item['max_ex_idx'] = len(ex_notets)
        
        return_data.append(new_item)
    return return_data, dataset

def pre_context(data, dataset):
    return_data = []
    if isinstance(data, str):
        data = load_json(data)
    else:
        assert isinstance(data, list)
    if isinstance(dataset, str):
        dataset = load_json(dataset)
    else:
        assert isinstance(dataset, list)
    assert len(data) == len(dataset)
    for item, point in zip(data, dataset):
        new_item = {}
        new_item['data'] = {}
        new_item['data']['question'] = point['question']

        new_item['data']['answer'] = point['answer']
        sents = item['results']
        new_item['data']['docs'] = [snet['citation'] for snet in sents]
        new_item['doc_cache'] = [snet['citation'] for snet in sents]
        new_item['output'] = ' '.join([add_citation_after(snet['sentence'].replace('<|eot_id|>', ''), i+1) for i, snet in enumerate(sents)])
        new_item['max_ex_idx'] = len(new_item['data']['docs'])
        
        return_data.append(new_item)

    return return_data, dataset

def pre_post(data, dataset):
    return_data = []
    if isinstance(data, str):
        data = load_json(data)
    else:
        assert isinstance(data, list)
    if isinstance(dataset, str):
        dataset = load_json(dataset)
    else:
        assert isinstance(dataset, list)
    print(len(data), len(dataset))
    assert len(data) == len(dataset)
    for item, point in zip(data, dataset):
        new_item = {}
        new_item['data'] = {}
        new_item['data']['question'] = point['question']

        new_item['data']['answer'] = point['answer']
        new_item['data']['docs'] = item['docs']
        new_item['doc_cache'] = item['docs']
        new_item['output'] = item['output']
        new_item['max_ex_idx'] = item['ex_idx']
        return_data.append(new_item)
    return return_data, dataset

def add_citation_after(sentence, citation):
    pattern = r'([.!?])$'  
    match = re.search(pattern, sentence)  
      
    if match:   
        new_sentence = sentence[:match.start()] + f' [{citation}]' + match.group(1)  
    else:   
        new_sentence = sentence + f' [{citation}]'
      
    return new_sentence

def pre_process_RECITE(data, dataset):
    return_data = []
    if isinstance(data, str):
        data = load_json(data)
    else:
        assert isinstance(data, list)
    if isinstance(dataset, str):
        dataset = load_json(dataset)
    else:
        assert isinstance(dataset, list)
    print(len(data), len(dataset))
    assert len(data) == len(dataset)
    for item, point in zip(data, dataset):
        new_item = {}
        new_item['data'] = {}
        new_item['data']['question'] = point['question']

        new_item['data']['answer'] = point['answer']
        psgs = point['inner_passages']
        ans, cite = item['outputs'][0]
        sents = sent_tokenize(ans)
        new_item['data']['docs'] = [doc['text'] for doc in psgs]
        new_item['doc_cache'] = [doc['text'] for doc in psgs]
        new_item['output'] = ' '.join(add_citation_after(sent, cite + 1) for sent in sents)
        new_item['max_ex_idx'] = len(psgs)
        return_data.append(new_item)
    return return_data, dataset

invalid_ppl_mentions = [
    "I could not find any information",
    "The search results do not provide",
    "There is no information",
    "There are no search results",
    "there are no provided search results",
    "not provided in the search results",
    "is not mentioned in the provided search results",
    "There seems to be a mistake in the question",
    "Not sources found",
    "No sources found",
    "Try a more general question",
    "there is no relevant information",
    "do not contain information",
]

def is_invalid_ppl(text):
    return np.any([text.lower().startswith(mention.lower()) for mention in invalid_ppl_mentions])

def is_invalid_paragraph_ppl(text):
    return len(text.strip())==0 or np.any([mention.lower() in text.lower() for mention in invalid_ppl_mentions])

def perplexity_ai_abstain_detect(generation):
    output = generation
    if is_invalid_ppl(output):
        return True
    valid_paras = []
    for para in output.split("\n\n"):
        if is_invalid_paragraph_ppl(para):
            break
        valid_paras.append(para.strip())

    if len(valid_paras) == 0:
        return True
    else:
        return False

def generic_abstain_detect(generation):
    return generation.startswith("I'm sorry") or "provide more" in generation

def is_response_abstained(generation):
    return perplexity_ai_abstain_detect(generation) or generic_abstain_detect(generation)

def load_json(file):
    if file.endswith('.json'):
        with open(file, encoding='utf-8') as f:
            d = json.load(f)
    elif file.endswith('.jsonl'):
        with open(file, encoding='utf-8') as f:
            data = f.read()
        d = json.loads('[' + data.replace('}\n{', '},{') + ']')
    return d

def split_answers(data, dataset):
    for item, dp in zip(data,dataset):
        question = item['question']
        answer = item['response']
        paras = answer.split('\n\n')
        if len(paras) == 1:
            output = paras[0]
            refs = None
        elif len(paras) >= 2:
            output, refs = answer.split('\n\n')[0], answer.split('\n\n')[1]
        else:
            output, refs = answer, None
        if refs:
            refs = refs.split('\n')
        else:
            refs = []
        for i, ref in enumerate(refs):
            if ref.startswith('['):
                refs[i] = ref[len('[4] '):]
        item['doc_cache'] = [doc['text'] for doc in dp['docs']] + refs
        item['output'] = output
        item['data'] = {}


def write(file_name, obj):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=4)


def answerable(model):
    def _answerable(item):
        key = f'{model} answerable'
        return item[key]
    return _answerable

def postprocessing(item):
    return item['question_type'] == 'post-processing'

PIPELINE_OUTPUT = 'output'
PIPELINE_DOC_CACHE = 'doc_cache'

global autoais_model, autoais_tokenizer
autoais_model = None
autoais_tokenizer = None
get_docs_by_index = lambda i,docs: docs[i] if i < len(docs) else None 
ais_LLM = None

QA_MODEL = "gaotianyu1350/roberta-large-squad"
AUTOAIS_MODEL = "google/t5_xxl_true_nli_mixture"
AUTOAIS_MODEL_ABSOLUTE = args.absolute_model

import os

def get_json_files(path):
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        return [os.path.join(path, f) for f in os.listdir(path)]
    else:
        return []


def load_auto_ais():
    global autoais_model, autoais_tokenizer
    print('Initializing eval model for citation precision and recall...') 
    try:
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)
        
    except:
        print('Unable to load model from hub, trying to load from local path...')
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL_ABSOLUTE, torch_dtype=torch.bfloat16, device_map="auto")
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL_ABSOLUTE, use_fast=False)
    print('Done!')

def _run_nli_autoais(passage, claim, test = args.test):
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
def process_citation(string, _float = False):
    import re
    pattern = r'\[(\d*?)\]\{(.*?)\}'  
    matches = re.findall(pattern, string)  
    
    refs = []
    confs = []
    for match in matches:
        x, y = match
        try:
            x = int(x)
        except ValueError:
            x = 0
        
        try:
            if _float:
                y = float(y)
            else:
                y = int(y)
        except ValueError:
            y = 0
        
        refs.append(x)
        confs.append(y)
    
    processed_string = re.sub(pattern, '', string)
    return processed_string, refs, confs


def find_confidence(input_text):

    confidence_pattern = r'\(Confidence:\s*(\d+\.\d+)\)'
    
    confidence_match = re.search(confidence_pattern, input_text)
    confidence = float(confidence_match.group(1)) if confidence_match else None
    return confidence


def find_confidence(input_text):

    confidence_pattern = r'\(Confidence:\s*(\d+\.\d+)\)'
    
    confidence_match = re.search(confidence_pattern, input_text)
    confidence = float(confidence_match.group(1)) if confidence_match else None
    return confidence


def cite_eval_sentence(sent, docs, ex_idx, in_idx, at_most_citations = 3, entail_function = _run_nli_autoais, citation_conf = False):

    sent_metrics = {
        'no_cite': False,
        'overcite': 0,
        'cite_count': 0,
        'entail': 0,
        'precision_count': 0,
        'illegal_cite': False,
        'multiple_cite': False,
        'external_cite': False,
        'mix_cite': False,
        'internal_cite': False,
        'external_golden': False,
        'internal_golden': False,
        'abstain': False,
        'confidence': -1,
    }
    target_sent = remove_citations(sent).strip() # Citation removed and (if opted for) decontextualized
    joint_entail = -1  # Undecided
    
    if is_response_abstained(sent) or _run_nli_autoais(sent, 'I abstain from answering due to any reason.'):

        sent_metrics['abstain'] = True
        sent_metrics['no_cite'] = True
        return sent_metrics
    
    # Find references
    #ref = [int(r[1:]) - 1 for r in re.findall(r"\[\d+", sent)]  # In text citation id starts from 1
    #matches = re.findall(r"\\cite\[(.*?)\]\{([^}]+)\}", sent)
    if not citation_conf:
        matches = re.findall(r"\[(\d+(?:,\s*\d+)*)\]", sent)
        ref = [num for match in matches for num in match.replace(' ', '').split(',')]
        ref = [int(num) - 1 if num in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'] else -1 for num in ref]
        ref = list(filter(lambda x: x != -1, ref))
        confs = [1.0] * len(ref)
    else:
        target_sent, ref, confs = process_citation(sent, _float = True)
        ref = [x-1 for x in ref]
    #cite_confidences = [num for match in matches for num in match[0].replace(' ', '').split(',')]
    #cite_confidences = [int(num) if num in ['1','2','3','4','5'] else 0 for num in cite_confidences]
    #if cite_confidences:
    #    sent_metrics['confidence'] = cite_confidences[0]
    if len(ref) == 0:
        # No citations
        sent_metrics['no_cite'] = True
        joint_entail = 0
    elif any([ref_id >= len(docs) for ref_id in ref]):
        # Citations out of range
        sent_metrics['illegal_cite'] = True
        joint_entail = 0
    else:
        if at_most_citations is not None:
            ref = ref[:at_most_citations]
        total_citations = len(ref)
        sent_metrics['cite_count'] = total_citations
        joint_passage = '\n'.join([(docs[psgs_id]) for psgs_id in ref])
        conf_list = [find_confidence((docs[psgs_id]))*cite_conf for psgs_id, cite_conf in zip(ref, confs) if find_confidence((docs[psgs_id])) is not None]
        sent_metrics['confidence'] = np.mean(conf_list) if conf_list else -1
        ref_source = 1 # 1 is external
        if any([psg_id + 1 in in_idx for psg_id in ref]):
            if any([psg_id + 1 in ex_idx for psg_id in ref]):
                sent_metrics['mix_cite'] = True
                ref_source = 0
            else:
                sent_metrics['internal_cite'] = True 
        else:
            sent_metrics['external_cite'] = True

    # If not directly rejected by citation format error, calculate the recall score
    if joint_entail == -1:
        full_ex_passage =  '\n'.join([(docs[psgs_id - 1]) for psgs_id in ex_idx]) 
        full_entail = entail_function(full_ex_passage, target_sent) # full_entail == 1: cite external, otherwise cite internal
        if full_entail:
            sent_metrics['external_golden'] = True
        else:
            sent_metrics['internal_golden'] = True
        joint_entail = entail_function(joint_passage, target_sent)

    sent_metrics['entail'] = bool(joint_entail)
    if len(ref) > 1:
        sent_metrics['multiple_cite'] = True

    # calculate the precision score if applicable
    if joint_entail and len(ref) > 1:
        # Precision check: did the model cite any unnecessary documents?
        for psgs_id in ref:
            # condition A
            passage = docs[psgs_id]
            nli_result = entail_function(passage, target_sent)

            # condition B
            if not nli_result:
                subset_exclude = copy.deepcopy(ref)
                subset_exclude.remove(psgs_id)
                passage = '\n'.join([docs[pid] for pid in subset_exclude])
                nli_result = entail_function(passage, target_sent)
                if nli_result:  # psgs_id is not necessary
                    sent_metrics['overcite'] += 1
                else:
                    sent_metrics['precision_count'] += 1
            else:
               sent_metrics['precision_count'] += 1
    else:
        sent_metrics['precision_count'] += joint_entail
    
    return sent_metrics

def compute_autoais(data,
                    decontext=False,
                    concat=False,
                    qampari=False,
                    citation_conf = False,
                    at_most_sents = 3,
                    at_most_citations=3,
                    entail_function = _run_nli_autoais,
                    ex_idx = [1,2],
                    in_idx = [3,4],
                    adaptive_ex = args.adaptive_ex):

    global autoais_model, autoais_tokenizer


    ais_scores = []
    ais_scores_prec = []

    for item in tqdm(data):
        item['evals'] = []
        if adaptive_ex:
            max_ex_idx = item.get('ex_idx', 0)
            ex_idx = [i+1 for i in range(max_ex_idx)]
            in_idx = [i+1 for i in range(max_ex_idx, 20)]
        if qampari:
            print('now qampari...')
            sents = [item['question'] + " " + x.strip() for x in
                     item['output'].rstrip().rstrip(".").rstrip(",").split(",")]
        else:
            if isinstance(item['output'], list):
                item['output'] = ' '.join(item['output'])
            sents = sent_tokenize(item['output'])[:at_most_sents]
        if len(sents) == 0:
            ais_scores.append(0.0)
            ais_scores_prec.append(0.0)  # len(sents))
            continue

        for sent in sents:
            res = cite_eval_sentence(sent, docs=item['_docs_'], ex_idx=ex_idx, in_idx=in_idx, citation_conf=citation_conf)
            item['evals'].append({'sentence': sent, **res})
    

    return data

def normalize_answer(s):
    if isinstance(s, int):
        s = f'{s}'
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def lower(text):
        return text.lower()

    return remove_citations(white_space_fix(remove_articles(lower(s))))

def compute_exact(a_gold, a_pred):
    """Check whether two strings are equal up to normalization."""

    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """

    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True

    return False

def generate_statement(template, answer):
    if template:
        return template.replace('[ANSWER]', str(answer)).capitalize()
    else:
        return answer

def presence(template, short_answers, context):
    n_context = normalize_answer(context)
    if isinstance(short_answers, list):
        n_short_answers = [normalize_answer(sa) for sa in short_answers]
        for ans in n_short_answers:
            if _run_nli_autoais(n_context, generate_statement(template, ans)):
                return True
    else:
        return  _run_nli_autoais(n_context, generate_statement(template, short_answers))

    return False
def part_sentences(output_sents, ex_idx):
    ex_sents = []
    in_sents = []
    for sent in output_sents:
        matches = re.findall(r"\[(\d+(?:,\s*\d+)*)\]", sent)
        refs = [int(num) for match in matches for num in match.replace(' ', '').split(',')]
        if any(ref in ex_idx for ref in refs):
            ex_sents.append(sent)
        else:
            in_sents.append(sent)
    return ex_sents, in_sents

def part_docs(docs, ex_idx):
    ex_docs = []
    in_docs = []
    for i, doc in enumerate(docs):
        if i+1 in ex_idx:
            ex_docs.append(doc)
        else:
            in_docs.append(doc)
    return ex_docs, in_docs

def compute_str_em(template, answer, response):

    loc_acc = []
    if isinstance(answer, list):
        for qa_pair in answer:
            loc_acc.append(exact_presence(qa_pair['short_answers'], response))
        if not args.loose:
            return np.mean(loc_acc)
        else:
            return int(np.mean(loc_acc) > 0.0)
    else:
        return int(normalize_answer(answer) in normalize_answer(response))

def compute_entail(template, answer, response):

    loc_acc = []
    if isinstance(answer, list):
        for qa_pair in answer:
            loc_acc.append(presence(template, qa_pair['short_answers'], response))
    else:
        return(int(presence(template,answer,response)))
    return np.mean(loc_acc)

def acc_eval(rag_data, nonrag_data, dataset, ex_idx, eval_func = compute_entail, adaptive_ex = False):
    if nonrag_data: 
        ll = len(nonrag_data)
    else: ll = 5000
    min_length = min(len(rag_data),ll )
    print('total:', min_length)
    for i in tqdm(range(min_length)):
        if adaptive_ex:
            max_ex_idx = dataset[i].get('max_ex_idx', 0)
            ex_idx = [i+1 for i in range(max_ex_idx)]
            # in_idx = [i+1 for i in range(max_ex_idx, 20)]
        rag_item = rag_data[i]
        if nonrag_data:
            nonrag_item = nonrag_data[i]
        else:
            nonrag_item = {'data':dataset[i]}
        if 'qa_pairs' in nonrag_item['data'].keys():
            answer = nonrag_item['data']['qa_pairs']
        else:
            answer = remove_citations(nonrag_item['data']['answer'])
            #answer = ''
        if not args.full_answer and 'answer_template' in nonrag_item['data'].keys():
            template = nonrag_item['data']['answer_template']
            if '[ANSWER]' not in template:
                template = '[ANSWER]'
        else:
            template = None


        output_sentences =[single_eval['sentence'] for single_eval in rag_item['evals']]
        ex_sents, in_sents = part_sentences(output_sentences, ex_idx)



        rag_response = remove_citations(' '.join(rag_item['output']) if isinstance(rag_item['output'], list) else rag_item['output'])
        if nonrag_data:
            nonrag_response = remove_citations(' '.join(nonrag_item['output']) if isinstance(nonrag_item['output'], list) else nonrag_item['output'])
            rag_item['direct_output'] = nonrag_response
        rag_em = eval_func(template, answer, rag_response)
        if nonrag_data:
            nonrag_em = eval_func(template, answer, nonrag_response)
        else:
            nonrag_em = 'N/A'
        ex_em = eval_func(template, answer, ' '.join(ex_sents))
        in_em = eval_func(template, answer, ' '.join(in_sents))



        
        rag_item['accuracy'] = {
            'ex': ex_em,
            'in': in_em,
            'rag': rag_em,
            'non-rag': nonrag_em,

        }
        for k in dataset[i].keys():
            if k not in rag_item.keys():
                rag_item[k] = dataset[i][k]

        #rag_item['question'] = nonrag_item['data']['question']
        rag_item['answer'] = answer
        



def eval_file(file_name, dataset, output, pre_processing = False, adaptive_ex = False):
    rich = True if 'detect' in file_name or 'post' in file_name else False
    if isinstance(file_name, str):
        data = load_json(file_name)
    else:
        assert isinstance(file_name, list)
        data = file_name
    if isinstance(dataset, str):
        dataset = load_json(dataset)
    else:
        assert isinstance(dataset, list)
    if pre_processing:
        split_answers(data, dataset)

    processed_data = [{'_docs_':item['doc_cache'], 'output': item['output'], 'ex_idx': item['max_ex_idx'] if adaptive_ex else args.ex_idx ,**item['data']} for item in data]

    r = compute_autoais(processed_data[args.start:args.total],ex_idx=[i+1 for i in range(args.ex_docs)], in_idx= [i+1 for i in range(args.ex_docs, 20)], citation_conf = rich, adaptive_ex = args.adaptive_ex)


    with open(output, 'w', encoding='utf-8') as file:
        json.dump(r, file, indent=4)

    if args.nonragfile:
        with open(args.nonragfile, 'r', encoding='utf-8') as file:
            nonrag_data = json.load(file)[args.start:args.total]
    else: 
        nonrag_data = None


    acc_eval(rag_data=r, nonrag_data=nonrag_data, dataset=dataset, ex_idx= [i+1 for i in range(args.ex_docs)], adaptive_ex=args.adaptive_ex)

    with open(output, 'w', encoding='utf-8') as file:
        json.dump(r, file, indent=4)

file_name = get_json_files(args.file)
short_names = [f.split('/')[-1] for f in file_name]
print(file_name)
print(short_names)
dataset = args.dataset

if args.output:
    if args.output.endswith('/'):
        if args.output_type == 'dir':
            output = [f"{args.output}{file[:-6]}_acc{file[-6:]}" for file in short_names]
        else:
            assert len(short_names) == 1, 'output type is file, but multiple files are given'
            output = [args.output]

    else:
        if args.output_type == 'dir':
            output = [f"{args.output}/{file[:-6]}_acc{file[-6:]}" for file in short_names]
        else:
            assert len(short_names) == 1, 'output type is file, but multiple files are given'
            output = [args.output]
else:
    output = [f"{file[:-6]}_acc{file[-6:]}" for file in file_name]

for i,o in zip(file_name, output):
    print(f'Evaluating {i}...')
    if 'no_' in i:
        dataset = dataset.replace('_gold', '_no_gold')
    if args.dpo:
        dataset = args.dpo_dataset
    try:
        if args.process_vllm_infer:
            i, dataset = pre_process_vllm_infer(data = i, dataset = dataset)
        elif args.process_front:
            i, dataset = pre_front(data = i, dataset = dataset)
        elif args.process_naive:
            i, dataset = pre_process_naive(data = i, dataset = dataset)
        elif args.process_RECITE:
            i, dataset = pre_process_RECITE(data = i, dataset = dataset)
        elif args.process_online:
            i, dataset = pre_online(data = i, dataset = dataset)
        elif args.process_context:
            i, dataset = pre_context(data = i, dataset = dataset)
        elif args.process_post:
            i, dataset = pre_post(data = i, dataset = dataset)
        eval_file(file_name = i, dataset = dataset, output = o, pre_processing = args.dpo, adaptive_ex=args.adaptive_ex)
    except Exception as e:
        
        print(f"failed, {e}")
        traceback.print_exc()
        continue
    dataset = args.dataset

