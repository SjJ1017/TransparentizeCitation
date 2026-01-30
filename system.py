import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os
import openai
import json
from tqdm import tqdm
from transformers import AutoTokenizer
import time
import argparse
import re
import torch
from nltk import sent_tokenize


def cut_passage(passage):

    # return first paragraph
    passage = passage.strip()
    passage = passage.split('\n')
    passage = passage[0]
    return passage


def load_model(model_name_or_path, dtype=torch.float16, int8=False, state_dict_path = None):
    # Load a huggingface model and tokenizer
    # dtype: torch.float16 or torch.bfloat16
    # int8: whether to use int8 quantization
    # reserve_memory: how much memory to reserve for the model on each gpu (in GB)

    # Load the FP16 model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Loading {model_name_or_path} in {dtype}...")
    if int8:
        logger.warn("Use LLM.int8")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map='auto',
        #torch_dtype=dtype,
        #max_memory=get_max_memory(),
        load_in_8bit=int8,
        token='your_token'
    )
    logger.info("Finish loading in %.2f sec." % (time.time() - start_time))

    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path, map_location='cpu')
        logger.info(f"Loading pre-trained weights from {state_dict_path}...")
        model.load_state_dict(state_dict['state'])
        logger.info("Loaded pre-trained weights")

    # Load the tokenizer


    # Fix OPT bos token problem in HF
    if "opt" in model_name_or_path:
        tokenizer.bos_token = "<s>"
    tokenizer.padding_side = "left"

    return model, tokenizer

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def load_jsonl(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        paris = [json.loads(line) for line in f]
        dict_pairs = {list(pair.keys())[0]: list(pair.values())[0] for pair in paris}
    return dict_pairs

def write_pair_to_jsonl(file_path, pair):
    with open(file_path, 'a') as f:
        f.write(json.dumps(pair) + '\n')

class LLM:

    def __init__(self, args, usecache=False, cache_dir=None):
        self.args = args
        self.usecache = usecache
        if usecache:
            self.cache_dir = cache_dir
            self.cached_qa_pairs = load_jsonl(self.cache_dir)
        else:
            self.cache_dir = None
            self.cached_qa_pairs = {}
        if args.openai_api:
            import openai
            OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
            OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID")
            OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")

            if args.azure:
                openai.api_key = OPENAI_API_KEY
                openai.api_base = OPENAI_API_BASE
                openai.api_type = 'azure'
                openai.api_version = '2023-05-15'
            else:
                openai.api_key = OPENAI_API_KEY
                openai.organization = OPENAI_ORG_ID

            self.tokenizer = AutoTokenizer.from_pretrained("gpt2",
                                                           fast_tokenizer=False)  # TODO: For ChatGPT we should use a different one
            # To keep track of how much the API costs
            self.prompt_tokens = 0
            self.completion_tokens = 0
        else:
            if hasattr(args, 'state_dict_path') and args.state_dict_path is not None:
                self.model, self.tokenizer = load_model(args.model, state_dict_path=args.state_dict_path)
            else:
                self.model, self.tokenizer = load_model(args.model)

        self.prompt_exceed_max_length = 0
        self.fewer_than_50 = 0
        self.azure_filter_fail = 0

    def generate(self, prompt, max_tokens, stop = None, system = 'You are a helpful assistant that finish the following task'):
        if prompt in self.cached_qa_pairs:
            return self.cached_qa_pairs[prompt]
        str_prompt = prompt
        args = self.args
        if max_tokens <= 0:
            self.prompt_exceed_max_length += 1
            logger.warning(
                "Prompt exceeds max length and return an empty string as answer. If this happens too many times, it is suggested to make the prompt shorter")
            return ""
        if max_tokens < 50:
            self.fewer_than_50 += 1
            #logger.warning(
            #    "The model can at most generate < 50 tokens. If this happens too many times, it is suggested to make the prompt shorter")

        if args.openai_api:
            #use_chat_api = ("turbo" in args.model and not args.azure) or ("gpt-4" in args.model and args.azure)
            use_chat_api = True

            if use_chat_api:
                # For chat API, we need to convert text prompts to chat prompts
                prompt = [
                    {'role': 'system',
                     'content': system},
                    {'role': 'user', 'content': prompt}
                ]
            if args.azure:
                deploy_name = args.model

            if use_chat_api:
                is_ok = False
                retry_count = 0
                while not is_ok:
                    retry_count += 1
                    try:
                        response = openai.ChatCompletion.create(
                            engine=deploy_name if args.azure else None,
                            model=args.model,
                            messages=prompt,
                            temperature=args.temperature,
                            max_tokens=max_tokens,
                            stop=stop,
                            top_p=args.top_p,
                        )
                        is_ok = True
                    except Exception as error:
                        if retry_count <= 5:
                            logger.warning(f"OpenAI API retry for {retry_count} times ({error})")
                            continue
                        print(error)
                self.prompt_tokens += response['usage']['prompt_tokens']
                self.completion_tokens += response['usage']['completion_tokens']
                if self.usecache:
                    write_pair_to_jsonl(self.cache_dir, {str_prompt: response['choices'][0]['message']['content']})
                return response['choices'][0]['message']['content']
            else:
                is_ok = False
                retry_count = 0
                while not is_ok:
                    retry_count += 1
                    try:
                        response = openai.Completion.create(
                            engine=deploy_name if args.azure else None,
                            model=args.model,
                            prompt=prompt,
                            temperature=args.temperature,
                            max_tokens=max_tokens,
                            top_p=args.top_p,
                            stop=["\n", "\n\n"] + (stop if stop is not None else [])
                        )
                        is_ok = True
                    except Exception as error:
                        if retry_count <= 5:
                            logger.warning(f"OpenAI API retry for {retry_count} times ({error})")
                            if "triggering Azure OpenAI’s content management policy" in str(error):
                                # filtered by Azure 
                                self.azure_filter_fail += 1
                                return ""
                            continue
                        print(error)
                self.prompt_tokens += response['usage']['prompt_tokens']
                self.completion_tokens += response['usage']['completion_tokens']
                if self.usecache:
                    write_pair_to_jsonl(self.cache_dir, {prompt: response['choices'][0]['text']})
                print(response['choices'][0]['text'])

                return response['choices'][0]['text']
        else:
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
            stop = [] if stop is None else stop
            stop = list(set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"]))  # In Llama \n is <0x0A>; In OPT \n is Ċ
            #stop_token_ids = list(set([self.tokenizer._convert_token_to_id(stop_token) for stop_token in stop] + [
                #self.model.config.eos_token_id]))
            outputs = self.model.generate(
                **inputs,
                temperature=args.temperature,
                max_new_tokens=max_tokens,
                eos_token_id=128001,
                do_sample=True,
            )
            generation = self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
            return generation


if __name__ == '__main__':
    text = 'I love you.'
    instruction = 'Translate the given text into Chinese: \n'
    llm_args = {
        'openai_api': True,
        'azure': False,
        'model': 'gpt-4o',
        'temperature': 0.5,
        'top_p': 0.9,
    }

    args = argparse.Namespace(**llm_args)
    llm = LLM(args=args)
    translation = llm.generate(instruction + text, max_tokens=100)
    print(translation)
        

