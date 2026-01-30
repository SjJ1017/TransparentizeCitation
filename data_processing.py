import json
import re

i = 0
def load_json(file_path):
    if isinstance(file_path, list):
        return file_path
    if file_path.endswith('.jsonl'):
        return [json.loads(line) for line in open(file_path, 'r')]
    return json.load(open(file_path, 'r'))

def adjust_citation(refs, answer):
    refs = refs.split('\n')
    for i, ref in enumerate(refs):
        # find ref number [x] at the first
        ref_number = re.search(r'\[\d+\]', ref)
        if not ref_number:
            if 'no relevant information' not in answer:
                print(refs, answer)
                return 'EMPTY', "I don't have sufficient knowledge to answer the question, and there is no relevant information in the provided documents to answer the question."
            return 'EMPTY', "I don't have sufficient knowledge to answer the question, and there is no relevant information in the provided documents to answer the question."
        ref_number = ref_number.group()
        ref = ref.replace(f"{ref_number}", f"[{i+1}]")
        #assert f"{ref_number}" in answer, f"ref number {ref_number} not found in answer: {answer}, refs: {refs}"
        answer = answer.replace(f"{ref_number}", f"[{i+1}]")
        refs[i] = ref
    return '\n'.join(refs), answer

def check_oracle(oracle):
    if oracle.startswith('Think step by step:'):
        oracle = oracle.replace('Think step by step:', '').strip()
    if '\"my knowledge\"' in oracle:
        oracle = oracle.replace('\"my knowledge\"', 'my knowledge')
    if '\"My knowledge\"' in oracle:
        oracle = oracle.replace('\"My knowledge\"', 'My knowledge')
    if 'References:' not in oracle:
        #print(oracle)
        return False
    if 'Answer: ' not in oracle:
        #print(oracle)
        return False
    cot, ref_ans = oracle.split('References:', 1)
    ref, ans = ref_ans.split('Answer: ', 1)
    cot = cot.strip()
    #replace all [x] in cot with empty string
    cot = re.sub(r'\[\d+\]', '', cot)
    ref = ref.strip()
    ans = ans.strip()
    cot = cot.replace('\n\n', '\n')
    ref, ans = adjust_citation(ref, ans)
    cot = "Think step by step:\n" + cot
    return cot, ref, ans

def fill_input_direct(item, with_answer = False, number = 'Arabic'):

    question = item['question']
    roman_numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
    try:
        documents = [doc['doc'] for doc in item['docs']]
    except:
        documents = [doc['text'] for doc in item['docs']]
    if number == 'Arabic':
        docs_str = '\n'.join([f"[{i + 1}]: {doc}" for i, doc in enumerate(documents)])
    elif number == 'Roman':
        docs_str = '\n'.join([f"{roman_numerals[i]}: {doc}" for i, doc in enumerate(documents)])
    prompt = f"""Question: {question}

Documents:
{docs_str}

Answer:
""" if  with_answer else f"""
Question: {question}

Documents:
{docs_str}
"""
    return prompt

def filter():
    data = load_json('path')
    total = 0
    for item in data:
        if item.get('skip', False):
            continue
        oracle = item['oracle']
        item['oracle'] = []
        item['cot'] = []
        item['ref'] = []
        item['ans'] = []
        item['raw_oracle'] = []
        if isinstance(oracle, str):
            oracle = [oracle]

        for o in oracle:
            check = check_oracle(o)
            if check:
                cot, ref, ans  = check
                item['skip'] = False
            else:
                item['skip'] = True
            item['cot'].append(cot)
            item['ref'].append(ref)
            item['ans'].append(ans)
            item['raw_oracle'].append(o)
            item['oracle'].append(f"{cot}\n\nReferences:\n{ref}\n\nAnswer: {ans}")
    json.dump(data, open('final_data/dataset/cot_guidance_nogold_8b_part2_.json', 'w'), indent=4)


if __name__ == '__main__':
    filter()