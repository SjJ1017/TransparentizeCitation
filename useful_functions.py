import json
import shutil
from data_processing import fill_input_direct
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
def nogold_cot():
    "Generate a NO GOLD version of a dataset"
    inner = json.load(open('path'))
    data = json.load(open('path'))


    for i, d, in zip(data, inner):
        assert i['question'] == d['question']
        i['inner_passages'] = d['inner_passages']
        i['direct_answers'] = d['direct_answers']
        i['p_conf'] = d['p_conf']
        i['a_conf'] = d['a_conf']
        i['model_conf'] = d['model_conf']

    json.dump(data, open('final_data/inner_confidence_70-.json', 'w'), indent=4)


def create_cot_eval_dataset():
    "Create a dataset for evaluation"
    INSTRUCTION = 'In this task, you will be given a question and chain of reasoning. Your task is to generate a relevant answer to the question using the provided chain of reasoning. The answer should be accurate, engaging, and concise. You can use and only use the chain of reasoning to help you answer the question. If the reasoning indicate the information is from a specific document like Document I, II..., you should cite the document using [I], [II]... in your answer at the end of the sentence. If the reasoning indicate the information is from internal knowledge, you should cite the information using [internal] in your answer at the end of the sentence.\n\n'
    INPUT_TEMPLATE = 'Question: {question}\n\nChain of Reasoning:\n{chain}\n\nAnswer: '

    dataset = json.load(open('dataset_path'))
    prompts = []
    for item in dataset:
        input = item['input']
        output = item['output']
        question = input.split('\n')[0].replace('Question: ', '')
        chain = output.split('\n\n')[0].replace('Think step by step:\n', '')
        prompt = {}
        prompt['instruction'] = INSTRUCTION
        prompt['input'] = INPUT_TEMPLATE.format(question = question, chain = chain)
        prompt['output'] = ''
        prompts.append(prompt)
    json.dump(prompts, open('final_data/dataset/cot_eval.json', 'w'), indent=4)

def get_scaled_traininig_data(training_data, output_dir, lens = [200, 400, 600]):
    import random

    for l in lens:
        random.shuffle(training_data)
        data = training_data[:l]
        json.dump(data, open(f'{output_dir}/train_{l}.json', 'w'), indent=4)


def push_data(dir):
    import os
    if dir.endswith('/'):
        dir = dir[-1]
    json_files = os.listdir(dir)
    json_files = [x for x in json_files if x.endswith('.json')]
    info_name = 'dataset_info.json' 
    shutil.copy2(f'{dir}/{info_name}', f'{dir}/{info_name}_backup')
    data_info = json.load(open(f'{dir}/{info_name}'))
    
    assert info_name in json_files, f'{info_name} is not in the directory'
    for file in json_files:
        if file == info_name:
            continue
        dataset_name = file.replace('.json','')
        if dataset_name not in data_info:
            data_info[dataset_name] = {'file_name': file}
            print(f'Added data {dataset_name}')
    json.dump(data_info, open(f'{dir}/{info_name}', 'w'), indent=4)

def gen_online_data(demo_path, file, out):

    data = load_json(file)
    demo = load_json(demo_path)
    instruction = demo['instruction']
    demos = demo['demos']
    demo_1, demo_2 = demos[0], demos[2]
    few_shot = fill_input_direct(demo_1, with_answer = True, number = 'Arabic')  + demo_1['answer'] + '\n\n' + fill_input_direct(demo_2, with_answer = True, number = 'Arabic')  + demo_2['answer'] + '\n\n'
    new = []
    for item in data:
        input = fill_input_direct(item, with_answer = True, number = 'Arabic')
        output = item['oracle']
        if isinstance(output, list):
            output = output[0]
        new_item = {}
        new_item['instruction'] = instruction
        new_item['input'] = few_shot + input
        new_item['output'] = output
        new.append(new_item)
    json.dump(new, open(out, 'w'), indent=4)

def gen_rag_data(file, out):

    instruction = """Instruction: Write an accurate, engaging, and concise answer for the given question. You can use the provided documents (some of which might be irrelevant). Use an unbiased and journalistic tone."""
    data = load_json(file)
    new = []
    for item in data:
        input = fill_input_direct(item, with_answer = True, number = 'Arabic')
        output = item['oracle']
        if isinstance(output, list):
            output = output[0]
        new_item = {}
        new_item['instruction'] = instruction
        new_item['input'] = input
        new_item['output'] = output
        new.append(new_item)
    json.dump(new, open(out, 'w'), indent=4)

def gen_training_data(file, super, out, super_is_self = False):
    instruction = """You are an assistant tasked with answering questions based on the provided documents with citations. You can use your internal knowledge to enhance the answer. 
Format your output as follows:

Step-by-step analysis: Write a detailed explanation of how the information was derived, explicitly referencing the documents and internal knowledge used.
References: Provide a numbered list of the sources used, formatted with the exact text cited from each document or internal knowledge point. Use the format [1], [2], [3]... to cite the sources. When citing the provided documents, you should select a fine-grained span from the documents, and ensure the spans is credible and less redundant. Use Roman numerals to mark the document and use Arabic numerals to mark spans. You can also use your internal knowledge to enhance the answer. In that case, use Arabic numerals to mark the internal knowledge points, and the number should follow the last document index. You also need to output your confidence about the facutal correctness of your knowledge scaled from 0.0 to 1.0.
Final Answer: Summarize the conclusion with appropriate citations in the same format as the references.

"""
    data = load_json(file)
    super_dataset =load_json(super)
    new = []
    for item in data:
        if not super_is_self:
            raw_oracle = item['raw_oracle']
            for i, super_item in enumerate(super_dataset):
                if super_item.get('skip', False):
                    continue
                if super_item["raw_oracle"] == raw_oracle:
                    super_item = super_dataset[i]
                    break
        else:
            super_item = item
        
        #prompt = fill_input_direct(item, with_answer = False, number = 'Roman')
        input = fill_input_direct(super_item, with_answer = True, number = 'Roman')
        output = item['oracle']
        if isinstance(output, list):
            output = output[0]
        new_item = {}
        new_item['instruction'] = instruction
        new_item['input'] = input
        new_item['output'] = output
        new.append(new_item)
    json.dump(new, open(out, 'w'), indent=4)
    return new

def gen_nocot_data_from_train(file, super, out, super_is_self = False):
    instruction = """You are an assistant tasked with answering questions based on the provided documents with citations. You can use your internal knowledge to enhance the answer. 
Format your output as follows:
References: Provide a numbered list of the sources used, formatted with the exact text cited from each document or internal knowledge point. Use the format [1], [2], [3]... to cite the sources. When citing the provided documents, you should select a fine-grained span from the documents, and ensure the spans is credible and less redundant. Use Roman numerals to mark the document and use Arabic numerals to mark spans. You can also use your internal knowledge to enhance the answer. In that case, use Arabic numerals to mark the internal knowledge points, and the number should follow the last document index. You also need to output your confidence about the facutal correctness of your knowledge scaled from 0.0 to 1.0.
Final Answer: Summarize the conclusion with appropriate citations in the same format as the references.

"""
    data = load_json(file)
    for item in data:
        assert 'References:' in item['output'], item['']
        item['output'] = 'References:' + item['output'].split('References:')[1]
        item['instruction'] = instruction
    json.dump(data, open(out, 'w'), indent=4)
    return data


def split_train_eval(file, out_train, out_eval):
    data = load_json(file)
    import random
    rate = 0.8
    random.shuffle(data)
    train = data[:int(len(data)*rate)]
    eval = data[int(len(data)*rate):]
    json.dump(train, open(out_train, 'w'), indent=4)
    json.dump(eval, open(out_eval, 'w'), indent=4)

def find_dataset(train_file, dataset, out):
    new = []
    train = json.load(open(train_file))
    data = json.load(open(dataset))
    for item in train:
        output = item['output']
        for i in data:
            if i['oracle'] == output:
                new.append(i)
                break

    assert len(new) == len(train)
    json.dump(new, open(out, 'w'), indent=4)

def find_dataset_reranked(train_file, dataset, out):
    new = []
    train = json.load(open(train_file))
    data = json.load(open(dataset))
    for item in train:
        output = item['output']
        for i in data:
            if i['oracle'] == output:
                new.append(i)
                break

    assert len(new) == len(train)
    json.dump(new, open(out, 'w'), indent=4)

def delete_abstain(file, out, abstain_rate = 0.0):
    import random
    data = json.load(open(file))
    new_data = []
    abstain = []
    max_num = 0
    curr_num = 0
    ABSTAIN_TEXT = "I don't have sufficient knowledge to answer the question, and there is no relevant information in the provided documents to answer the question."
    for item in data:
        if ABSTAIN_TEXT in item['output']:
            curr_num += 1
            if curr_num > max_num:
                abstain.append(item)
            else:
                new_data.append(item)
        else:
            new_data.append(item)
    print(len(abstain))
    print(len(new_data))
    abstain_len = int(len(new_data)*abstain_rate)
    abstain = random.sample(abstain, abstain_len)
    if not abstain:
        print('No abstain data')
    json.dump(new_data, open(out, 'w'), indent=4)
    return new_data

def convert_dataset_to_grounding(file, out, doc = 5):
    INSTRUCTION = "Extract the relevant content from the provided documents and then use the extracted content to guide answer generation and cite the sources properly."
    data = load_json(file)
    for item in data:
        item['instruction'] = INSTRUCTION
        item['input'] = item['input'].replace('\n\nAnswer:\n', '').replace('\n\nDocuments:\nI: ', '\n\nI: ')
        map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10}
        for key, value in map.items():
            if value <= doc:
                if f'\n{key}: ' not in item['input']:
                    print(item['input'])
                    raise Exception('Not found')
            item['input'] = item['input'].replace(f'\n{key}: ', f'\nDocument [{value}]: ')
    json.dump(data, open(out, 'w'), indent=4)
def get_eval_plus(train, full, out, non_seen_questions = True):
    train_dataset = load_json(train)
    full_dataset = load_json(full)
    all_questions = [x['question'] for x in train_dataset]
    eval_plus = []
    for item in full_dataset:
        if item['question'] not in all_questions or not non_seen_questions:
            found = False
            for train_item in train_dataset:
                if train_item['raw_oracle'] == item.get('raw_oracle', ''):
                    found = True
                    break
            if not found:
                eval_plus.append(item)

    import random
    random.shuffle(eval_plus)
    eval_plus = eval_plus[:1000]
    json.dump(eval_plus, open(out, 'w'), indent=4)
    return eval_plus

def combine_for_rerank(file, out, accurate_file= None):
    from tqdm import tqdm
    data = load_json(file)
    if accurate_file:
        accurate_data = load_json(accurate_file)
        assert len(data) == len(accurate_data)
        for i, ai in zip(data, accurate_data):
            i['accuracy'] = ai['accuracy']
    #data = list(filter(lambda x: not x.get("reference_scores", False), data))
    #data = list(filter(lambda x: not x.get("skip", False), data))
    step = 3
    for i in tqdm(range(0, len(data), step)):
        questions = [data[i + j]['question'] for j in range(step)]
        assert len(set(questions)) == 1
        all_scores = []
        for j in range(step):
            item = data[i + j]
            reference_scores = item.get("reference_scores", [])
            if not reference_scores:
                scores = (0, 0)
            else:
                cred = reference_scores[0]['convincing']['score']
                concise = reference_scores[0]['concise']['score']
                scores = (cred, concise)
            item['scores'] = scores
            all_scores.append(scores)
        for j in range(step):
            data[i + j]['scores'] = all_scores

    filtered = []
    for i in tqdm(range(0, len(data), step)):
        options = [data[i + j] for j in range(step)]
        scores = options[0]['scores']
        max_idx = max(range(len(scores)), key=lambda x: scores[x][0] + scores[x][1])
        max_score_option = options[max_idx]
        #raise KeyboardInterrupt
        if scores[max_idx][0] + scores[max_idx][1] < 0.1:
            continue
        max_score_option['idx'] = max_idx
        max_score_option['oracle'] = max_score_option['oracle'][max_idx]
        filtered.append(max_score_option)
    filtered = list(filter(lambda x: not x.get("skip", False), filtered))
    filtered = list(filter(lambda x: x["accucary"]["rag"], filtered))
    filtered = list(filter(lambda x: all([sent['entail'] for sent in x['evals']]), filtered))
    json.dump(filtered, open(out, 'w'), indent=4)


