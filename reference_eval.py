from data_create.system import LLM
import argparse
import json
import re
from nltk import sent_tokenize
from collections import defaultdict
import random
from tqdm import tqdm
import traceback
import os
find = True

def load_json(path):
    with open (path, 'r') as f:
        return json.load(f)

def remove_confidence(text):
    """
    Remove the confidence from the text.
    Confidence like (Confidece: 1.0)
    """
    # remove Document [x](Title:xxx):
    text = re.sub(r'Document\s*\[\d+\]\s*\(Title:\s*.+\):\s*', '', text)
    return re.sub(r'\(Confidence:\s*\d+\.\d+\)', '', text)

def re_order_by_question(standard, data):
    """
    Re-order the data by the standard data.
    """
    standard_dict = {item['question']: item for item in standard}
    data_dict = {item['question']: item for item in data}
    new_data = []
    for key in standard_dict.keys():
        new_data.append(data_dict[key])
    return new_data

def all_re_order(datas):
    """
    Re-order all the data by the first data.
    """
    standard = datas[0]
    return [re_order_by_question(standard, data) for data in datas]
 
def same_docs(data1, data2):
    """
    Check if the two data have the same docs.
    """
    for item1, item2 in zip(data1, data2):
        if item1['question'] != item2['question']:
            return False
        if any(doc1['title'] != doc2['title'] for doc1, doc2 in zip(item1['docs'], item2['docs'])):
            return False
    return True

def convert_passages(input_text, find = find):
    if find:
        title_pattern = r'\(Title:(.*?)\)'
        confidence_pattern = r'\(Confidence:\s*(\d+(?:\.\d+)?)\)'  # Matches decimal numbers
        
        # Extract title
        title_match = re.search(title_pattern, input_text)
        title = title_match.group(1).strip() if title_match else ''
        raw_title = title
        if '(' in title and ')' not in title:
            title = title + ')'
        
        # Extract confidence
        confidence_match = re.search(confidence_pattern, input_text)
        confidence_values = {0.0, 0.25, 0.5, 0.75, 1.0}  # Allowed confidence values
        confidence = float(confidence_match.group(1)) if confidence_match else None
        if confidence not in confidence_values:
            confidence = None
        
        # Extract text content
        text_start = input_text.find(title) + len(title) + 2
        text_end = confidence_match.start() if confidence_match else len(input_text)
        text = input_text[text_start:text_end].strip()
    else:
        title = ''
        text = input_text
        confidence = None
    # Create output dictionary
    output = {
        "title": title,
        "text": text,
        "confidence": confidence
    }

    return output

def find_score(text):
    # int score 1-5
    score = re.search(r'Score:\s*(\d)', text)
    if score:
        return int(score.group(1))
    else:
        return None

def extract_passages(data, max_ex_idx = 3):
    passages = []
    for item in data:
        if '_docs_' in item.keys():
            in_passages = item['_docs_'][max_ex_idx:]
        elif 'docs' in item.keys():
            in_passages = item['docs'][max_ex_idx:]
            if isinstance(in_passages[0], dict):
                in_passages = [txt['text_non_fact'] for txt in in_passages if 'text_non_fact' in txt.keys()] 
        else:
            in_passages = item['doc_cache'][max_ex_idx:]
        if isinstance(in_passages, list):
            in_passages = [{**convert_passages(txt), 'question': item['question']} for txt in in_passages]
        passages.extend(in_passages)

    return passages

def select_confidence_items(input_list, k = 50, eq= True):
    # Create a dictionary to store lists of items by confidence level
    confidence_dict = defaultdict(list)
    
    # Populate the dictionary with items from the input list
    for item in input_list:
        confidence = item['confidence']
        confidence_dict[confidence].append(item)
    
    # Create the output list
    output_list = []
    min_k = min([k] + [len(x) for x in confidence_dict.values()])
    print([len(x) for x in confidence_dict.values()])
    if min_k < k:
        print(f'Warning: k is too large and be adjusted to {min_k}')
    # Ensure we have enough items for each confidence level
    for confidence, conf_value in confidence_dict.items():
        if len(conf_value) < k:
            print(f"Not enough items with confidence {confidence}")
    
    # Randomly select k items for each confidence level
    if eq:
        for confidence, conf_value in confidence_dict.items():
            if not len(conf_value) < k:
                output_list.extend(random.sample(conf_value, k))
            else:
                output_list.extend(conf_value)
    else:
        output_list =random.sample(input_list, k)
    
    return output_list

class ReferenceLLM(LLM):
    def __init__(self, args, short_ans =  False):
        super().__init__(args, usecache = False)
        self.relevance_return = 'float'
        if short_ans:
            self.length = 5
        else:
            self.length = 1000
    def generate(self, prompt, max_tokens, stop=None, system='You are a helpful assistant that answers the following questions'):
        return super().generate(prompt, max_tokens, stop, system)
    

    def mark_entities(self, text):
        instruction = "Mark the entities in the following text, use names such as Alice, Bob, and Charlie as placeholders for names. Use placeholders like Location A/B/C, Organization A/B/C, Group A/B/C to mark other entities. You don't need to mark time or date entities. Make sure the same entity is marked consistently throughout the text. Here is the text:\n\n"
        return self.generate(instruction + text + 'Marked text:', 500)

    def convincing(self, text, max_retries=3):
        instruction = """"Evaluate the epistemic strength of the passage ocale of 1 to 5. IMPORTANT: Placeholder terms like 'Alice,' 'Bob,' 'Location A,' or 'Organization B.' are used to avoid any reference to the real world, you should never give a low score becaus of it. Focus instead on properties like the passage's objectivity. Assign a score from 1 (very weak) to 5 (very strong) based solely on these factors, regardless of entity placeholders. 
        
        Examples:
        5: A Wikipedia-like passage with a formal tone, structured, and enough relative information.
        "In August 2012, Alice and Bob announced that they were divorcing. In April 2015, Alice married former NFL linebacker Claro. In March 2016, it was announced that the couple was expecting their first child. On September 7, 2016, Alice gave birth to a son. Alix (born January 22, 1981), known by her stage name Ali, is an American singer, songwriter, dancer, model, television personality and film actress. She released her debut album in 2001. "
        Brief Explanation: This text is objective, prividing enough information to identify the entities and their relationships.

        3: Moderate, may lead to some doubt. 
        "I was reflecting on how, exactly 17 years after their first Super Bowl showdown, the New England Patriots faced off against the Los Angeles Rams in Super Bowl LIII, triumphing in Atlanta. The Patriots kicked off the scoring with a touchdown from Sony Michel, who powered his way in from two yards out with just 7 minutes left in the fourth quarter, bringing the score to 10-3. This was a notable achievement, as he became the first rookie in Patriots history to score a touchdown in a Super Bowl. "
        Brief Explanation: The text is not objective, but still providing enough information to clarify the relationship between entities.

        1: Poor orgnized, subjective, lack of information.
        ""Today\u2019s film is **Film A**. I realized that this film must be the top 1 this year. Information has shown that it is filmed in a European contry, with a large number of crews spending several years on it.
        Brief Explanation: The text is not objective, and it is difficult to identify the entities and their relationships.
        
        Subjective or opinionated text (e.g. including "I" or "my") should be scored low.
        Directly output a number first as the score. After scoring, briefly explain why the score was given. Here is then a s text:\n\n"""

        instruction = """
You are an evaluator tasked with assessing the convincing of a text. convincing is defined as the text's ability to avoid raising doubts about its truthfulness in the reader. Consider the following criteria while scoring:

1. **Subjectivity**: Assess whether the language is objective and free from excessive bias or personal opinions. If a personal opinion is present, it should be considered doubtful.
2. **Style and tone**: Consider whether the tone and structure of the text is appropriate. phrases like "Hi there" or "As you may know" or informal title will give a conversation tone, which undermines the convincing.
3. **Possible source**: Evaluate whether the text is likely to be from a convincing source. If the style of the text make it seem to come from informal forums, it should be considered doubtful, and source like wikipedia should be considered convincing.

**Scoring Guidelines:**

- **Score: 1 (Very Low convincing)**  
  - Dominated by subjective or emotional language.  
  - The tone or structure is highly inappropriate for the context (e.g., casual or conversational).  
  - The text is sure to be from an informal source.

- **Score: 2 (Low convincing)**  
  - Has a noticeable bias or subjective tone.    
  - The tone or structure partially undermines the convincing.  
  - The text is likely to be from an informal source.

- **Score: 3 (Moderate convincing)**  
  - Language is somewhat balanced but may lean towards subjectivity.  
  - The tone and structure are generally appropriate but may have occasional lapses.  
  - The text is likely to be from a reliable source.

- **Score: 4 (High convincing)**  
  - Language is objective and neutral.   
  - The tone and structure are well-suited to the context.  
  - The text is likely to be from a convincing source.

- **Score: 5 (Very High convincing)**  
  - Language is fully objective and professional.  
  - The tone and structure perfectly align with the expected standard for the context.  
  - The text is likely to be from a highly convincing source.

Provide a short explanation of your reasoning, and then output a score between 1 and 5, with formatting like "Score: 3"."
"""
        instruction = """
You are an evaluator tasked with assessing the convincing of a text. convincing is defined as the text's ability to avoid raising doubts about its truthfulness in the reader. Consider the following criteria while scoring: 

1. **Logical Consistency**: Evaluate whether the text avoids logical errors or contradictions.
2. **Subjectivity**: Assess whether the language is objective and free from excessive bias or personal opinions.
3. **Coherence and Focus**: Determine if the arguments are well-connected and focused rather than scattered or overly parallel.
4. **Information Density**: Consider whether the text provides sufficient relevant information to substantiate its claims.

Please assign a score between 1 and 5 based on the following detailed guidelines:

1. **Score: 1 (Very Low convincing)**  
   - Contains multiple logical errors or glaring contradictions.  
   - Dominated by subjective or emotional language.  
   - Arguments are highly scattered, with no clear connections between points.  
   - Lacks sufficient information to support its claims.

2. **Score: 2 (Low convincing)**  
   - Contains some logical inconsistencies or weak reasoning.  
   - Has a noticeable bias or subjective tone.  
   - Arguments are somewhat scattered, with limited connections between points.  
   - Provides insufficient evidence or relies on vague statements.

3. **Score: 3 (Moderate convincing)**  
   - Mostly logical with minor inconsistencies.  
   - Language is somewhat balanced but may lean towards subjectivity.  
   - Arguments are somewhat connected but may lack focus or clarity.  
   - Contains adequate but not robust information density.

4. **Score: 4 (High convincing)**  
   - Logically consistent with no major errors.  
   - Language is objective and neutral.  
   - Arguments are mostly coherent and focused.  
   - Provides substantial and relevant evidence for its claims.

5. **Score: 5 (Very High convincing)**  
   - Completely free from logical errors or contradictions.  
   - Language is fully objective and professional.  
   - Arguments are tightly connected and maintain a clear focus.  
   - Provides rich, detailed, and highly relevant information to support its claims.

Provide a short explanation of your reasoning, and then output a score between 1 and 5, with formatting like "Score: 3".
"""

        text = self.mark_entities(text)
        retry = 0
        while retry < max_retries:
            try:
                ans = self.generate(instruction + text + 'Epistemic Strength:', self.length)
                number = find_score(ans)
                assert 1 <= number <= 5
                break
            except Exception as e:
                retry += 1
                if retry == max_retries:
                    return {'score': -1, 'explanation': ans, 'retry': retry}
        return {'score': number, 'explanation': ans, 'retry': retry}
    

    def sentence_level_relevance(self, question, text, max_retries=3):
        instruction = f"""
        Instruction:
        Evaluate the Conciseness Score of the passage on a scale from 1 to 5. Conciseness Score refers to the level of mental effort required to understand the main point, considering how much unnecessary information or filler content is included.:

        A sentence answering part of the question is considered 0% redundant, e.g. "Billie Eilish won a 2019 Grammy awards. " is 0% redundant to the question "How many Grammy awards did Billie Eilish win in 2019?".
        A sentence with several related entities about the question is 50% redundant, e.g. "Billie Eilish is a promising competitor for Grammy awards." is 50% redundant to the question "How many Grammys did Billie Eilish win in 2019?".
        A sentence related to the question, but only with some basic background or even not related is considered 100% redundant, e.g. "Billie Eilish is a popular singer." is 100% redundant to the question "How many Grammys did Billie Eilish win in 2019?".
        You should consider the relevance of each sentence to the question and give a score based on the average Conciseness of the passage.

        Here are eaxmples of the Conciseness Score to the question "How many Formula One world championships has Max Verstappen won?":

        1: Very High Conciseness (the passage is overly verbose, making it challenging to grasp the main and useful point, with Conciseness ~ 100%)
        Example:
        Formula One, known for its fierce competition and technical excellence, attracts drivers from around the world who aspire to win the prestigious World Drivers' Championship. The word formula in the name refers to the set of rules all participants' cars must follow. A Formula One season consists of a series of races, known as Grands Prix. Grands Prix take place in multiple countries and continents on either purpose-built circuits or closed roads.
        Brief Explanation: Each sentence provides general information about Formula One, but the passage is overly verbose and lacks specific details about Max Verstappen's championship wins, each can be considered as 0%, so the score is 4/4 = 100%.

        2: High Conciseness (some supporting information is present, but it is overshadowed by less relevant content, with Conciseness ~ 75%).

        3: Moderate Conciseness (the passage provide supporting and unnecessary information, with Conciseness ~ 50%).
        Example:
        Max Verstappen is widely regarded as one of the most influential drivers in modern Formula One. Max Verstappen has achieved remarkable success in Formula One, winning his first World Championship in 2021 after a dramatic season finale. Formula One, known for its fierce competition and technical excellence, attracts drivers from around the world who aspire to win the prestigious World Drivers' Championship. Verstappen's career, fueled by skill and determination, has been celebrated in many racing seasons, leaving fans and analysts curious about his championship successes and ongoing impact on the sport.
        Brief Explanation: The first sentence is about Max and F1, so it is 50% related. The second sentence detailed the information of a chanpionship, strongly related. The third sentence is only about F1, so it is 100% redundant. The last sentence is about Max's championship, so it is 50% related. The score is (0.5 + 0 + 1 + 0.5)/4 = 50%.

        4: Low Conciseness (most sentences are relevant, with minimal non-essential information, Conciseness ~ 25%).

        5: Very Low Conciseness (the passage is clear, concise, and directly supports the main answer without unnecessary details, with Conciseness ~ 0%).
        Example:
        Max Verstappen has achieved remarkable success in Formula One, winning his first World Championship in 2021 after a dramatic season finale. He continued his dominance by securing a second title in 2022, showcasing his skill and consistency on the track. In 2023, Verstappen solidified his legacy with a third consecutive championship, demonstrating his exceptional talent and dedication. These three titles underscore his position as one of the premier drivers in the sport, marking him as a modern champion with three Formula One World Championships to his name.
        Brief Explanation: The first three sentences provide detailed information about each championships won by Max Verstappen, followed by a conclusion that directly answers the question. Each sentence is highly related, so the Conciseness is 0/4 = 0%

        Directly output a number first as the score. THE SCORE SHOULD BE GIVEN BASED ON THE AVERAGE Conciseness YOU CALCULATED. After scoring, briefly explain the calculation. """

        instruction = """
        You are tasked with assessing the Conciseness of a document sentence by sentence in response to a given question and answer. For each sentence:  
        1. Judge whether it positively contributes to answering the question (positive), partially contributes but feels unnecessary or tangential (neutral), or detracts from the relevance (negative).  
        2. Provide a brief explanation for your judgment.  

        After reviewing all sentences, summarize the overall Conciseness of the document and assign a score between 1 and 5, following these guidelines:  
        - **5 (Very Low Conciseness):** All sentences are relevant or contribute directly to answering the question.  
        - **4 (Low Conciseness):** Most sentences are relevant, with a few mildly tangential or unnecessary.  
        - **3 (Moderate Conciseness):** A balance of relevant and irrelevant content; reader effort is moderate.  
        - **2 (High Conciseness):** Many sentences are tangential or unnecessary, requiring significant effort to find relevant information.  
        - **1 (Very High Conciseness):** The majority of the document is irrelevant or distracting, with little useful content.  

        **Example:**  

        **Question:** How many championships has Messi won?  

        **Document:**  
        1. "Lionel Messi was born on June 24, 1987, in Rosario, Argentina, and is a professional footballer."  
        - **Positive:** This sentence establishes Messi as the subject, making it clear the document is on topic.  

        2. "His parents are Jorge Messi, a steel factory manager, and Celia Cuccittini, who worked in a magnet manufacturing workshop."  
        - **Negative:** This sentence delves into his family background, which feels irrelevant to the question about championships.  

        3. "He won his first championship in 2005, leading his team to victory in the U-20 World Cup."  
        - **Positive:** This sentence is highly relevant, directly addressing Messi’s championship history.  

        4. "His most recent championship was the 2022 FIFA World Cup, where he captained Argentina to victory."  
        - **Positive:** This sentence is also highly relevant, discussing a key championship victory.  

        5. "Messi hopes to continue playing at a high level and achieve more milestones in his career."  
        - **Neutral (slightly negative):** While unrelated to his past championships, it serves as a closing summary and doesn’t significantly detract from the document.  

        **Overall Assessment:**  
        The document is mostly focused on answering the question, with only one sentence being significantly off-topic. While the fifth sentence is mildly tangential, it serves as a conclusion and does not greatly impact the overall relevance.  

        Score: 4 (Low Conciseness)  

        Provide a short explanation of your reasoning for each sentence, and then output a score between 1 and 5, with formatting like "Score: 3"."
        """
        ans = self.generate(instruction + f'\n\nQuestion: {question}\n\n{text}\n\nScore:', self.length)
        retry = 0
        while retry < max_retries:
            try:
                number = find_score(ans)
                assert 1 <= number <= 5
                break
            except Exception as e:
                retry += 1
                if retry == max_retries:
                    return {'score': -1, 'explanation': ans, 'retry': retry}
        return {'score': number, 'explanation': ans, 'retry': retry}    
    
    def get_score(self, text, question, score = None):
        text = remove_confidence(text)
        if score and score['convincing']['score'] != -1:
            credi_score = score['convincing']
        else:
            credi_score = self.convincing(text)
        if score and score['concise']['score'] != -1:
            rel_score = score['concise']
        else:
            rel_score = self.sentence_level_relevance(question, text)
        return {'convincing': credi_score, 'concise': rel_score}

def main(args):
    args = {
    'openai_api': True,
    'azure': False,
    'max_tokens': 100,
    'model': 'gpt-4o-mini',
    'temperature': 0.5,
    'top_p': 0.9,   
    }

    args = argparse.Namespace(**args)
    llm = ReferenceLLM(args=args)

    personalized = lambda text: ' I ' in text or ' my ' in text or ' me ' in text

    def fact_check_pipeline(in_file, sample_amount, out_file, max_ex_idx = 3, eq = True, person = False):
        data = load_json(in_file)
        psg = extract_passages(data, max_ex_idx = max_ex_idx)
        if person:
            psg = [p for p in psg if personalized(p['text'])]
        psg = select_confidence_items(psg, k = sample_amount, eq = eq)
        for i, p in tqdm(list(enumerate(psg))):
            p['score'] = llm.get_score(p['text'], p['question'])
            if i%2 == 0:
                with open(out_file, 'w') as f:
                    json.dump(psg[:i], f, indent = 4)
        with open(out_file, 'w') as f:
            json.dump(psg, f, indent = 4)
        return psg

    def fact_check_pipeline_combine(in_filew, sample_amount, out_file, max_ex_idx = 3, eq = True, person = False):
        datas = [load_json(in_file) for in_file in in_filew]
        datas = all_re_order(datas)
        assert all(same_docs(datas[0], data) for data in datas)
        psgs = [extract_passages(data, max_ex_idx = max_ex_idx) for data in datas]
        random_indices = random.sample(range(len(psgs[0])), sample_amount)
        psgs = [[psg[i] for i in random_indices] for psg in psgs]
        assert all(len(psg) == sample_amount for psg in psgs)

        output = []
        for i in tqdm(range(sample_amount)):
            ps = [psg[i] for psg in psgs]
            for p in ps:
                p['score'] = llm.get_score(p['text'], p['question'])
            output.append(ps)
            if i%2 == 0:
                with open(out_file, 'w') as f:
                    json.dump(output, f, indent = 4)
        with open(out_file, 'w') as f:
            json.dump(output, f, indent = 4)
        return psgs
    
def check_results(accuracy_file, output_file, max_retries = 3, filtering = None, dataset = None, max_doc = 100, ex = False):
    args = {
    'openai_api': True,
    'azure': False,
    'max_tokens': 100,
    'model': 'gpt-4o-mini',
    'temperature': 0.5,
    'top_p': 0.9,   
    }

    args = argparse.Namespace(**args)
    llm = ReferenceLLM(args=args)
    data = load_json(accuracy_file)
    total_scores = 0
    total_convincing = 0
    total_concise = 0
    for i, item in tqdm(list(enumerate(data))):
        if not filtering or filtering(item, i):
            all_docs = item['docs']
            all_sents = item['evals']
            docs = []
            for sent in all_sents:
                if sent['entail']:
                    sentence = sent['sentence']
                    citations = re.findall(r'\[\d+\]', sentence)
                    citation_numbers = [int(c[1:-1]) for c in citations]
                    citation_indices = [c - 1 for c in citation_numbers]
                    doc_texts = [all_docs[c] for c in citation_indices]
                    joined_text = ' '.join(doc_texts)
                    docs.append(joined_text)
            item['reference_scores'] = []
            question = item['question']
            #ex_idx = item['ex_idx']
            #if ex_idx >= len(docs):
            #    in_docs = []
            #else:
            #    in_docs = docs[ex_idx:]
            #ex_docs = docs[:ex_idx]
            eval_docs = docs
            for doc in eval_docs:
                curr_score = None
                for _ in range(max_retries):
                    try:
                        curr_score = llm.get_score(doc, question, curr_score)
                        if 'convincing' in curr_score and curr_score['convincing']['score'] != -1 and 'concise' in curr_score and curr_score['concise']['score'] != -1:
                            item['reference_scores'].append(curr_score)
                            credi_score = curr_score['convincing']['score']
                            rel_score = curr_score['concise']['score']
                            total_scores += 1
                            total_convincing += credi_score
                            total_concise += rel_score
                            if total_scores == max_doc:
                                print(f'FINAL: Evaluated {i} examples in file {str(accuracy_file)}, {total_scores} documents, average convincing: {total_convincing/total_scores}, average concise: {total_concise/total_scores}')
                                with open(output_file, 'w') as f:
                                    json.dump(data, f, indent = 4)
                                return
                            break
                    except:
                        traceback.print_exc()
                        continue
        if i % 100 == 0 and total_scores > 0:
            print(f'PROCESSING: Evaluated {i} examples in file {str(accuracy_file)}, {total_scores} documents, average convincing: {total_convincing/total_scores}, average concise: {total_concise/total_scores}')
            with open(output_file, 'w') as f:
                json.dump(data, f, indent = 4)
    
    print(f'FINAL: Evaluated {i} examples in file {str(accuracy_file)}, {total_scores} documents, average convincing: {total_convincing/total_scores}, average concise: {total_concise/total_scores}')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent = 4)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--accuracy_file', type=str, help='The file to evaluate')
    argparser.add_argument('--output_file', type=str, help='The output file')
    argparser.add_argument('--max_retries', type=int, default=3, help='The maximum number of retries')
    argparser.add_argument('--dataset', type=str, default=None, help='The dataset to filter')
    argparser.add_argument('--dir', type=str, default=None, help='The directory to evaluate')
    argparser.add_argument('--out_dir', type=str, default=None, help='The directory to evaluate')
    argparser.add_argument('--ex', action='store_true', help='Whether to evaluate the examples')
    argparser.add_argument('--max_doc', type=int, default=50, help='The maximum number of documents to evaluate')
    argparser.add_argument('--split', action='store_true', help='Whether to split the evaluation')
    args = argparser.parse_args()
    
    def filter_gold(item, i):
        data = dataset[i]
        assert item['question'] == data['question']
        return any(doc['gold'] for doc in data['docs'])
    
    def filter_non_gold(item, i):
        data = dataset[i]
        assert item['question'] == data['question']
        return not any(doc['gold'] for doc in data['docs'])
    
    def knowledge(item, i):
        data = dataset[i]
        assert item['question'] == data['question']
        return item['p_conf'] > 0.01
    def non_knowledge(item, i):
        data = dataset[i]
        assert item['question'] == data['question']
        return item['p_conf'] < 0.01
    g_k = lambda item, i: filter_gold(item, i) and knowledge(item, i)
    g_nk = lambda item, i: filter_gold(item, i) and non_knowledge(item, i)
    ng_k = lambda item, i: filter_non_gold(item, i) and knowledge(item, i)
    ng_nk = lambda item, i: filter_non_gold(item, i) and non_knowledge(item, i)
    
    ex_str = '_ex' if args.ex else ''
    dataset = load_json(args.dataset) if args.dataset else None
    if args.dir:
        for file in os.listdir(args.dir):
            # Eval json files but skip files that are already scored
            if (file.endswith('.json') or file.endswith('.jsonl')) and 'score' not in file:

                try:
                    if args.split:
                        print('Checking file, gold, with knowldge:', file)
                        check_results(args.dir + '/' + file, args.out_dir + '/' + file.replace('.json', f'_gold_score{ex_str}.json'), args.max_retries, g_k, dataset, max_doc = args.max_doc, ex = args.ex)
                        print('Checking file, gold, no knowledge:', file)
                        check_results(args.dir + '/' + file, args.out_dir + '/' + file.replace('.json', f'_gold_noknowledge_score{ex_str}.json'), args.max_retries, g_nk, dataset, max_doc = args.max_doc, ex = args.ex)
                        print('Checking file, no gold, with knowldge:', file)
                        check_results(args.dir + '/' + file, args.out_dir + '/' + file.replace('.json', f'_nogold_score{ex_str}.json'), args.max_retries, ng_k, dataset, max_doc = args.max_doc, ex = args.ex)
                        print('Checking file, no gold, no knowledge:', file)
                        check_results(args.dir + '/' + file, args.out_dir + '/' + file.replace('.json', f'_nogold_noknowledge_score{ex_str}.json'), args.max_retries, ng_nk, dataset, max_doc = args.max_doc, ex = args.ex)
                    else:
                        print('Checking file:', file)
                        check_results(args.dir + '/' + file, args.out_dir + '/' + file.replace('.json', f'_score{ex_str}.json'), args.max_retries, lambda item, i: True, dataset, args.max_doc, args.ex)
                except:
                    print('Error in file:', file)
                    traceback.print_exc()
    else:
        check_results(args.accuracy_file, args.output_file, args.max_retries, lambda item, i: True, dataset, args.max_doc, args.ex)
