import json
import re
from system import LLM
import argparse
from tqdm import tqdm
import traceback


TIP_EXTERNAL = "It seems that the answer supposed to be in the provided documents. you may need not to use your internal knowledge."
TIP_INTERNAL = "It seems that the answer is not in the provided documents. You may need to use your internal knowledge to answer the question."
TIP_ABSTENTION = "It seems that the answer is not in the provided documents, and you don't have enough knowledge to answer the question, you may need to abstain from answering the question. (Think step by step, generate empty references and an abstention response)"
tip_dict = {
    'external': TIP_EXTERNAL,
    'internal': TIP_INTERNAL,
    'abstention': TIP_ABSTENTION
}

EXTERANL_EXAMPLE = """Question: How many albums does journey have?

`Documents`:
I: American rock band Journey has released 15 studio albums, five live albums, 11 compilation albums, and 52 singles since 1975. ... ^ &quot;Separate Ways (Worlds Apart)&quot; did not enter the NZ Top 40 Singles Chart, but peaked at number 21 on the NZ Hot Singles Chart in 2022.American rock band Journey has released 15 studio albums, five live albums, 11 compilation albums, and 52 singles since 1975. ... ^ \"Separate Ways (Worlds Apart)\" did not enter the NZ Top 40 Singles Chart, but peaked at number 21 on the NZ Hot Singles Chart in 2022. 
II: The movie, directed by Ramona S. Diaz, chronicles the discovery of Arnel Pineda and his first year with Journey.  Journey is an American rock band that formed in San Francisco in 1973, composed of former members of Santana and Frumious Bandersnatch. The band has gone through several phases; its strongest commercial success occurred between 1978 and 1987. During that period, the band released a series of hit songs, including \"Don't Stop Believin' (1981), which in 2009 became the top-selling track in iTunes history among songs not released in the 21st century. Its parent studio album, \"Escape\", the band's eighth and most successful, reached No. 1 on the \"Billboard\" 200 
III: American rock band Journey has released 15 studio albums, five live albums, 11 compilation albums, and 52 singles since 1975. ... ^ &quot;Separate Ways (Worlds Apart)&quot; did not enter the NZ Top 40 Singles Chart, but peaked at number 21 on the NZ Hot Singles Chart in 2022.

`my knowledge`:
Journey is an American rock band that formed in San Francisco, California in 1973. The band has had a successful career spanning over 40 years, with a total of 15 studio albums, 3 live albums, and 5 compilation albums. Their debut album, \"Journey\", was released in 1975 and was not a commercial success. However, their second album, \"Look into the Future\", released in 1976, showed a significant improvement in sales. The band's most successful album to date is \"Escape\", released in 1981, which included the hit singles \"Who's Crying Now\" and \"Open Arms\". (Confidence: 0.4)

Golden Answer: Journey has 15 studio albums.

Output:
Think step by step:
I need to find the number of all the albums released by Journey, or a list of all the names of the albums and sum them up. Document II provides the exact number of studio albums released by Journey, stating that the band has released 15 studio albums since 1975. However, the second half of the document is redundant, and the first sentence includes basic information like career and number of singles about Journey, which is credible.  Other documents are providing redundant or similar informations, I don't need to leverage them. I might know some exact names of Journeys albums but not pretty sure. I decided not to use them because the provided documents are sufficient to answer the question. 

References:
[1] "American rock band Journey has released 15 studio albums, five live albums, 11 compilation albums, and 52 singles since 1975.", Document II.

Answer: The band Journey has released five live albums, 11 compilation albums, and 52 singles since 1975 [1]. """

INTERNAL_EXAMPLE = """Question: How many albums does journey have?

`Documents`:
I: Guide\" gave each of the band's albums only one star, with Dave Marsh writing that \"Journey was a dead end for San Francisco area rock.\" Marsh later would anoint \"Escape\" as one of the worst number-one albums of all time. Journey's next album, \"Frontiers\" (1983), continued their commercial success, reaching No. 2 on the album charts, selling nearly six million copies. The album generated four Top 40 hits, \"Separate Ways (Worlds Apart)\", which reached No. 8, \"Faithfully\", which reached No. 12, \"Send Her My Love\" and \"After the Fall\", both of which reached No. 23. 
II: inspiration from 1960s-era soul singers such as Aretha Franklin and Gladys Knight, and blending it with blues runs similar to B. B. King. He was influenced by guitarists such as Eric Clapton, Jimi Hendrix, Carlos Santana and Wes Montgomery. In addition to his five solo albums and 14 studio albums with Journey, his work also includes: a pair of albums with keyboardist Jan Hammer, short-term collaborations with Sammy Hagar (HSAS and Planet Us) and Paul Rodgers, stints with Bad English (a supergroup that featured Journey\u2019s Jonathan Cain and Deen Castronovo and Jonathan Cain's former Babys bandmates John Waite and Ricky
III: Greatest Hits (Journey album) Greatest Hits is a compilation album by the American rock band Journey, originally released in 1988 by Columbia Records. It is the band's best-selling career disc, spending 517 weeks on the \"Billboard\" 200 album chart (more than any other compilation album, except for Bob Marley's \"Legend\" disc, in history). Additionally, as of late 2014, it has logged nearly 1,000 weeks on \"Billboard\"'s Catalog albums chart. As of December 2008, it was reported to have been the sixth highest certified 'greatest hits' package in the United States according to the RIAA, behind only similar collections by The

`my knowledge`:
Journey is an American rock band that formed in San Francisco, California in 1973. The band has had a successful career spanning over 40 years, with a total of 15 studio albums, 3 live albums, and 5 compilation albums. Their debut album, \"Journey\", was released in 1975 and was not a commercial success. However, their second album, \"Look into the Future\", released in 1976, showed a significant improvement in sales. The band's most successful album to date is \"Escape\", released in 1981, which included the hit singles \"Who's Crying Now\" and \"Open Arms\". (Confidence: 0.8)
...
American rock band Journey has released 15 studio albums, five live albums, 11 compilation albums, and 52 singles since 1975." (Confidence: 0.8)

Golden Answer: Journey has 15 studio albums.


Output: 
Think step by step:
I need to find the number of all the albums released by Journey, or a list of all the names of the albums and sum them up. However, Document I mentions specific albums such as Escape and Frontiers, but it does not provide a complete count of all Journey's albums. Document II states "14 studio albums with Journey" and mentions additional works like solo albums and collaborations, but it does not directly answer the question. Document III only discusses the compilation album Greatest Hits. As the documents do not provide any supporting evidence for the number of studio albums released by Journey, I will need to rely on my internal knowledge to answer the question. I may know that Journey has released 15 studio albums with 3 live albums and 5 compilation albums, but I am not completely certain, maybe 5 live albums or 11 compilation albums actually, so I should lower the confidence. I will use this information to answer the question.

References:
[1] "The band has had a successful career spanning over 40 years, with a total of 15 studio albums, 3 live albums, and 5 compilation albums. Their debut album, \"Journey\", was released in 1975 and was not a commercial success. However, their second album, \"Look into the Future\", released in 1976, showed a significant improvement in sales. The band's most successful album to date is \"Escape\", released in 1981, which included the hit singles \"Who's Crying Now\" and \"Open Arms\".", Internal Knowledge. (Confidence: 0.8)

Answer: The band, formed in San Francisco in 1973, has released 3 live albums and 5 compilation albums since 1975 [1]. Their debut album, \"Journey,\" was released in 1975 [1].
"""

ABSTENTION_EXAMPLE = """Question: How many albums does journey have?

`Documents`:
I: Guide\" gave each of the band's albums only one star, with Dave Marsh writing that \"Journey was a dead end for San Francisco area rock.\" Marsh later would anoint \"Escape\" as one of the worst number-one albums of all time. Journey's next album, \"Frontiers\" (1983), continued their commercial success, reaching No. 2 on the album charts, selling nearly six million copies. The album generated four Top 40 hits, \"Separate Ways (Worlds Apart)\", which reached No. 8, \"Faithfully\", which reached No. 12, \"Send Her My Love\" and \"After the Fall\", both of which reached No. 23. 
II: inspiration from 1960s-era soul singers such as Aretha Franklin and Gladys Knight, and blending it with blues runs similar to B. B. King. He was influenced by guitarists such as Eric Clapton, Jimi Hendrix, Carlos Santana and Wes Montgomery. In addition to his five solo albums and 14 studio albums with Journey, his work also includes: a pair of albums with keyboardist Jan Hammer, short-term collaborations with Sammy Hagar (HSAS and Planet Us) and Paul Rodgers, stints with Bad English (a supergroup that featured Journey\u2019s Jonathan Cain and Deen Castronovo and Jonathan Cain's former Babys bandmates John Waite and Ricky
III: Greatest Hits (Journey album) Greatest Hits is a compilation album by the American rock band Journey, originally released in 1988 by Columbia Records. It is the band's best-selling career disc, spending 517 weeks on the \"Billboard\" 200 album chart (more than any other compilation album, except for Bob Marley's \"Legend\" disc, in history). Additionally, as of late 2014, it has logged nearly 1,000 weeks on \"Billboard\"'s Catalog albums chart. As of December 2008, it was reported to have been the sixth highest certified 'greatest hits' package in the United States according to the RIAA, behind only similar collections by The

`my knowledge`:
I have no internal knowledge about this question.

Golden Answer: Journey has 15 studio albums.


Output: 
Think step by step:
I need to find the number of all the albums released by Journey, or a list of all the names of the albums and sum them up. However, Document I mentions specific albums such as Escape and Frontiers, but it does not provide a complete count of all Journey's albums. Document II states "14 studio albums with Journey" and mentions additional works like solo albums and collaborations, but it does not directly answer the question. Document III only discusses the compilation album Greatest Hits. As the documents do not provide any supporting evidence for the number of studio albums released by Journey, I will need to rely on my internal knowledge to answer the question. However, I cannot recall the exact number of studio albums released by Journey, so I will abstain from answering the question.

References:
<EMPTY>

Answer: I don't have sufficient knowledge to answer the question, and there is no relevant information in the provided documents to answer the question.
"""

example_dict = {
    'external': EXTERANL_EXAMPLE,
    'internal': INTERNAL_EXAMPLE,
    'abstention': ABSTENTION_EXAMPLE
}

def extract_bracketed_texts(text):
    matches = re.findall(r'\[\d+\]:.*?(?=\n)', text)
    return matches


def fill_prompt_free(question, docs_str, golden_answer, internal_knowledge, tip = '', example = ''):

    if tip:
        tip_message = f"Here is a tip specific to this question: {tip.upper()}. The tip may be not accurate, please use your own judgement.\n\n"
    else:
        tip_message = ''
    p = f"""You are a Large Language Model with limited knowlege. Given a question, documents, "my knowledge" and an golden answer, please generate a high-quality answer with citation. You should simulate a Large Language Model that think step by step and output references and an answer using the provided documents and "Knwoledge in Yourself" (in the "my knowledge section"), but simulate that you cannot see the golden answer. Simulate that you are generating the knowledge yourself, not refering the "my knowledge" section and the gold answer. 

The response needs to follow the following requirements:
1. Your answer should contain all the information in the golden answer provided (i.e., the golden answer is a subset of your full answer).
2. each statement in your answer should be cited properly, with marks like [1], [2] to indicate the source of the information. When multiple sources are available, cite a munimum set.
3. Your answer should be concise and contain supporting evidence from the documents provided.

Think step by step to generate the full answer, by considering the provided `Documents`, `my knowledge>`and the golden answer. Here is a guidance:
1. Analyze what kind of knowledge you need to answer the question, and try to find supporting evidence in the documents.
2. Use provided `Documents` first, and if the information is not enough, use "my knowledge" for supplement. Scrutinize all the possible "my knowledge" and give an appropriate confidence level according to all the possible "my knowledge".
3. Only use `my knowledge` when provided `Documents` are not sufficient. You don't need to use "my knowledge" for comfirming the information in the provided documents or other unnecessary situations.
4. You pretend to be a Large Language Model with limited knowlege, so you can only use the given documents and "my knowledge" to generate the answer. When using "my knowledge", pretend that you are using the knowledge that you have generated yourself. When thinking about my knowledges, use appropriate uncertainty words to indicate the "Confidence provided at the end of "my knowledge", and use 'Internal Knowledge' to mark the source of the knowledge.
5. When citing the provided documents, you should select a fine-grained span from the documents, and ensure the spans is credible and less redundant. Use Roman numerals to mark the document and use Arabic numerals to mark spans. Use 'Document I' to refer to the first document, and so on.
6. Cite spans using Arabic numerals like [1]. Do not use Roman numerals to cite spans.
7. When using "my knowledge", you should generate a more credible and less redundant version of the knowledge, and use Arabic numerals to mark the spans, and output the provided confidence in the last.
8. If none of "my knowledge" is available, admit it honestly and say that is because of your limited capabilities.
9. If none of the documents and "my knowledge" are relevant to the question, you should still output the steps and an empty reference and then generate an abstention response: "I don't have sufficient knowledge to answer the question, and there is no relevant information in the provided documents to answer the question" with an empty reference.

Here is an example:

Exapmple :

{example}


You have to follow the instruction to generate the full answer for the question below:

{tip_message}
Question: {question}

`Documents`:
{docs_str}

`my knowledge`: 
{internal_knowledge}

Golden Answer: {golden_answer}

Output:
"""
    return p.format(example, tip_message, question, docs_str, internal_knowledge, golden_answer)

def fill_item(item, free = False):
    def _list_to_str(lst, with_index = False, number = 'Arabic', start = 1, delete = False, default = 'No Information'):
        if delete:
            # delete [x] at the beginning of the string
            lst = [re.sub(r'^\[\d+\]: ', '', doc) for doc in lst]
        lst = [lst.split('\n')[0] for lst in lst]
        Roman = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
        if not lst:
            return default
        if with_index:
            if number == 'Arabic':
                return '\n'.join([f"[{i + start}]: {doc}" for i, doc in enumerate(lst)])
            elif number == 'Roman':
                return '\n'.join([f"{Roman[i + start - 1]}: {doc}" for i, doc in enumerate(lst)])
        else:
            return '\n'.join(lst)
    question = item['question']
    if 'docs' in item:
        docs = [doc['text'] for doc in item['docs']][:5]
    else:
        docs = [item[f'doc_{i+1}_doc'] for i in range(5)]
    docs_str = _list_to_str(docs, with_index = True, number='Roman')
    internal_passages = item["inner_passages"]
    confidence = item["p_conf"]
    internal_passages = [f"{passage['text']} (Confidence: {confidence})" for passage in internal_passages]
    if confidence < 0.01:
        internal_passages = []
    internal_knowledge_str = _list_to_str(internal_passages, with_index = False, delete = True, default='I have no internal knowledge about this question.')
    
    ex_gold = any([doc['gold'] for doc in item['docs']])
    in_gold = any([doc['gold'] for doc in item['inner_passages']])
    if ex_gold:
        tip = 'external'
    elif in_gold:
        tip = 'internal'
    else:
        tip = 'abstention'

    golden_answer = item['answer']
    return fill_prompt_free(question, docs_str, golden_answer, internal_knowledge_str, tip_dict[tip], example_dict[tip])

def load_json(file_path):
    return json.load(open(file_path, 'r'))


def main():
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='DPO/crag_300.json')
    argparser.add_argument('--output', type=str, default='_res.json')
    argparser.add_argument('--total', type=int, default=1000)
    args = argparser.parse_args()


    llm_args = {
        'openai_api': True,
        'azure': False,
        'max_tokens': 100,
        'model': 'gpt-4o',
        'temperature': 0.5,
        'top_p': 0.9,
    }
    llm_args = argparse.Namespace(**llm_args)

    gpt_4o = LLM(args=llm_args)

    data = load_json(args.dataset)[0:args.total]
    for i, item in tqdm(list(enumerate(data))):
        try:
            prompt = fill_item(item)
            if i == 0:
                # print example
                print('EXAMPLE PROMPT FOR GENERATION')
                print(prompt)
            answer = gpt_4o.generate(prompt, 1024)
            item['oracle'] = answer
        except Exception as e:
            print(f"Error in item {i}")
            item['oracle'] = str(e)
        if i % 20 == 0:
            print(f"Processed {i} items")
            json.dump(data[:i+1], open(args.output, 'w'), indent=4)
    json.dump(data, open(args.output, 'w'), indent=4)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='final_data/inner_confidence+.json')
    argparser.add_argument('--output', type=str, default='_res.json')
    argparser.add_argument('--total', type=int, default=1500)
    argparser.add_argument('--max_sample', type=int, default=1)
    args = argparser.parse_args()


    llm_args = {
        'openai_api': True,
        'azure': False,
        'max_tokens': 100,
        'model': 'gpt-4o',
        'temperature': 0.5,
        'top_p': 0.9,
    }
    print('start')
    llm_args = argparse.Namespace(**llm_args)

    gpt_4o = LLM(args=llm_args)

    data = load_json(args.dataset)[0:args.total]
    # existing output
    import os

    if os.path.exists(args.output):
        print('loading previous')
        out = load_json(args.output)
        previous = 0
        for o, d in zip(out, data):
            previous += 1
            d['oracle'] = o['oracle']
        print(f"Previous {previous} items")
    for i, item in tqdm(list(enumerate(data))):
        if i % 20 == 0:
            print(f"Processed {i} items")
            json.dump(data[:i+1], open(args.output, 'w'), indent=4)
        if 'oracle' in item and item['oracle']:
            continue
        try:
            prompt = fill_item(item, free = True)
            print(prompt)
            answers = [gpt_4o.generate(prompt, 1024) for i in range(args.max_sample)]
            
            item['oracle'] = answers
        except Exception as e:
            traceback.print_exc()
            print(f"Error in item {i}")
            item['oracle'] = str(e)

        
    json.dump(data, open(args.output, 'w'), indent=4)
