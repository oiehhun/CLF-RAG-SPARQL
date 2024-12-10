import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import cosine_similarity
from SPARQLWrapper import SPARQLWrapper, JSON
from pandas import json_normalize
import pickle
import warnings
warnings.filterwarnings("ignore")
from langchain_openai import ChatOpenAI
from langchain.prompts import (ChatPromptTemplate,FewShotChatMessagePromptTemplate)
from dotenv import load_dotenv
load_dotenv()



wiki_sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'

model_paths = {
    'emb': 'BAAI/bge-m3',
    'clf': './bert_finetuning/best_model'
}

emb_tokenizer = AutoTokenizer.from_pretrained(model_paths['emb'])
emb_model = AutoModel.from_pretrained(model_paths['emb'])
emb_model.to(device)

clf_tokenizer = AutoTokenizer.from_pretrained(model_paths['clf'])
clf_model = AutoModelForSequenceClassification.from_pretrained(model_paths['clf'], num_labels=13)
clf_model.to(device)
clf_model.eval()


def get_prediction(query, mode): 
    inputs = clf_tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clf_model(**inputs)
    logits = outputs.logits
    if mode == 1: # 1 cluster
        predicted_class_idx = logits.argmax().item()
    elif mode == 2: # 2 cluster
        top2 = torch.topk(logits, 2)  
        predicted_class_idx = top2.indices.squeeze().tolist()
    elif mode == 0:
        return None
    return predicted_class_idx


def search(query, es_num, mode, top_k=1):  
    example = ''

    if mode == 2:  # 2 cluster, 2 shot
        for i in es_num:
            with open(f'./data/embedding/cluster/embedding_space_{i}.pkl', 'rb') as f:
                embedding_space = pickle.load(f)
            
            embedding_space_tensor = {key: torch.tensor(embedding).to(device) for key, embedding in embedding_space.items()}
            query_embedding = get_embeddings(query)

            similarities = []
            
            for key, embedding in embedding_space_tensor.items():
                sim = cosine_similarity(query_embedding, embedding.unsqueeze(0))  # Use unsqueeze to match dimensions
                similarities.append((key, sim.item()))  # Convert to scalar
            
            result = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
            
            for k, s in result:
                k = k.split('\t')
                example += f'\n[question]: {k[0]}\n[entities]: {k[-3]}\n[relations]: {k[-2]}\n[sparql]: {k[-1]}\n'
    else:
        if mode == 0 : # no cluster, 2 shot
            with open('./data/embedding/base/train_question_embedding.pkl', 'rb') as f:
                embedding_space = pickle.load(f)

        elif mode == 1:  # 1 cluster, 2 shot
            with open(f'./data/embedding/cluster/embedding_space_{es_num}.pkl', 'rb') as f:
                embedding_space = pickle.load(f)
            
        embedding_space_tensor = {key: torch.tensor(embedding).to(device) for key, embedding in embedding_space.items()}
        query_embedding = get_embeddings(query)

        similarities = []
        
        for key, embedding in embedding_space_tensor.items():
            sim = cosine_similarity(query_embedding, embedding.unsqueeze(0))  # Use unsqueeze to match dimensions
            similarities.append((key, sim.item()))  # Convert to scalar
        
        result = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k+1]
        
        for k, s in result:
            k = k.split('\t')
            example += f'\n[question]: {k[0]}\n[entities]: {k[-3]}\n[relations]: {k[-2]}\n[sparql]: {k[-1]}\n'

    return example


def get_embeddings(texts):
    inputs = emb_tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(device) 
    with torch.no_grad():
        outputs = emb_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


def prompt_building(question, entities, relations, mode):
    if mode == 0:
        examples = search(question, es_num=None, mode=mode)
    else:
        examples = search(question, es_num=get_prediction(question, mode), mode=mode)
    
    prompt = f'''
    <s>[INST]
    Task: convert question to SPARQL query for 'Wikidata' knowledge graph.
    Description: given an input question and a list of 'Wikidata' ID for the mentioned entities in the question and relations mentioned in the question.
    Write the correct SPARQL code to query these ‘Wikidata’ IDs in the ‘Wikidata’ knowledge graph.
    You can formulate your SPARQL query as the following examples.
    
    <examples>
    {str(examples)}
    </examples>
    
    [question]: {question}
    [entities]: {entities}
    [relations]: {relations}[/INST]
    '''
    return prompt


def rait_generate_sparql(prompt, pipe):
    response = pipe(prompt)[0]['generated_text']
    try:
        generated_text = response.split('[/INST]')[1].split('[sparql]:')[1].split('[/INST]')[0].split('\n')[0]
    except:
        generated_text = 'Error'
    return generated_text.strip()


def base_generate_sparql(prompt, pipe):
    response = pipe(prompt)[0]['generated_text']
    generated_text = response.split('[sparql]:')[-1]
    return generated_text.strip()


def wikidata_query_service(sparql):
    
    result = []
    try:
        if sparql.upper().startswith("SELECT"):
            wiki_sparql.setQuery(sparql)
            wiki_sparql.setReturnFormat(JSON)
            results = wiki_sparql.query().convert()
            results_df = json_normalize(results['results']['bindings'])
            for col in results_df.columns:
                if '.value' in col:
                    result.extend(results_df[col].tolist())
        elif sparql.upper().startswith("ASK"):
            wiki_sparql.setQuery(sparql)
            wiki_sparql.setReturnFormat(JSON)
            results = wiki_sparql.query().convert()
            result = results['boolean']
    except:
        result = ["Error"]
    
    return result


def example_convert(example):
    ex = []
    example = example.split('\n\n')
    for e in example:
        e = e.split('\n')
        e = [i for i in e if i != '']
        ex.append({
            "question": e[0].split(']:')[-1].strip(),
            "entities": e[1].split(']:')[-1].strip(),
            "relations": e[2].split(']:')[-1].strip(),
            "sparql": e[3].split(']:')[-1].strip()
        })
    return ex


def sparql_convert(question, entities, relations, mode):
    chat_model = ChatOpenAI(temperature=0.1, model_name = 'gpt-4o')
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "question: {question}\nentities: {entities}\nrelations: {relations}\n"),
            ("ai", "sparql: {sparql}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=example_convert(search(question, get_prediction(question, mode), mode)),
    )
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", '''
            Task: convert question to SPARQL query for 'Wikidata' knowledge graph.
            Description: given an input question and a list of 'Wikidata' ID for the mentioned entities in the question and relations mentioned in the question.
            Write the correct SPARQL code to query these ‘Wikidata’ IDs in the ‘Wikidata’ knowledge graph.
            You can formulate your SPARQL query as the following examples.
            '''),
            few_shot_prompt,
            ("human", "question: {question}\nentities: {entities}\nrelations: {relations}\n")
        ]
    )
    chain = final_prompt | chat_model
    sparql = chain.invoke({"question": question, "entities": entities, "relations": relations})
    sparql = sparql.content.split("sparql:")[-1].strip()
    return sparql


def exact_match(row):
    if isinstance(row['answer'], bool) or isinstance(row['pred_answer'], bool):
        if row['answer'] == row['pred_answer']:
            return 1
        else:
            return 0
    else:
        pred = set(row['pred_answer'])
        gold = set(row['answer'])
        if gold == pred:
            return 1
        else:
            return 0
        

def f1_score(row):
    if isinstance(row['answer'], bool) or isinstance(row['pred_answer'], bool):
        if row['answer'] == row['pred_answer']:
            return 1
        else:
            return 0
    else:
        gold = set(row['answer'])
        pred = set(row['pred_answer'])
        if len(pred) == 0:
            return 0
        precision = len(gold & pred) / len(pred)
        recall = len(gold & pred) / len(gold)
        if precision + recall == 0:
            return 0
        f1 = 2 * precision * recall / (precision + recall)
        return f1