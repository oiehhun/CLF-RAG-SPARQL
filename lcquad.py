import json
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
from pandas import json_normalize
from tqdm import tqdm
from linking import entity_linking, get_wikidata_label, relation_extraction


### ========== final_train_lcquad2.json, final_test_lcquad2.json ========== ###

### token_id_list ###
with open("./data/LC-QuAD2.0/train.json", "r") as f:
    train_data = json.load(f)
    
lcquad_train = []

for data in train_data:
    dic = {}
    dic['uid'] = data['uid']
    dic['question'] = data['question']
    dic['sparql_wikidata'] = data['sparql_wikidata'].strip()
    dic['template_id'] = data['template_id']
    sparql = data['sparql_wikidata'].split(' ')
    token_id_list = []
    for s in sparql:
        if '}' in s:
            s = s.replace('}', '')
        if 'Q' in s:
            token_id_list.append(s)
        elif 'P' in s:
            token_id_list.append(s)
        else:
            continue
    dic['token_id_list'] = token_id_list
    
    lcquad_train.append(dic)
    print("completed")
    
with open("./data/LC-QuAD2.0/train_lcquad2.json", "w") as f:
    json.dump(lcquad_train, f, indent=4)
    
    
### entities, relations, new_LabelsEnt, new_LabelsRel ###
with open("./data/LC-QuAD2.0/train_lcquad2.json", "r") as f:
    lcquad_train = json.load(f)

for data in lcquad_train:
    token_id = data['token_id_list']
    entities = []
    relations = []
    new_LabelsEnt = []
    new_LabelsRel = []
    
    error = []
    for token in token_id:
        try:
            if 'Q' in token:
                qid = token.split(':')[-1]
                label = get_wikidata_label(qid)
                entities.append(label)
                if label:
                    new_LabelsEnt.append(label + ': ' + token)
                else:
                    error.append(token)
            else:
                pid = token.split(':')[-1]
                label = get_wikidata_label(pid)
                relations.append(label)
                if label:
                    new_LabelsRel.append(label + ': ' + token)
                else:
                    error.append(token)
        except:
            error.append(token)
            continue
                
    data['entities'] = entities
    data['relations'] = relations
    data['new_LabelsEnt'] = new_LabelsEnt
    data['new_LabelsRel'] = new_LabelsRel
    print(data['question'])
    print(data['sparql_wikidata'])
    print(f'token_id: {token_id}')
    print(f'entities: {entities}')
    print(f'relations: {relations}')
    print(f'new_LabelsEnt: {new_LabelsEnt}')
    print(f'new_LabelsRel: {new_LabelsRel}')
    print()

with open("./data/LC-QuAD2.0/train_lcquad2.json", "w") as f:
    json.dump(lcquad_train, f, indent=4)


### answer ###
with open("./data/LC-QuAD2.0/train_lcquad2.json", "r") as f:
    lcquad_train = json.load(f)

wiki_sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

for data in lcquad_train:
    sparql = data['sparql_wikidata'].strip()
    
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
    
    data['answer'] = result
    print(f"answer : {data['answer']}")
    print()

with open("./data/LC-QuAD2.0/final_train_lcquad2.json", "w") as f:
    json.dump(lcquad_train, f, indent=4)
    

### ========== final_test_lcquad2_link.json  ========== ###   

### Entity,Relation Linking ###
with open("./data/LC-QuAD2.0/final_test_lcquad2.json", "r") as f:
    lcquad_test = json.load(f)

for i in tqdm(range(len(lcquad_test))):
    question = lcquad_test[i]['question']
    entities = entity_linking(question) # spacy
    relations = relation_extraction(question) # rebel
    lcquad_test[i]['link_entities'] = entities
    lcquad_test[i]['link_relations'] = relations
    
    print(f"question : {question}")
    print(f"link_entities : {entities}")
    print(f"link_relations : {relations}")
    print('='*100)
    
with open("./data/LC-QuAD2.0/final_test_lcquad2_linking.json", "w") as f:
    json.dump(lcquad_test, f, indent=4)
    
    
# Load the Falcon dataset
falcon_data = pd.read_csv('./data/LC-QuAD2.0/falcon_lcquad2.csv') # falcon
# Load the LC-QUAD dataset
with open('./data/LC-QuAD2.0/final_test_lcquad2_linking.json', 'r') as f:
    lcquad_data = json.load(f)
    
for i in range(len(lcquad_data)):
    lcquad_data[i]['falcon_entities'] = []
    lcquad_data[i]['falcon_relations'] = []

# Falcon dataset, LC-QAUD dataset Mapping
print('Mapping Falcon dataset to LC-QUAD dataset...')
for i in tqdm(range(len(lcquad_data))):
    for j in range(len(falcon_data)):
        if str(lcquad_data[i]['question']).strip().lower() == str(falcon_data['Question'][j]).strip().lower():
            lcquad_data[i]['falcon_entities'] = eval(falcon_data['FALCON_Entities'][j])
            lcquad_data[i]['falcon_relations'] = eval(falcon_data['FALCON_Relations'][j])
print('completed!')


# Get the Wikidata label for each entity and relation in the LC-QUAD dataset
print('Getting Wikidata labels for Falcon entities and relations...')
for i in tqdm(range(len(lcquad_data))):
    id_list = []
    id_list.extend(lcquad_data[i]['falcon_entities'])
    id_list.extend(lcquad_data[i]['falcon_relations'])
    token_id = lcquad_data[i]['token_id_list']
    token_id_list = [token.split(':')[-1].strip() for token in token_id]
    
    new_FalconEnt = []
    new_FalconRel = []
    
    error = []
    for id in id_list:
        try:
            if 'Q' in id:
                label = get_wikidata_label(id)
                if label:
                    new_FalconEnt.append(label + ': ' + 'wd:'+id)
                else:
                    error.append(id)
            else:
                label = get_wikidata_label(id)
                if label:
                    if id in token_id_list:
                        for token in token_id:
                            if id in token:
                                new_FalconRel.append(label + ': ' + token)
                    else:
                        new_FalconRel.append(label + ': ' + 'wdt:'+id)
                else:
                    error.append(id)
        except:
            error.append(id)
            continue
    print(f'ori_token_id: {token_id}')
    print(f'ori_FalconEnt: {lcquad_data[i]["falcon_entities"]}')
    print(f'ori_FalconRel: {lcquad_data[i]["falcon_relations"]}')
    print(f'new_FalconEnt: {new_FalconEnt}')
    print(f'new_FalconRel: {new_FalconRel}')
    print()
    lcquad_data[i]['new_FalconEnt'] = new_FalconEnt
    lcquad_data[i]['new_FalconRel'] = new_FalconRel
print('completed!')
    

# Save the updated LC-QUAD dataset
print('Saving the updated LC-QUAD dataset...')
with open('./data/LC-QuAD2.0/final_test_lcquad2_linking.json', 'w') as f:
    json.dump(lcquad_data, f, indent=4)
print('completed!')
    