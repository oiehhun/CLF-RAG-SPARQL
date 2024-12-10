import spacy
import requests


def entity_linking(question):
    
    """This function takes NL question and KB and returns the linked entites list"""
    nlp = spacy.load("en_core_web_lg")
    entities = []
    nlp.add_pipe("entityLinker", last=True)
    doc = nlp(question)
    all_linked_entities = doc._.linkedEntities
    for i in range(len(all_linked_entities)):
        url = all_linked_entities[i].get_url()
        entity_id = url.split('/')[-1]
        entity_text = all_linked_entities[i].get_label()
        entities.append(entity_text + ': wd:' + entity_id)
      
    return entities


def execute_sparql_query(label):
    """This function takes label generated from REBEL and return its wikidata identfier"""
    #label = f'"{label}"'
    endpoint_url = "https://query.wikidata.org/sparql"
    sparql_query = """
    SELECT ?item ?itemLabel
    WHERE
    {
      ?item rdfs:label \"""" + label + """\"@en.
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    LIMIT 1
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'}

    params = {
        'query': sparql_query,
        'format': 'json'
    }
    
    relations = []
    response = requests.get(endpoint_url, headers=headers, params=params)
    data = response.json()
    results = data['results']['bindings']
    for result in results:
        relation_id = result['item']['value'].split('/')[-1]
        relation_text = result['itemLabel']['value']
        relations = relation_text + ': wdt:' + relation_id
    
    return relations


def get_wikidata_label(id):
    endpoint_url = "https://query.wikidata.org/sparql"
    
    # Corrected SPARQL query to use the actual entity URI
    sparql_query = f"""
    SELECT ?itemLabel
    WHERE 
    {{
      wd:{id} rdfs:label ?itemLabel.
      FILTER(LANG(?itemLabel) = "en")
    }}
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'
    }
    params = {
        'query': sparql_query,
        'format': 'json'
    }
    
    response = requests.get(endpoint_url, headers=headers, params=params)
    data = response.json()
    results = data['results']['bindings']
    
    # Extract the label
    if results:
        return results[0]['itemLabel']['value']
    else:
        return None
    
    
def relation_extraction(question):
    
    """This function takes NL question and KB and returns the relations found in the question"""
   
    rel_list = []
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe("rebel", after="senter", config={'device':0, 'model_name':'Babelscape/rebel-large'})
    doc = nlp(question)
    for value, rel_dict in doc._.rel.items():
        relation =  str(execute_sparql_query(rel_dict['relation']))
        rel_list.append(relation)
    return rel_list