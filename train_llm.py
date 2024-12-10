import json
import pandas as pd
import torch
import pickle
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from torch.nn.functional import cosine_similarity


with open('./data/LC-QUAD2.0/final_train_lcquad2.json') as f:
    train_data = json.load(f)
    
with open('./data/embedding/lcquad_base/train_question_embedding.pkl', 'rb') as f:
    embedding_space = pickle.load(f)
    
emb_path = 'BAAI/bge-m3'
emb_tokenizer = AutoTokenizer.from_pretrained(emb_path)
emb_model = AutoModel.from_pretrained(emb_path)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
emb_model.to(device)

clf_path = './bert_finetuning/best_model'
clf_tokenizer = AutoTokenizer.from_pretrained(clf_path)
clf_model = AutoModelForSequenceClassification.from_pretrained(clf_path, num_labels=13)
clf_model.to(device)
clf_model.eval()


def get_prediction(query): # 1 cluster
    inputs = clf_tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clf_model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax().item()
    return predicted_class_idx


def get_embeddings(texts):
    inputs = emb_tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(device)  # Move inputs to GPU
    with torch.no_grad():
        outputs = emb_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1) 
    return embeddings


def search(query, es_num, top_k=3):
    with open(f'./data/embedding/cluster/embedding_space_{es_num}.pkl', 'rb') as f:
        embedding_space = pickle.load(f)

    embedding_space_tensor = {key: torch.tensor(embedding).to(device) for key, embedding in embedding_space.items()}
    query_embedding = get_embeddings(query)

    similarities = []
    
    for key, embedding in embedding_space_tensor.items():
        sim = cosine_similarity(query_embedding, embedding.unsqueeze(0))  # Use unsqueeze to match dimensions
        similarities.append((key, sim.item()))  # Convert to scalar
    
    result = sorted(similarities, key=lambda x: x[1], reverse=True)[1:top_k]
    example = ''
    for k, s in result:
        k = k.split('\t')
        example += f'\n[question]: {k[0]}\n[entities]: {k[-3]}\n[relations]: {k[-2]}\n[sparql]: {k[-1]}\n'
    return example


def prompt_building(question, entities, relations, sparql):
    prompt = f'''
    <s>[INST]
    Task: convert question to SPARQL query for 'Wikidata' knowledge graph.
    Description: given an input question and a list of 'Wikidata' ID for the mentioned entities in the question and relations mentioned in the question.
    Write the correct SPARQL code to query these ‘Wikidata’ IDs in the ‘Wikidata’ knowledge graph.
    You can formulate your SPARQL query as the following examples.
    
    <examples>
    {str(search(question, get_prediction(question)))}
    </examples>
    
    [question]: {question}
    [entities]: {entities}
    [relations]: {relations}[/INST]
    
    [sparql]: {sparql}
    </s>
    '''
    return prompt

def generate_prompt(data):
    prompts = []
    for i in range(len(data)):
        question = data['question'][i]
        entities = data['new_LabelsEnt'][i]
        relations = data['new_LabelsRel'][i]
        sparql = data['sparql_wikidata'][i]
        prompt = prompt_building(question, entities, relations, sparql)
        prompts.append(prompt)
    return prompts


### Model Load
BASE_MODEL = './hf_model/models--meta-llama--CodeLlama-7b-Instruct-hf/snapshots/4ce0c40b2ea823bd1d8f1f3fd5bc8a7e80d749bc'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL,
                                             quantization_config=bnb_config,
                                             torch_dtype = torch.bfloat16,
                                             device_map=device)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

tokenizer.padding_side = 'right' 
tokenizer.pad_token = tokenizer.eos_token 
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.eos_token_id


lora_config = LoraConfig(
    r=16,
    lora_alpha = 16,
    lora_dropout = 0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)


### Data Load
train_df = pd.read_csv('./data/LC-QUAD2.0/final_train_lcquad2_cluster_over_rait.csv')
train_df = train_df[train_df['label'].isin([4, 6, 7, 8, 9])].copy() # Different types of clusters

### Data Preprocessing
train_df['question_len'] = train_df['question'].apply(lambda x: len(str(x)))
train_df = train_df[train_df['question_len'] > 5]
train_df = train_df[['question', 'new_LabelsEnt', 'new_LabelsRel', 'sparql_wikidata']]

with open('./data/LC-QUAD2.0/final_test_lcquad2.json', 'r') as f:
    data = json.load(f)
test_df = pd.DataFrame(data)
test_df = test_df[['question', 'new_LabelsEnt', 'new_LabelsRel', 'sparql_wikidata']]

train_datasets = Dataset.from_pandas(train_df, preserve_index=False)
test_datasets = Dataset.from_pandas(test_df, preserve_index=False)
datasets = DatasetDict({'train': train_datasets, 'test': test_datasets})
train_data = datasets['train']
test_data = datasets['test']


### Training
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    max_seq_length=700,
    args=TrainingArguments(
        output_dir="rait-codellama2-outputs",
        max_steps=3000,  
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,  
        optim="paged_adamw_8bit",
        warmup_steps=80,  
        learning_rate=2e-4,
        fp16=True,
        logging_steps=100,
        push_to_hub=False,
        report_to='none',
        save_total_limit=2,
    ),
    peft_config=lora_config,
    formatting_func=generate_prompt,
)

trainer.train()

### Model Save
ADAPTER_MODEL = "rait-codellama2-adapter-it"
trainer.model.save_pretrained(ADAPTER_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
peft_model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL, torch_dtype=torch.bfloat16)

merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained('rait-codellama2-7b-it')

### Model Load
FINETUNE_MODEL = './rait-codellama2-7b-it'

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
finetune_model = AutoModelForCausalLM.from_pretrained(FINETUNE_MODEL, device_map=device, torch_dtype=torch.bfloat16)