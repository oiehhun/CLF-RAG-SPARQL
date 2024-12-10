import json
import torch
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils import prompt_building, rait_generate_sparql, base_generate_sparql, wikidata_query_service, sparql_convert


def parse_arguments():
    parser = argparse.ArgumentParser(description= "RAG-SPARQL")
    parser.add_argument('--cuda_no', type=int, default=0)
    parser.add_argument('--task', type=str, default='GT', help='GT, linking')
    parser.add_argument('--mode', type=int, default=2, help='0 :no cluster, 1 :1cluster, 2 :2cluster')
    parser.add_argument('--llm_model', type=str, default='rait', help='rait, base, chat')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--max_length', type=int, default=700)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./test/lcquad2_GT_2cluster_rait_result.json')
    return parser.parse_args()


def main():
    
    args = parse_arguments()
    device = f'cuda:{args.cuda_no}' if torch.cuda.is_available() else 'cpu'
    model_paths = {'base': './hf_model/models--meta-llama--CodeLlama-7b-Instruct-hf/snapshots/4ce0c40b2ea823bd1d8f1f3fd5bc8a7e80d749bc',
                   'rait': './llama_finetuning/rait-codellama2-7b-it'}
    
    ### Model Load ###
    if args.llm_model != 'chat':
        llm_tokenizer = AutoTokenizer.from_pretrained(model_paths['base'])
        llm_model = AutoModelForCausalLM.from_pretrained(model_paths[args.llm_model], device_map=device, torch_dtype=torch.bfloat16)
        llm_tokenizer.padding_side = 'right'
        llm_tokenizer.pad_token = llm_tokenizer.eos_token 

        pipe = pipeline(
            "text-generation",
            model=llm_model,
            tokenizer=llm_tokenizer,
            use_cache=True,
            device_map="auto",    
            do_sample=True,
            temperature=args.temperature,
            max_length=args.max_length,
            repetition_penalty=args.repetition_penalty,
            num_return_sequences=args.num_return_sequences,
            eos_token_id=llm_tokenizer.eos_token_id,
            pad_token_id=llm_tokenizer.eos_token_id)
    
    ### Test ###
    with open('./data/LC-QuAD2.0/final_test_lcquad2_linking.json') as f:
        test_data = json.load(f)
        
    test_data = [x for x in test_data if x['answer'] != []] # remove empty answer : (6021 -> 5030)
        
    for i in tqdm(range(len(test_data))):
        question = test_data[i]['question']
        if args.task == 'GT':
            entities = test_data[i]['new_LabelsEnt']
            relations = test_data[i]['new_LabelsRel']
        elif args.task == 'linking':
            entities = test_data[i]['new_SpacyEnt']
            relations = test_data[i]['new_FalconRel']
        gold_sparql = test_data[i]['sparql_wikidata']
        gold_answer = test_data[i]['answer']
        if args.llm_model == 'rait' or args.llm_model == 'base':
            prompt = prompt_building(question, entities, relations, args.mode)
            print('\n[prompt]:', prompt)
            if args.llm_model == 'rait':
                sparql = rait_generate_sparql(prompt, pipe)
            elif args.llm_model == 'base':
                sparql = base_generate_sparql(prompt, pipe)
        elif args.llm_model == 'chat':
            sparql = sparql_convert(question, entities, relations, args.mode)
        test_data[i]['pred_sparql'] = sparql
        print('[pred_sparql]:', sparql)
        print('[gold_sparql]:', gold_sparql)
        
        result = wikidata_query_service(sparql)
        test_data[i]['pred_answer'] = result
        print('[pred_answer]:', result)
        print('[gold_answer]:', gold_answer)
        print('='*100)
    
    with open(args.save_path, 'w') as f:
        json.dump(test_data, f, indent=4)

if __name__ == '__main__':
    main()