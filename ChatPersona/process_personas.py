from datasets import load_dataset
import pandas as pd
import os
import csv
from main_persona import run_persona_pipeline, PersonaChatModel
import evaluate
import ast

def preprocess_persona(persona_list):
    processed_sentences = []
    for sentence in persona_list:
        sentence = sentence.strip().lower()
        if not sentence.endswith('.'):
            sentence += '.'
        processed_sentences.append(sentence)
    return ' '.join(processed_sentences)

def main():
    model_instance = PersonaChatModel()
    
    dataset = load_dataset('AlekseyKorshuk/persona-chat', split='validation')
    output_csv = 'persona_reconstruction_results.csv'
    fieldnames = [
        'persona_id', 'ground_truth', 'baseline_persona', 'initial_persona', 'final_persona',
        'rouge1_baseline', 'rouge2_baseline', 'rougeL_baseline',
        'rouge1_initial', 'rouge2_initial', 'rougeL_initial',
        'rouge1_final', 'rouge2_final', 'rougeL_final'
    ]
    
    file_exists = os.path.isfile(output_csv)
    
    if not file_exists:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    # Load the ROUGE metric
    rouge = evaluate.load('rouge')
    
    for idx, data in enumerate(dataset):
        persona = data['personality']
        try:
            baseline_persona, initial_persona, final_persona = run_persona_pipeline(persona, model_instance)
        except Exception as e:
            print(f"An error occurred with persona_id {idx}: {e}")
            continue
    
        # Preprocess the ground truth persona
        ground_truth_text = preprocess_persona(persona)
        
        # Parse and preprocess the baseline reconstructed persona
        try:
            baseline_persona_list = ast.literal_eval(baseline_persona)
            baseline_persona_text = preprocess_persona(baseline_persona_list)
            # Compute ROUGE scores for baseline
            scores_baseline = rouge.compute(predictions=[baseline_persona_text], references=[ground_truth_text], use_stemmer=True)
        except Exception as e:
            print(f"An error occurred while parsing baseline_persona for persona_id {idx}: {e}")
            scores_baseline = {'rouge1': None, 'rouge2': None, 'rougeL': None}
        
        # Parse and preprocess the initial reconstructed persona
        try:
            initial_persona_list = ast.literal_eval(initial_persona)
            initial_persona_text = preprocess_persona(initial_persona_list)
            # Compute ROUGE scores for initial persona
            scores_initial = rouge.compute(predictions=[initial_persona_text], references=[ground_truth_text], use_stemmer=True)
        except Exception as e:
            print(f"An error occurred while parsing initial_persona for persona_id {idx}: {e}")
            scores_initial = {'rouge1': None, 'rouge2': None, 'rougeL': None}
        
        # Parse and preprocess the final reconstructed persona
        try:
            final_persona_list = ast.literal_eval(final_persona)
            final_persona_text = preprocess_persona(final_persona_list)
            # Compute ROUGE scores for final persona
            scores_final = rouge.compute(predictions=[final_persona_text], references=[ground_truth_text], use_stemmer=True)
        except Exception as e:
            print(f"An error occurred while parsing final_persona for persona_id {idx}: {e}")
            scores_final = {'rouge1': None, 'rouge2': None, 'rougeL': None}
        
        row = {
            'persona_id': idx,
            'ground_truth': ' | '.join(persona),  
            'baseline_persona': baseline_persona.replace('\n', ' '), 
            'initial_persona': initial_persona.replace('\n', ' '),
            'final_persona': final_persona.replace('\n', ' '),
            'rouge1_baseline': scores_baseline['rouge1'],
            'rouge2_baseline': scores_baseline['rouge2'],
            'rougeL_baseline': scores_baseline['rougeL'],
            'rouge1_initial': scores_initial['rouge1'],
            'rouge2_initial': scores_initial['rouge2'],
            'rougeL_initial': scores_initial['rougeL'],
            'rouge1_final': scores_final['rouge1'],
            'rouge2_final': scores_final['rouge2'],
            'rougeL_final': scores_final['rougeL'],
        }
        
        with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)
        
        print(f"Processed persona_id: {idx}, ROUGE-1 Final: {scores_final['rouge1']}")
    
    print("Processing complete.")

if __name__ == "__main__":
    main()