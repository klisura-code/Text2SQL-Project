import pandas as pd
import os
import csv
import sys
import ast
import evaluate  # Using the evaluate library as specified

def preprocess_persona(persona_text):
    """Preprocesses a persona string by lowercasing and ensuring punctuation."""
    persona_list = persona_text.split(' | ')  # Split ground_truth by '|'
    processed_sentences = []
    for sentence in persona_list:
        sentence = sentence.strip().lower()
        if not sentence.endswith('.'):
            sentence += '.'
        processed_sentences.append(sentence)
    return ' '.join(processed_sentences)

def main():
    # Check for command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <dataset.csv>")
        return
    
    # Load the input CSV dataset from the argument
    input_csv = sys.argv[1]
    if not os.path.isfile(input_csv):
        print(f"File {input_csv} does not exist.")
        return
    
    # Load the dataset using pandas
    dataset = pd.read_csv(input_csv)
    
    # Output results file
    output_csv = 'results.csv'
    
    # Load the ROUGE metric using evaluate.load('rouge')
    rouge = evaluate.load('rouge')
    
    all_ground_truth = []
    all_baseline_persona = []
    all_initial_persona = []
    all_final_persona = []
    
    # Iterate over the dataset and process the persona data
    for idx, data in dataset.iterrows():
        try:
            # Preprocess the ground truth persona (split by '|' character)
            ground_truth_text = preprocess_persona(data['ground_truth'])
            all_ground_truth.append(ground_truth_text)
            
            # Preprocess baseline, initial, and final personas (stringified lists)
            baseline_persona_list = ast.literal_eval(data['baseline_persona'])
            baseline_persona_text = preprocess_persona(' | '.join(baseline_persona_list))
            all_baseline_persona.append(baseline_persona_text)
            
            initial_persona_list = ast.literal_eval(data['initial_persona'])
            initial_persona_text = preprocess_persona(' | '.join(initial_persona_list))
            all_initial_persona.append(initial_persona_text)
            
            final_persona_list = ast.literal_eval(data['final_persona'])
            final_persona_text = preprocess_persona(' | '.join(final_persona_list))
            all_final_persona.append(final_persona_text)
            
        except Exception as e:
            print(f"An error occurred while processing persona_id {idx}: {e}")
            continue
    
    # Concatenate all rows
    ground_truth_concat = ' '.join(all_ground_truth)
    baseline_concat = ' '.join(all_baseline_persona)
    initial_concat = ' '.join(all_initial_persona)
    final_concat = ' '.join(all_final_persona)
    
    # Compute ROUGE scores for concatenated personas using evaluate.load('rouge')
    rouge_scores_baseline = rouge.compute(predictions=[baseline_concat], references=[ground_truth_concat], use_stemmer=True)
    rouge_scores_initial = rouge.compute(predictions=[initial_concat], references=[ground_truth_concat], use_stemmer=True)
    rouge_scores_final = rouge.compute(predictions=[final_concat], references=[ground_truth_concat], use_stemmer=True)
    
    # Save the results to results.csv
    fieldnames = [
        'rouge1_baseline', 'rouge2_baseline', 'rougeL_baseline',
        'rouge1_initial', 'rouge2_initial', 'rougeL_initial',
        'rouge1_final', 'rouge2_final', 'rougeL_final'
    ]
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        row = {
            'rouge1_baseline': rouge_scores_baseline['rouge1'],
            'rouge2_baseline': rouge_scores_baseline['rouge2'],
            'rougeL_baseline': rouge_scores_baseline['rougeL'],
            'rouge1_initial': rouge_scores_initial['rouge1'],
            'rouge2_initial': rouge_scores_initial['rouge2'],
            'rougeL_initial': rouge_scores_initial['rougeL'],
            'rouge1_final': rouge_scores_final['rouge1'],
            'rouge2_final': rouge_scores_final['rouge2'],
            'rougeL_final': rouge_scores_final['rougeL'],
        }
        
        writer.writerow(row)
    
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main()
