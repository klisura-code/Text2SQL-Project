import os
import sys
import argparse
import pandas as pd

from main_pipeline import pipeline, model_classes  

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process database schemas and generate outputs.')
    parser.add_argument('--model', type=str, required=True, help='The model name to use (gpt4, gpt4sec, llama2, llama2sec, llama3, llama3sec, t5large, sqlcoder, codellama)')
    parser.add_argument('--num_questions', type=int, default=25, help='Number of initial questions to use in the pipeline')
    parser.add_argument('--output_file', type=str, required=True, help='Output CSV file name')
    parser.add_argument('--dataset_path', type=str, default = '/home/dorde/Desktop/project/data', help='Base path to the data directory')

    return parser.parse_args()

def main():
    args = parse_arguments()

    model_name = args.model
    num_initial_questions = args.num_questions
    output_file = args.output_file
    dataset_path = args.dataset_path

    df = pd.read_csv(os.path.join(dataset_path, 'spider-schema.csv'))

    pd.DataFrame(columns=['db_id', 'schema_input1', 'schema_input2', 'baseline', 'one_iter', 'two_iter']).to_csv(output_file, index=False)

    if model_name not in model_classes:
        raise ValueError(f"Model '{model_name}' is not supported.")
    model_instance = model_classes[model_name]()

    with open(output_file, 'a', buffering=1) as f:
        for index, row in df.iterrows():
            db_id = row['db_id'] 
            schema_input1 = row['schema_input1']
            schema_input2 = row['schema_input2']

            if model_name == 't5large':
                schema_input = schema_input1
                if pd.isna(schema_input) or not schema_input.strip():
                    print(f"schema_input1 is missing or empty for db_id: {db_id}. Skipping...")
                    continue
            else:
                schema_input = schema_input2
                if pd.isna(schema_input) or not schema_input.strip():
                    print(f"schema_input2 is missing or empty for db_id: {db_id}. Skipping...")
                    continue

            try:
                baseline, one_iter_response, two_iter_response = pipeline(schema_input, model_instance, num_initial_questions)
            except Exception as e:
                print(f"An error occurred while processing db_id: {db_id}. Error: {e}")
                continue  

            output_df = pd.DataFrame({
                'db_id': [db_id],
                'schema_input1': [schema_input1],
                'schema_input2': [schema_input2],
                'baseline': [baseline],
                'one_iter': [one_iter_response],
                'two_iter': [two_iter_response],
            })

            output_df.to_csv(f, header=False, index=False)
            print(f"Processed db_id: {db_id}")

if __name__ == "__main__":
    main()