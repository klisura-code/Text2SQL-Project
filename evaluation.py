import sys
import pandas as pd
import re

def split_column_definitions(columns_text):
    # Splits columns_text into individual column definitions, taking care of commas inside parentheses
    column_defs = []
    bracket_level = 0
    current_col = ''
    for c in columns_text:
        if c == '(':
            bracket_level += 1
            current_col += c
        elif c == ')':
            bracket_level -= 1
            current_col += c
        elif c == ',' and bracket_level == 0:
            column_defs.append(current_col.strip())
            current_col = ''
        else:
            current_col += c
    if current_col.strip():
        column_defs.append(current_col.strip())
    return column_defs

def normalize_name(name):
    # Normalize table and column names
    name = name.lower()
    name = name.replace(' ', '_')
    name = re.sub(r'\W+', '', name)  # Remove non-alphanumeric characters
    return name

def normalize_data_type(data_type):
    # Normalize data types, treating similar types as equivalent
    data_type = data_type.lower().strip(' ,')
    if 'varchar' in data_type or 'text' in data_type:
        return 'text'
    elif 'int' in data_type or 'integer' in data_type:
        return 'int'
    elif 'date' in data_type:
        return 'date'
    elif 'boolean' in data_type or 'bool' in data_type:
        return 'bool'
    elif 'float' in data_type or 'double' in data_type or 'real' in data_type:
        return 'float'
    else:
        return data_type

def parse_create_table_statement(sql_text):
    # Extracts table name and columns from a CREATE TABLE statement
    pattern = r'CREATE\s+TABLE\s+([^\s(]+)\s*\((.*)\)'
    match = re.search(pattern, sql_text, re.DOTALL | re.IGNORECASE)
    if match:
        table_name = normalize_name(match.group(1).strip('`"[]'))
        columns_text = match.group(2)
        columns = []
        column_defs = split_column_definitions(columns_text)
        for col_def in column_defs:
            col_def = col_def.strip()
            tokens = col_def.split()
            if len(tokens) >= 2:
                col_name = normalize_name(tokens[0].strip('`"[]'))
                data_type = normalize_data_type(' '.join(tokens[1:]).strip('`,'))
                columns.append((col_name, data_type))
        return table_name, columns
    else:
        return None, []

def parse_sql_text(sql_text):
    # Parses multiple CREATE TABLE statements and returns a dictionary of tables
    statements = re.split(r';\s*(?:--.*\n)*', sql_text)
    tables = {}
    for stmt in statements:
        stmt = stmt.strip()
        if stmt and 'CREATE TABLE' in stmt.upper():
            table_name, columns = parse_create_table_statement(stmt)
            if table_name:
                tables[table_name] = columns
    return tables

def compute_metrics(gold_items, pred_items):
    TP = len(gold_items & pred_items)
    FP = len(pred_items - gold_items)
    FN = len(gold_items - pred_items)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def compute_all_metrics(gold_tables, pred_tables):
    # Computes metrics for tables, table+column, and table+column+type
    gold_table_names = set(gold_tables.keys())
    pred_table_names = set(pred_tables.keys())
    table_precision, table_recall, table_f1 = compute_metrics(gold_table_names, pred_table_names)

    gold_table_columns = {(table, col[0]) for table, cols in gold_tables.items() for col in cols}
    pred_table_columns = {(table, col[0]) for table, cols in pred_tables.items() for col in cols}
    tc_precision, tc_recall, tc_f1 = compute_metrics(gold_table_columns, pred_table_columns)

    gold_table_col_types = {(table, col[0], col[1]) for table, cols in gold_tables.items() for col in cols}
    pred_table_col_types = {(table, col[0], col[1]) for table, cols in pred_tables.items() for col in cols}
    tct_precision, tct_recall, tct_f1 = compute_metrics(gold_table_col_types, pred_table_col_types)

    return {'precision': table_precision, 'recall': table_recall, 'f1': table_f1}, \
           {'precision': tc_precision, 'recall': tc_recall, 'f1': tc_f1}, \
           {'precision': tct_precision, 'recall': tct_recall, 'f1': tct_f1}

def accumulate_metrics(metrics, p_r_f1_table, p_r_f1_col, p_r_f1_coltype):
    for key in ['precision', 'recall', 'f1']:
        metrics['table'][key] += p_r_f1_table[key]
        metrics['table+column'][key] += p_r_f1_col[key]
        metrics['table+column+type'][key] += p_r_f1_coltype[key]

def average_metrics(metrics, total):
    for level in metrics.keys():
        for key in metrics[level]:
            metrics[level][key] /= total

def main():
    input_csv = sys.argv[1]
    df = pd.read_csv(input_csv)
    total_databases = 0

    # Initialize metrics dictionaries
    metrics_baseline = {'table': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                        'table+column': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                        'table+column+type': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}}
    metrics_one_iter = {'table': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                        'table+column': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                        'table+column+type': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}}
    metrics_two_iter = {'table': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                        'table+column': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                        'table+column+type': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}}

    for idx, row in df.iterrows():
        schema_input2 = row['schema_input2']
        baseline = row['baseline']
        one_iter = row['one_iter']
        two_iter = row['two_iter']

        # Parse the schemas
        gold_tables = parse_sql_text(schema_input2)
        baseline_tables = parse_sql_text(baseline)
        one_iter_tables = parse_sql_text(one_iter)
        two_iter_tables = parse_sql_text(two_iter)

        # Compute metrics for each comparison
        # Baseline
        p_r_f1_table, p_r_f1_col, p_r_f1_coltype = compute_all_metrics(gold_tables, baseline_tables)
        accumulate_metrics(metrics_baseline, p_r_f1_table, p_r_f1_col, p_r_f1_coltype)

        # One Iteration
        p_r_f1_table, p_r_f1_col, p_r_f1_coltype = compute_all_metrics(gold_tables, one_iter_tables)
        accumulate_metrics(metrics_one_iter, p_r_f1_table, p_r_f1_col, p_r_f1_coltype)

        # Two Iterations
        p_r_f1_table, p_r_f1_col, p_r_f1_coltype = compute_all_metrics(gold_tables, two_iter_tables)
        accumulate_metrics(metrics_two_iter, p_r_f1_table, p_r_f1_col, p_r_f1_coltype)

        total_databases += 1

    # Average the metrics
    average_metrics(metrics_baseline, total_databases)
    average_metrics(metrics_one_iter, total_databases)
    average_metrics(metrics_two_iter, total_databases)

    # Prepare the output DataFrame
    output_df = pd.DataFrame([
        {'Method': 'baseline',
         'table_precision': metrics_baseline['table']['precision'],
         'table_recall': metrics_baseline['table']['recall'],
         'table_f1': metrics_baseline['table']['f1'],
         'table+column_precision': metrics_baseline['table+column']['precision'],
         'table+column_recall': metrics_baseline['table+column']['recall'],
         'table+column_f1': metrics_baseline['table+column']['f1'],
         'table+column+type_precision': metrics_baseline['table+column+type']['precision'],
         'table+column+type_recall': metrics_baseline['table+column+type']['recall'],
         'table+column+type_f1': metrics_baseline['table+column+type']['f1']},
        {'Method': 'one_iter',
         'table_precision': metrics_one_iter['table']['precision'],
         'table_recall': metrics_one_iter['table']['recall'],
         'table_f1': metrics_one_iter['table']['f1'],
         'table+column_precision': metrics_one_iter['table+column']['precision'],
         'table+column_recall': metrics_one_iter['table+column']['recall'],
         'table+column_f1': metrics_one_iter['table+column']['f1'],
         'table+column+type_precision': metrics_one_iter['table+column+type']['precision'],
         'table+column+type_recall': metrics_one_iter['table+column+type']['recall'],
         'table+column+type_f1': metrics_one_iter['table+column+type']['f1']},
        {'Method': 'two_iter',
         'table_precision': metrics_two_iter['table']['precision'],
         'table_recall': metrics_two_iter['table']['recall'],
         'table_f1': metrics_two_iter['table']['f1'],
         'table+column_precision': metrics_two_iter['table+column']['precision'],
         'table+column_recall': metrics_two_iter['table+column']['recall'],
         'table+column_f1': metrics_two_iter['table+column']['f1'],
         'table+column+type_precision': metrics_two_iter['table+column+type']['precision'],
         'table+column+type_recall': metrics_two_iter['table+column+type']['recall'],
         'table+column+type_f1': metrics_two_iter['table+column+type']['f1']}
    ])

    output_df.to_csv('output_metrics.csv', index=False)

if __name__ == '__main__':
    main()
