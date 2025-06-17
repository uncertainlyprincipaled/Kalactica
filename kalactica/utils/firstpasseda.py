import os
import pandas as pd

DATA_DIR = '/kaggle/input/meta-kaggle/'

def analyze_csv(file_path):
    print(f'\n===== {os.path.basename(file_path)} =====')
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f'Could not read {file_path}: {e}')
        return
    print('Columns:', list(df.columns))
    print('Dtypes:')
    print(df.dtypes)
    print('\nDescriptive statistics:')
    # Numeric stats
    num_cols = df.select_dtypes(include='number').columns
    if len(num_cols) > 0:
        print('\nNumeric columns:')
        print(df[num_cols].describe())
    # Categorical stats
    cat_cols = df.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        print('\nCategorical columns:')
        print(df[cat_cols].describe())

def main():
    for fname in os.listdir(DATA_DIR):
        if fname.endswith('.csv'):
            analyze_csv(os.path.join(DATA_DIR, fname))

if __name__ == '__main__':
    main() 