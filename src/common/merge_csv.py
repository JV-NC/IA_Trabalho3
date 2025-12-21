import os
import pandas as pd

csv_dir = 'data/kaggle_dataset'
train_csv = os.path.join(csv_dir, 'train.csv')
test_csv = os.path.join(csv_dir, 'test.csv')
output_csv = os.path.join(csv_dir, 'FlightSatisfaction.csv')


def merge_csvs(train_csv: str,test_csv: str,output_csv: str)->pd.DataFrame:
    """
    Merge training and testing CSV files into a single CSV file.
    Safe and idempotent version.
    """

    #Ensure that dir exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    #Verify input files
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f'Missing file: {train_csv}')

    if not os.path.exists(test_csv):
        raise FileNotFoundError(f'Missing file: {test_csv}')

    #If output already exists, print on screen
    if os.path.exists(output_csv):
        print(f'Dataset already exists: {output_csv}')
        print('Recreating dataset to ensure consistency...')

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
        return df

    df_train = clean(df_train)
    df_test = clean(df_test)
    print('Cleaned Unnamed and id columns')

    if list(df_train.columns) != list(df_test.columns):
        raise ValueError('Train and test CSV columns do not match!')

    df_full = pd.concat([df_train, df_test], ignore_index=True)
    df_full.to_csv(output_csv, index=False)

    print(f'Dataset generated: {output_csv}')
    return df_full

def main():
    merge_csvs(train_csv, test_csv, output_csv)

if __name__ == '__main__':
    main()