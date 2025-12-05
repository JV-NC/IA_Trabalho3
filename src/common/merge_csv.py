import pandas as pd

csv_dir = 'data/kaggle_dataset/'
train_csv = f'{csv_dir}train.csv'
test_csv = f'{csv_dir}test.csv'
output_csv = f'{csv_dir}FlightSatisfaction.csv'

def merge_csvs(train_csv: str, test_csv: str, output_csv: str='unified_dataset.csv')->pd.DataFrame:
    """
    Merge training and testing CSV files into a single CSV file.
    """
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    def clean(df):
        # Remove any Unnamed column
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        # Remove, if exists, 'id' column
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
        return df
    
    df_train = clean(df_train)
    df_test = clean(df_test)
    print('Cleaned Unnamed and "id" columns!')

    if list(df_train.columns) != list(df_test.columns):
        raise ValueError('CSVs columns are different, verify if they are correct!')
    print('Equal columns confirmed!')

    df_full = pd.concat([df_train,df_test],axis=0,ignore_index=True)

    df_full.to_csv(output_csv,index=False)
    print(f'{output_csv} created!')

    return df_full

def main():
    merge_csvs(train_csv,test_csv,output_csv)

if __name__ == '__main__':
    main()