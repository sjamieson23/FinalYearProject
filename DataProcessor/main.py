# This is a sample Python script.
from email.contentmanager import raw_data_manager

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split




def print_csv_details(name):
    print("\n" + "-"*20)
    print(f"\nFile: {name}")
    file_path = "Data/RawData/" + name
    df = pd.read_csv(file_path)
    p_total_rows = len(df)
    p_label_equals_1 = (df['label'] == 1).sum()
    columns = list(df.columns)
    print(f"Column names: {columns}")
    print(f"Total number of emails: {p_total_rows}")
    print(f"Total number of phishing emails: {p_label_equals_1}")

    return p_total_rows, p_label_equals_1, columns

def combineDataSets(itemDir):
    seven_column_df = pd.DataFrame()
    all_df = pd.DataFrame()

    for item in itemDir.iterdir():
        if item.name.endswith(".csv"):
            df = pd.read_csv(item)
            all_df = pd.concat([all_df, df[['subject', 'body', 'label']]], ignore_index=True)
            if len(df.columns) == 7:
                seven_column_df = pd.concat([seven_column_df, df], ignore_index=True)

    all_df.to_csv("Data/ProcessedData/all_data.csv", index=False)
    seven_column_df.to_csv("Data/ProcessedData/seven_column_data.csv", index=False)

def process_all_data(name):
    print("\n" + "-" * 20)
    print("Processing data")
    print("\n" + "-" * 20)
    print(f"\nFile: {name}")
    file_path = "Data/ProcessedData/" + name
    df = pd.read_csv(file_path)

    df = df.dropna(subset=['label'])
    df = df.reset_index(drop=True)
    total_rows = len(df)
    label_equals_1 = (df['label'] == 1).sum()
    print("\n" + "-" * 10 + "After dropna label" + "-" * 10)
    print(f"After Total number of emails: {total_rows}")
    print(f"After Total number of phishing emails: {label_equals_1}")

    df = df.dropna(subset=['body', 'subject'], how='all')
    df = df.reset_index(drop=True)
    total_rows = len(df)
    label_equals_1 = (df['label'] == 1).sum()
    print("\n" + "-" * 10 + "After dropna body and subject" + "-" * 10)
    print(f"After Total number of emails: {total_rows}")
    print(f"After Total number of phishing emails: {label_equals_1}")

    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    total_rows = len(df)
    label_equals_1 = (df['label'] == 1).sum()
    print("\n" + "-" * 10 + "After dropna duplicates" + "-" * 10)
    print(f"After Total number of emails: {total_rows}")
    print(f"After Total number of phishing emails: {label_equals_1}")

    df['subject_missing'] = df['subject'].isna().astype(int)
    df['body_missing'] = df['body'].isna().astype(int)
    df['subject'] = df['subject'].fillna('<EMPTY_SUBJECT>')
    df['body'] = df['body'].fillna('<EMPTY_BODY>')

    df.to_csv("Data/ProcessedData/" + name, index=False)
    return total_rows, label_equals_1, list(df.columns)

def process_seven_column_data(name):
    file_path = "Data/ProcessedData/" + name

def split_data(name):
    file_path = "Data/ProcessedData/" + name
    df = pd.read_csv(file_path)
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=1)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=1)

    file_name = name.removesuffix(".csv")
    train_df.to_csv("Data/SplitData/" + file_name + "_train.csv", index=False)
    val_df.to_csv("Data/SplitData/" + file_name + "_val.csv", index=False)
    test_df.to_csv("Data/SplitData/" + file_name + "_test.csv", index=False)

    train_total_rows = len(train_df)
    train_label_equals_1 = (train_df['label'] == 1).sum()

    val_total_rows = len(val_df)
    val_label_equals_1 = (val_df['label'] == 1).sum()

    test_total_rows = len(test_df)
    test_label_equals_1 = (test_df['label'] == 1).sum()

    print("\n" + "-" * 20)
    print(f"Total number of training emails: {train_total_rows} of which phishing emails: {train_label_equals_1}")
    print(f"Total number of validation emails: {val_total_rows} of which phishing emails: {val_label_equals_1}")
    print(f"Total number of testing emails: {test_total_rows} of which phishing emails: {test_label_equals_1}")

if __name__ == '__main__':
    raw_path = Path('Data/RawData')
    p_total_rows_all = 0
    p_total_phish_all = 0
    p_total_rows_7_colums = 0
    p_total_phish_7_colums = 0
    total_rows_all = 0
    total_phish_all = 0
    total_rows_7_colums = 0
    total_phish_7_colums = 0
    for item in raw_path.iterdir():
        if item.name.endswith(".csv"):
            response = print_csv_details(item.name)
            p_total_rows_all += response[0]
            p_total_phish_all += response[1]
            if len(response[2]) == 7:
                p_total_rows_7_colums += response[0]
                p_total_phish_7_colums += response[1]
    combineDataSets(raw_path)
    processed_path = Path('Data/ProcessedData')
    for item in processed_path.iterdir():
        if item.name.endswith(".csv"):
            response = process_all_data(item.name)
            if len(response[2]) == 7:
                total_rows_7_colums += response[0]
                total_phish_7_colums += response[1]
            else:
                total_rows_all += response[0]
                total_phish_all += response[1]

    print("\n" + "-"*20)
    print(f"Total Emails: {p_total_rows_all} -> {total_rows_all}")
    print(f"Total Phishing emails: {p_total_phish_all} -> {total_phish_all}")
    print(f"Total emails with 7 columns: {p_total_rows_7_colums} -> {total_rows_7_colums}")
    print(f"Total phish emails with 7 columns: {p_total_phish_7_colums} -> {total_phish_7_colums}")

    for item in processed_path.iterdir():
        if item.name.endswith(".csv"):
            split_data(item.name)
