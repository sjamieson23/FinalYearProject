import pandas as pd
from sklearn.model_selection import train_test_split


def create_dataset():
    fine_tuned_spearbot_df = pd.read_csv("Data/RefinedSpearPhishing/refined_spearbot_data.csv")
    spearbot_df = pd.read_csv("Data/SpearBot/spearbot_data.csv")
    old_data_df = pd.read_csv("Data/Original/SplitData/all_data_test.csv")

    # Creating 4 datasets here
    fine_tune_training_df = pd.DataFrame()
    fine_tune_testing_df = pd.DataFrame()
    ensemble_training_df = pd.DataFrame()
    ensemble_testing_df = pd.DataFrame()

    # Take 1000 emails from spearbot dataset and put into finetune dataset
    # Put the other 200 emails into fine tune test dataset
    fine_tune_training_df, fine_tune_testing_df = train_test_split(spearbot_df, test_size=200, stratify=spearbot_df['label'], random_state=1)

    # Of those 200 put 150 into ensemble dataset
    # Put the other 50 into ensemble test dataset
    ensemble_training_df, ensemble_testing_df = train_test_split(fine_tune_testing_df, test_size=50,
                                                                   stratify=fine_tune_testing_df['label'], random_state=1)

    # Then using the old dataset
    # Add 112 into finetune dataset
    temp_df, old_train_finetune = train_test_split(old_data_df, test_size=112,
                                                                   stratify=old_data_df['label'], random_state=1)
    fine_tune_training_df = pd.concat([fine_tune_training_df, old_train_finetune], ignore_index=True)

    # (not) Add 23 into fine tune test dataset
    #temp_df, old_test_finetune = train_test_split(temp_df, test_size=23,
    #                                               stratify=temp_df['label'], random_state=1)
    #fine_tune_testing_df = pd.concat([fine_tune_testing_df, old_test_finetune], ignore_index=True)
    # Add 300 into ensemble dataset
    temp_df, old_train_ensemble = train_test_split(temp_df, test_size=300,
                                                  stratify=temp_df['label'], random_state=1)
    ensemble_training_df = pd.concat([ensemble_training_df, old_train_ensemble], ignore_index=True)

    # Add 150 into ensemble test dataset
    temp_df, old_test_ensemble = train_test_split(temp_df, test_size=150,
                                                   stratify=temp_df['label'], random_state=1)
    ensemble_testing_df = pd.concat([ensemble_testing_df, old_test_ensemble], ignore_index=True)

    # Then from proper spearbot dataset
    # Add 800 into ensemble dataset
    temp_train_df, temp_test_df = train_test_split(fine_tuned_spearbot_df, test_size=400,
                                                   stratify=fine_tuned_spearbot_df['label'], random_state=1)

    # Add 400 into ensemble test dataset
    ensemble_training_df = pd.concat([ensemble_training_df, temp_train_df], ignore_index=True)
    ensemble_testing_df = pd.concat([ensemble_testing_df, temp_test_df], ignore_index=True)

    fine_tune_training_df.to_csv("Data/SpearBot/finetune_training_data.csv", index=False)
    fine_tune_testing_df.to_csv("Data/SpearBot/finetune_testing_data.csv", index=False)
    ensemble_training_df.to_csv("Data/SpearBot/ensemble_training_data.csv", index=False)
    ensemble_testing_df.to_csv("Data/SpearBot/ensemble_testing_data.csv", index=False)


def check_datasets():
    fine_tune_training_df = pd.read_csv("Data/SpearBot/finetune_training_data.csv")
    fine_tune_testing_df = pd.read_csv("Data/SpearBot/finetune_testing_data.csv")
    ensemble_training_df = pd.read_csv("Data/SpearBot/ensemble_training_data.csv")
    ensemble_testing_df = pd.read_csv("Data/SpearBot/ensemble_testing_data.csv")
    print(f"Finetune traing length: {len(fine_tune_training_df)}, Finetune test length: {len(fine_tune_testing_df)}, "
          f"Ensemble train length: {len(ensemble_training_df)}, Ensemble test length: {len(ensemble_testing_df)}")
    print(f"Finetune traing label: {fine_tune_training_df['label'].value_counts()}, Finetune test label: {fine_tune_testing_df['label'].value_counts()}, "
          f"Ensemble train label: {ensemble_training_df['label'].value_counts()}, Ensemble test label: {ensemble_testing_df['label'].value_counts()}")

create_dataset()
check_datasets()