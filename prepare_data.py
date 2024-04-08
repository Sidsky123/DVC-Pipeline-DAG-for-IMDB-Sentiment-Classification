from omegaconf import OmegaConf
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(config):
    print(" <-- Preparation of Data has begun -->")
    df = pd.read_csv(config.data.csv_file_path)
    df['label'] = [1 if i == "positive" else 0 for i in df['sentiment']] 
    print(df.head())
    test_size = config.data.test_set_ratio
    train_df, test_df = train_test_split(df, test_size = test_size, stratify = df['Sentiment'], random_seed = 1234)

    train_df.to_csv(config.data.train_csv_save_path, index = False)
    test_df.to_csv(config.data.test_csv_save_path, index = False)
    

if __name__ == "__main__":
    config = OmegaConf.load("params.yaml")
    prepare_data(config)