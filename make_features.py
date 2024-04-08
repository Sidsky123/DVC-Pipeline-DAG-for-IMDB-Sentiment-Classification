from omegaconf import OmegaConf

def make_features(config):
    print("<-- Making Features for you --> ")
    train_df = pd.read_csv(config.data.train_csv_save_path)
    test_df = pd.read_csv(config.data.test_csv_save_path)

    providedVectorizer = config.feature.vectorizer
    vectorizer = {"count-vectorizer": CountVectorizer, "tfidf-vectorizer": TfidfVectorizer}
    [vectorzier_name](stop_words = english)

    train_inputs = vectorizer.fit_transform (train_df['review'])
    test_inputs = vectorizer.fit_transform (test_df['review'])

    joblib.dump(train_inpputs, config.feature.train_features_save_path)
    joblib.dump(test_inpputs, config.feature.test_features_save_path)



if __name__ == "__main__":
    config = OmegaConf.load("./params.yaml")
    make_features(config)
