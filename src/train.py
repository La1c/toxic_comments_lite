
if __name__ == "__main__":
    import argparse
    import os
    from sklearn.linear_model import LogisticRegressionCV
    import pickle
    from utils import try_mkdir
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument('input_data_path', type=str)
    parser.add_argument('input_features_path', type=str)
    parser.add_argument('artifacts_path', type=str)
    args = parser.parse_args()  

    print('Reading data from {}'.format(args.input_data_path))
    data_df = pd.read_csv(args.input_data_path)
    try_mkdir(args.artifacts_path)

    for category in ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]:
        with open(os.path.join(args.input_features_path, '{}_features_train.pkl'.format(category)), 'rb') as f:
            features = pickle.load(f)
        model = LogisticRegressionCV([0.01, 0.1, 1, 10, 100], cv=5)
        model.fit(features, data_df[category])

        with open(os.path.join(args.artifacts_path, '{}_lr.pkl'.format(category)), 'wb') as f:
            pickle.dump(model, f)

            
                
    