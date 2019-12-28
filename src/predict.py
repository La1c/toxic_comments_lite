if __name__ == "__main__":
    import argparse
    import os
    from sklearn.linear_model import LogisticRegression
    import pickle
    from utils import try_mkdir
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument('input_data_path', type=str)
    parser.add_argument('input_features_path', type=str)
    parser.add_argument('artifacts_path', type=str)
    parser.add_argument('output_data_path', type=str)
    args = parser.parse_args()

    data_df = pd.read_csv(args.input_data_path)
    pred_df = data_df[['id']]
    try_mkdir(args.output_data_path)

    for category in ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]:
        with open(os.path.join(args.input_features_path, '{}_features_test.pkl'.format(category)), 'rb') as f:
            features = pickle.load(f)
            
        with open(os.path.join(args.artifacts_path, '{}_lr.pkl'.format(category)), 'rb') as f:
            lr = pickle.load(f)
            
        pred = lr.predict_proba(features)[:, 1]
        pred_df[category] = pred

    pred_df.to_csv(os.path.join(args.output_data_path, 'test_pred.csv'), index=False)  
