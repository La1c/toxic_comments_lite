
if __name__ == "__main__":
    import argparse
    import os
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score, make_scorer
    import pickle
    from utils import try_mkdir
    import pandas as pd
    import mlflow
    from datetime import datetime
    import numpy as np
    
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
            
        C_s = [0.01, 0.1, 1, 10, 100]
        model = LogisticRegressionCV(C_s,
                                     cv=5, 
                                     n_jobs=-1,
                                     max_iter=1000,
                                     scoring=make_scorer(roc_auc_score))
        
        print("Fitting lr for category {}".format(category))
        model.fit(features, data_df[category])
        

        with open(os.path.join(args.artifacts_path, '{}_lr.pkl'.format(category)), 'wb') as f:
            print("Saving predictor locally for category {}".format(category))
            pickle.dump(model, f)
        
        try: 
            mlflow.set_experiment('/log_reg_cv_{}_{}'.format(category, datetime.today().strftime('%Y%m%d'))) 
            with mlflow.start_run():
                print("Sending cv parameters and scores to ML Flow")
                for i, c in enumerate(model.C_):
                    mlflow.log_param('C_{}'.format(i), c)
                    mlflow.log_metric("mean_roc_auc_C_{}".format(C_s[i]), np.mean(model.scores_[1][:, i]))  
                
                print("Sending model artifact to ML Flow")
                mlflow.log_artifact(os.path.join(args.artifacts_path, '{}_lr.pkl'.format(category)))     
        except Exception as e:
            print(e)
            
                

            
                
    