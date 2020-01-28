import luigi
import os
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, make_scorer
import pickle
from toxic_comments.utils import try_mkdir
import pandas as pd
import mlflow
from datetime import datetime
import numpy as np
from toxic_comments.featurize import GenerateMNBFeaturesTask
from toxic_comments.global_config import globalconfig
import logging

logger = logging.getLogger('luigi-interface')

class TrainLogRegTask(luigi.Task):
    input_file_path = luigi.Parameter(default='./data/prepared/train_prepared.csv')
    input_features_path = luigi.Parameter(globalconfig().featurized_data_folder)
    category_name = luigi.Parameter()
    output_artefact_path = luigi.Parameter(globalconfig().model_artefacts_folder)
    
    def requires(self):
        return GenerateMNBFeaturesTask(input_file_path=self.input_file_path,
                                       input_artefact_path=globalconfig().featurizers_artefacts_folder,
                                       data_output_path=globalconfig().featurized_data_folder,
                                       category_name=self.category_name)
    
    def output(self):
        output_path = os.path.join(self.output_artefact_path, f'{self.category_name}_lr.pkl')
        return luigi.LocalTarget(output_path)
    
    def run(self):
        logger.info(f'Reading data from {self.input_file_path}')
        data_df = pd.read_csv(self.input_file_path)
        try_mkdir(self.output_artefact_path)
        
        features_file_name = self.input_file_path.split('/')[-1].split('.csv')[0]
        with open(os.path.join(self.input_features_path, f'{features_file_name}_{self.category_name}_features.pkl'), 'rb') as f:
            features = pickle.load(f)
            
        C_s = [0.01, 0.1, 1, 10, 100]
        model = LogisticRegressionCV(C_s,
                                     cv=5, 
                                     n_jobs=-1,
                                     max_iter=1000,
                                     scoring=make_scorer(roc_auc_score))
        
        logger.info(f"Fitting lr for category {self.category_name}")
        model.fit(features, data_df[self.category_name])
        
        with open(self.output().path, 'wb') as f:
            logger.info(f"Saving predictor locally for category {self.category_name}")
            pickle.dump(model, f)
            
        try: 
            mlflow.set_experiment(f'/lr_category_{self.category_name}') 
            with mlflow.start_run():
                logger.info("Sending cv parameters and scores to ML Flow")
                for i, c in enumerate(model.C_):
                    mlflow.log_param(f'C_{i}', c)
                    mlflow.log_metric(f"mean_roc_auc_C_{C_s[i]}", np.mean(model.scores_[1][:, i]))  
                
                logger.info("Sending model artifact to ML Flow")
                mlflow.log_artifact(self.output().path)     
        except Exception as e:
            logger.error("Something went wrong while trying to use MLFlow tracking: ", e)

class TrainLogRegAllWrapperTask(luigi.WrapperTask):
    input_file_path = luigi.Parameter(default='./data/prepared/train_prepared.csv')
    input_features_path = luigi.Parameter(globalconfig().featurized_data_folder)
    output_artefact_path = luigi.Parameter(globalconfig().model_artefacts_folder)
    
    def requires(self):
        for category in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
            yield TrainLogRegTask(input_file_path=self.input_file_path,
                              input_features_path=self.input_features_path,
                              output_artefact_path=self.output_artefact_path,
                              category_name=category)
    
if __name__ == "__main__":
    luigi.run()
            
                

            
                
    