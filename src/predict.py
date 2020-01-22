import luigi
from sklearn.linear_model import LogisticRegression
import os
import pickle
from utils import try_mkdir
import pandas as pd
import logging
import mlflow
from prepare import PreparationTask
from featurize import GenerateFeaturesWrapperTask
from global_config import globalconfig

logger = logging.getLogger('luigi-interface')

class GetArtifactsTask(luigi.Task):
    pass

class PredictLogRegTask(luigi.Task):
    input_batch_data = luigi.Parameter()
    output_prediction_path = luigi.Parameter()
    
    def output(self):
        output_path = os.path.join(self.output_prediction_path, input_batch_data.split('.csv')[0] + f'_prediction.csv')
        return luigi.LocalTarget(output_path)     
    
    def requires(self):
        return PreparationTask(input_df_file=self.input_batch_data,
                               output_df_folder=globalconfig().preprocessed_data_folder),
        GenerateFeaturesWrapperTask(input_file_path=self.input_batch_data.split('.csv')[0] + '_prepared.csv',
                                           input_artefact_path=globalconfig().featurizers_artefacts_folder,
                                           data_output_path=globalconfig().featurized_data_folder
                                           )
    
    def run(self):
        logger.info(f'Reading data from {self.input_batch_data}')
        data_df = pd.read_csv(self.input_batch_data)
        pred_df = data_df[['id']]
        try_mkdir(self.output().path)
        
        for category in ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]:
            filename = f'{self.input_batch_data.split('.csv')[0] + '_prepared'}_{category}_features.pkl'
            with open(os.path.join(globalconfig().featurized_data_folder, filename), 'rb') as f:
                features = pickle.load(f)
             
            with open(os.path.join(globalconfig().model_artefacts_folder, f'{category}_lr.pkl'), 'rb') as f:
                lr = pickle.load(f)
            
            pred = lr.predict_proba(features)[:, 1]
            pred_df[category] = pred

        pred_df.to_csv(self.output().path, index=False)  
        
        
if __name__ == "__main__":
    luigi.run()