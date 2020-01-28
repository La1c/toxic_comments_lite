import luigi
from sklearn.linear_model import LogisticRegression
import os
import pickle
from toxic_comments.utils import try_mkdir
import pandas as pd
import logging
import mlflow
from toxic_comments.prepare import PreparationTask
from toxic_comments.featurize import GenerateFeaturesWrapperTask, GenerateMNBFeaturesTask
from toxic_comments.train import TrainLogRegAllWrapperTask
from toxic_comments.global_config import globalconfig

logger = logging.getLogger('luigi-interface')

class PredictLogRegTask(luigi.Task):
    input_batch_data = luigi.Parameter(default=globalconfig().test_data_path)
    output_prediction_path = luigi.Parameter(default=globalconfig().output_prediction_path)
    
    def output(self):
        output_path = os.path.join(self.output_prediction_path, 
                                   self.input_batch_data.split('/')[-1].split('.csv')[0] + f'_prediction.csv')
        
        return luigi.LocalTarget(output_path)     
    
    def requires(self):
        test_preparation = PreparationTask(input_df_file=self.input_batch_data,
                               output_df_folder=globalconfig().preprocessed_data_folder)
        
        requirements_dict = {'prepared_test':test_preparation}
        
        for category in ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]:
            requirements_dict[f'generated_test_features_{category}'] = GenerateMNBFeaturesTask(input_file_path=test_preparation.output().path,
                                                                                               input_artefact_path=globalconfig().featurizers_artefacts_folder,
                                                                                               data_output_path=globalconfig().featurized_data_folder,
                                                                                               category_name=category)
        
        requirements_dict['trained_log_reg']= TrainLogRegAllWrapperTask(input_file_path=globalconfig().prepared_train_data_path,
                                             input_features_path=globalconfig().featurized_data_folder,
                                             output_artefact_path = globalconfig().model_artefacts_folder)
            
        return requirements_dict
    
    def run(self):
        logger.info(f'Reading data from {self.input_batch_data}')
        data_df = pd.read_csv(self.input_batch_data)
        pred_df = data_df[['id']]
        try_mkdir(self.output_prediction_path)
        
        for category in ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]:
            filename = self.input()[f'generated_test_features_{category}'].path
            with open(filename, 'rb') as f:
                features = pickle.load(f)
             
            with open(os.path.join(globalconfig().model_artefacts_folder, f'{category}_lr.pkl'), 'rb') as f:
                lr = pickle.load(f)
            
            pred = lr.predict_proba(features)[:, 1]
            pred_df[category] = pred

        pred_df.to_csv(self.output().path, index=False)  
        
        
if __name__ == "__main__":
    luigi.run()