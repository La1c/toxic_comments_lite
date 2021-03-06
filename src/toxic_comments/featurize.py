import luigi
from bpemb import BPEmb
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import pickle
from toxic_comments.utils import try_mkdir
from toxic_comments.featurizers.mnb_featurizer import MNBFeaturizer
from toxic_comments.prepare import PreparationTask
from toxic_comments.global_config import globalconfig
import logging
import mlflow

logger = logging.getLogger('luigi-interface')

class TrainTfidfTask(luigi.Task):
    input_file_path = luigi.Parameter(globalconfig().prepared_train_data_path)
    artefact_output_path = luigi.Parameter(globalconfig().featurizers_artefacts_folder)

    def requires(self):
        return PreparationTask(input_df_file=globalconfig().train_data_path, 
                               output_df_folder=globalconfig().preprocessed_data_folder)
        
    def output(self):
        output_name = os.path.join(self.artefact_output_path, 'tfidf_vectorizer.pkl')
        return luigi.LocalTarget(output_name)

    def run(self):
        logger.info('Reading data from {}'.format(self.input_file_path))
        data_df = pd.read_csv(self.input_file_path)
        bpemb_en = BPEmb(lang="en", dim=50, vs=200000, cache_dir='./bpemb_cache')
        tfidf = TfidfVectorizer(tokenizer=bpemb_en.encode)
        logger.info("Fitting tfidf")
        tfidf.fit(data_df['comment_text'])

        try_mkdir(self.artefact_output_path)
        with open(self.output().path, 'wb') as f:
            pickle.dump(tfidf, f)
            
        try: 
            mlflow.set_experiment('/tfidf') 
            with mlflow.start_run():
                logger.info("Sending tfidf artefact to MLFlow")
                mlflow.log_artifact(self.output().path)     
        except Exception as e:
            logger.error("Something went wrong while trying to use MLFlow tracking: ", e)

class TrainMNBTask(luigi.Task):
    input_file_path = luigi.Parameter(globalconfig().prepared_train_data_path)
    artefact_output_path = luigi.Parameter(globalconfig().featurizers_artefacts_folder)
    category_name = luigi.Parameter()

    def requires(self):
        return TrainTfidfTask(input_file_path=self.input_file_path,
                              artefact_output_path=self.artefact_output_path)

    def output(self):
        output_name = os.path.join(self.artefact_output_path, 
                                   f'mnb_featurizer_{self.category_name}.pkl')
        return luigi.LocalTarget(output_name)

    def run(self):
        data_df = pd.read_csv(self.input_file_path)
        
        with open(self.requires().output().path, 'rb') as f:
            tfidf=pickle.load(f)
            
        featurizer = MNBFeaturizer(tfidf)
        logger.info("Fitting MNB for category {}".format(self.category_name))
        featurizer.fit(data_df['comment_text'], data_df[self.category_name])
        try_mkdir(self.artefact_output_path)
        featurizer.save(self.output().path)
        
        try: 
            mlflow.set_experiment(f'/mnb_category_{self.category_name}') 
            with mlflow.start_run():
                logger.info("Sending MNB artefact to MLFlow")
                mlflow.log_artifact(self.output().path)     
        except Exception as e:
            logger.error("Something went wrong while trying to use MLFlow tracking: ", e)
        
class GenerateMNBFeaturesTask(luigi.Task):
    input_file_path = luigi.Parameter('./data/prepared/train_prepared.csv')
    input_artefact_path = luigi.Parameter(globalconfig().featurizers_artefacts_folder)
    data_output_path = luigi.Parameter(globalconfig().featurized_data_folder)
    category_name = luigi.Parameter()
    
    def requires(self):
        return TrainMNBTask(input_file_path=globalconfig().prepared_train_data_path,
                            artefact_output_path=self.input_artefact_path,
                            category_name=self.category_name)
        
    def output(self):
        file_name = self.input_file_path.split('/')[-1].split('.csv')[0]
        output_name = os.path.join(self.data_output_path,
                                   f'{file_name}_{self.category_name}_features.pkl')
        
        return luigi.LocalTarget(output_name)
    
    def run(self):
        logger.info("Generating features")
        logger.info("Reading data at {}".format(self.input_file_path))
        data_df = pd.read_csv(self.input_file_path)
        
        featurizer_name = os.path.join(self.input_artefact_path,
                                       f'mnb_featurizer_{self.category_name}.pkl')
        
        featurizer = MNBFeaturizer.load(featurizer_name)
        logger.info('Transofrming data')
        transformed = featurizer.transform(data_df['comment_text'])
        
        try_mkdir(self.data_output_path)
        with open(self.output().path, 'wb') as f:
            pickle.dump(transformed, f)
            
class GenerateFeaturesWrapperTask(luigi.WrapperTask):
    input_file_path = luigi.Parameter('./data/prepared/train_prepared.csv')
    input_artefact_path = luigi.Parameter(globalconfig().featurizers_artefacts_folder)
    data_output_path = luigi.Parameter(globalconfig().featurized_data_folder)
    
    def requires(self):
        for category in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
            yield GenerateMNBFeaturesTask(input_file_path=self.input_file_path,
                input_artefact_path=self.input_artefact_path,
                data_output_path=self.data_output_path,
                category_name=category)
    
if __name__ == "__main__":
    luigi.run()
            


            


            


