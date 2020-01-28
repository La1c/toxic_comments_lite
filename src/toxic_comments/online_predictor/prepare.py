import mlflow
from toxic_comments.global_config import globalconfig
from toxic_comments.utils import try_mkdir
    
    
def get_artefacts():
    client = mlflow.tracking.MlflowClient()
    
    try_mkdir(globalconfig().featurizers_artefacts_folder)
    try_mkdir(globalconfig().model_artefacts_folder)
    
    client.download_artifacts(client.list_run_infos(client.get_experiment_by_name('/tfidf').experiment_id)[0].run_id,
                                   'tfidf_vectorizer.pkl',
                                   globalconfig().featurizers_artefacts_folder)
    
    for category in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        client.download_artifacts(client.list_run_infos(client.get_experiment_by_name(f'/mnb_category_{category}').experiment_id)[0].run_id,
                                   f'mnb_featurizer_{category}.pkl',
                                   globalconfig().featurizers_artefacts_folder)
        
        client.download_artifacts(client.list_run_infos(client.get_experiment_by_name(f'/lr_category_{category}').experiment_id)[0].run_id,
                                   f'{category}_lr.pkl',
                                   globalconfig().model_artefacts_folder)

if __name__ == "__main__":
    get_artefacts()