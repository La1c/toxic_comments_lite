import mlflow
from typing import Dict, List

class ToxicClassifier():
    def __init__(self):
        self.get_artefacts()
        self.models = {}
        self.clinet = mlflow.tracking.MlflowClient()
    
    def get_artefacts(self):
        # clinet.download_artifacts(clinet.list_run_infos(clinet.get_experiment_by_name('/tfidf').experiment_id)[0].run_id,
        #                            'tfidf_vectorizer.pkl',
        #                            '/home/vageev/artefacts')
        pass
        
    
    def predict(self, entries:List[str]) -> List[Dict[str,str]]:
        return [{
            "comment": "dummy comment",
            "toxic": 0.5,
            "severe_toxic":0.5,
            "obscene":0.5,
            "threat":0.5,
            "insult":0.5,
            "identity_hate":0.5
        }]
        
    
    
    