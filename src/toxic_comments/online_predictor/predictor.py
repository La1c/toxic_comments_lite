import mlflow
from typing import Dict, List
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from bpemb import BPEmb
from toxic_comments.featurizers.mnb_featurizer import MNBFeaturizer
from sklearn.linear_model import LogisticRegression
from toxic_comments.global_config import globalconfig
import pickle
import os

class ToxicClassifier():
    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    def __init__(self):
        self.models = {category: self.build_pipeline(category) for category in self.categories}
        
    def build_pipeline(self, category: str) -> Pipeline:
        with open(os.path.join(globalconfig().featurizers_artefacts_folder,
                                f'mnb_featurizer_{category}.pkl'), 'rb') as f:
            mnb = pickle.load(f) 
        
        with open(os.path.join(globalconfig().model_artefacts_folder, 
                               f'{category}_lr.pkl'), 'rb') as f:
            lr = pickle.load(f)
            
        pipeline = Pipeline([('mnb',mnb), 
                             ('lr', lr)])
        
        return pipeline
    
    def predict(self, entries:List[str]) -> List[Dict[str, str]]:
        prediction = {category: self.models[category].predict_proba(entries)[:, 1] for category in self.categories}
        
        output = []
        for i,entry in enumerate(entries):
            output_entry={"comment": entry}
            
            for category in self.categories:
                output_entry[category] = f'{prediction[category][i]:.2f}'
                
            output.append(output_entry) 
            
        return output

    
    
    
    