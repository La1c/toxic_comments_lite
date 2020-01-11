import luigi

class globalconfig(luigi.Config):
    train_data_path = luigi.Parameter(default='./data/train.csv')
    test_data_path = luigi.Parameter(default='./data/test.csv')
    preprocessed_data_folder = luigi.Parameter(default='./data/prepared')
    featurized_data_folder = luigi.Parameter(default='./data/featurized')
    
    model_artefacts_folder= luigi.Parameter(default='./model/artefacts')
    featurizers_artefacts_folder = luigi.Parameter(default='./model/featurizers')
    
    