import luigi

class globalconfig(luigi.Config):
    train_data_path = luigi.Parameter(default='/toxic_comments/data/train.csv')
    test_data_path = luigi.Parameter(default='/toxic_comments/data/test.csv')
    preprocessed_data_folder = luigi.Parameter(default='/toxic_comments/data/prepared')
    featurized_data_folder = luigi.Parameter(default='/toxic_comments/data/featurized')
    
    model_artefacts_folder= luigi.Parameter(default='/toxic_comments/model/artefacts')
    featurizers_artefacts_folder = luigi.Parameter(default='/toxic_comments/model/featurizers')
    
    