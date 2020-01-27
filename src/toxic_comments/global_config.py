import luigi

class globalconfig(luigi.Config):
    train_data_path = luigi.Parameter(default='/toxic_classifier/data/train.csv')
    test_data_path = luigi.Parameter(default='/toxic_classifier/data/test.csv')
    output_prediction_path=luigi.Parameter(default='/toxic_classifier/data/predictions')
    preprocessed_data_folder = luigi.Parameter(default='/toxic_classifier/data/prepared')
    featurized_data_folder = luigi.Parameter(default='/toxic_classifier/data/featurized')
    prepared_train_data_path = luigi.Parameter(default='/toxic_classifier/data/prepared/train_prepared.csv')
    model_artefacts_folder= luigi.Parameter(default='/toxic_classifier/data/models')
    featurizers_artefacts_folder = luigi.Parameter(default='/toxic_classifier/data/featurizers')
    
    