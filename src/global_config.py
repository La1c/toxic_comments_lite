import luigi

class globalconfig(luigi.Config):
    train_data_path = luigi.Parameter(default='./data/train.csv')