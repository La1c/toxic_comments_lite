if __name__ == "__main__":
    import argparse
    from bpemb import BPEmb
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pickle
    import os
    import pandas as pd
    from utils import try_mkdir
    from featurizers.mnb_featurizer import MNBFeaturizer

    parser = argparse.ArgumentParser()
    parser.add_argument('input_data_path', type=str)
    parser.add_argument('output_data_path', type=str)
    parser.add_argument('artifacts_path', type=str)
    parser.add_argument('--train', help='Trains featurizer', action='store_true')
    parser.add_argument('--test', help='Transforms test data', action='store_true')
    args = parser.parse_args()

    if args.test:
        print("Generating features")
        print("Reading data at {}".format(args.input_data_path))
        data_df = pd.read_csv(args.input_data_path)
        try_mkdir(args.output_data_path)
        for category in ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]:
            print("Working on category: {}".format(category))
            featurizer = MNBFeaturizer.load(os.path.join(args.artifacts_path, category))
            transformed = featurizer.transform(data_df['comment_text'])
            
            with open(os.path.join(args.output_data_path, '{}_features_test.pkl'.format(category)), 'wb') as f:
                pickle.dump(transformed, f)

    else:
        print("Fitting featurizers")
        print('Reading data from {}'.format(args.input_data_path))
        data_df = pd.read_csv(args.input_data_path)
        bpemb_en = BPEmb(lang="en", dim=50, vs=200000)
        tfidf = TfidfVectorizer(tokenizer=bpemb_en.encode)
        print("Fitting tfidf")
        tfidf.fit(data_df['comment_text'])

        try_mkdir(args.artifacts_path)
        try_mkdir(args.output_data_path)

        for category in ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]:
            featurizer = MNBFeaturizer(tfidf)
            print("Fitting MNB for category {}".format(category))
            transformed = featurizer.fit_transform(data_df['comment_text'], data_df[category])
            with open(os.path.join(args.output_data_path, '{}_features_train.pkl'.format(category)), 'wb') as f:
                pickle.dump(transformed, f)
            
            try_mkdir(os.path.join(args.artifacts_path, category))
            featurizer.save(os.path.join(args.artifacts_path, category))
            


            


            


