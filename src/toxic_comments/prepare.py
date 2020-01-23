import pandas as pd
from toxic_comments.utils import try_mkdir
import luigi
import os
from toxic_comments.global_config import globalconfig
import logging

logger = logging.getLogger('luigi-interface')

class PreparationTask(luigi.Task):
    input_df_file = luigi.Parameter(globalconfig().train_data_path)
    output_df_folder = luigi.Parameter(globalconfig().preprocessed_data_folder)
    
    
    def clean_df(self, df, column_name):
        """This function removes line breaks from the specified column
    
        Arguments:
            df {pd.DataFrame} -- data to process
            column_name {string} -- dataFrame column to clean
    
        Returns:
            pd.DataFrame -- processed dataframe
        """
        
        df[column_name] = df[column_name].fillna('').str.replace('\n', ' ')
        return df
    
    def output(self):
        input_file_name, extention = self.input_df_file.split(os.path.sep)[-1].split('.')
        self.output_file_name = '.'.join([input_file_name + '_prepared', extention])
        return luigi.LocalTarget(os.path.join(self.output_df_folder, self.output_file_name))
    
    def run(self):
        df = pd.read_csv(self.input_df_file)
        logger.info('processing {}'.format(self.input_df_file))
        df_clean = self.clean_df(df, 'comment_text')
        
        logger.info('writing {} to {}'.format(self.output_file_name, self.output_df_folder))
        with self.output().open('w') as f:
            df_clean.to_csv(f, index=False, encoding='utf-8')
        
if __name__ == "__main__":
    luigi.run()


    