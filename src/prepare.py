import pandas as pd
import re

def clean_df(df, column_name):
    """This function removes line breaks from the specified column
    
    Arguments:
        df {pd.DataFrame} -- data to process
        column_name {string} -- dataFrame column to clean
    
    Returns:
        pd.DataFrame -- processed dataframe
    """
    df[column_name] = df[column_name].fillna('').str.replace('\n', ' ')
    return df


if __name__ == "__main__":
    import sys
    import os

    input_df_path = sys.argv[1]
    input_file_name, extention = input_df_path.split(os.path.sep)[-1].split('.')
    output_file_name = '.'.join([input_file_name + '_prepared', extention])
    output_path = os.path.join('data', 'prepared')

    try:
        os.makedirs(output_path)
    except FileExistsError as e:
        pass

    print('processing {}'.format(input_df_path))
    df = pd.read_csv(input_df_path)
    df_clean = clean_df(df, 'comment_text')
    print('writing {} to {}'.format(output_file_name, output_path))
    df_clean.to_csv(os.path.join(output_path, output_file_name), index=False)


    