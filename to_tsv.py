import pandas as pd
import config

def convert_to_csv():
    data = pd.read_csv(config.input_file_dict['input_file'])
    data.to_csv('spacy_data_costshare.tsv',sep='\t',index=False)


if __name__== "__main__":
    convert_to_csv()
