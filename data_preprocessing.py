import pandas
import pandas as pd
import json

if __name__ == "__main__":
    df = pandas.read_json('data/yelp_academic_dataset_business.json', lines=True)
    df[df['city'] == 'Austin'].to_csv('Austin.csv')
    df[df['city'] == 'Orlando'].to_csv('Orlando.csv')
    print(df['city'].value_counts())


