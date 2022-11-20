import pandas as pd
import math

import json
import pandas as pd

# import ijson
#
# for prefix, type_of_object, value in ijson.parse(open("../inputs/data-all-json/data-all-2020.json")):
#     print(prefix, value)

csv_file_dir = '../inputs/data-all-csv'
json_file_dir = '../inputs/data-all-json'

for year in range(2010,2021):
    csv_file = csv_file_dir + f'/data-all-{year}.csv'
    json_file = json_file_dir + f'/data-all-{year}.json'
    print(json_file)
    df_json_sample_records_year = pd.DataFrame()
    with open(json_file, "r") as f:
        reader = pd.read_json(f, orient="records", lines=True, chunksize=5)
        for chunk in reader:
            print(len(chunk))
    # for chunk in pd.read_json(json_file, lines=True, chunksize=5, nrows = 5000000000, orient='records'):
    #
    #     chunk = chunk.dropna(subset=['title'])
    #     count = len(chunk)
    #     sample_ratio = 0.05
    #     sample_count = math.ceil(count * sample_ratio)
    #
    #     sample_indexes = np.random.choice(
    #         count,
    #         sample_count
    #     )
    #
    #     sample_record = chunk.loc[sample_index]
    #     sample_records = sample_records_year.append(sample_record, ignore_index=True)
    # print(df_json_sample_records_year.head())
    # df_csv = pd.read_csv(csv_file)
    # df_merged = df_csv.merge(df_json_sample_records_year, left_on='tender_id', right_on='id', how='left', )