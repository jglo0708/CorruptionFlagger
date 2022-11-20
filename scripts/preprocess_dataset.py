import pandas as pd
import pycld2 as cld2
df = pd.read_csv('../inputs/sample.csv').drop(columns = ['Unnamed: 0'])
df.dropna(subset = ['WIN_NAME', 'CAE_NAME','B_AWARDED_TO_A_GROUP','B_MULTIPLE_CAE'], inplace=True)
#avoid CAE GROUPs and contrator groups
df = df.astype({'WIN_NAME': 'str', 'WIN_TOWN': 'str','WIN_ADDRESS': 'str', })
df = df.astype({'CAE_NAME': 'str','CAE_TOWN': 'str', 'CAE_ADDRESS': 'str'})


def discrepancy(df)
    df['AWARD_EST_DISCREPANCY'] = df['AWARD_EST_VALUE_EURO'] - df['AWARD_VALUE_EURO']
    df['AWARD_EST_DISCREPANCY_RATIO'] = df['AWARD_EST_DISCREPANCY'] / df['AWARD_EST_VALUE_EURO']
    return df


def split_conglomerates(df):
    df['WIN_NAME'] = df['WIN_NAME'].apply(lambda x: x.split('---') if (len(x.split('---')) > 1) else x)
    df['WIN_TOWN'] = df['WIN_TOWN'].apply(lambda x: x.split('---') if (len(x.split('---')) > 1) else x)
    df['WIN_ADDRESS'] = df['WIN_ADDRESS'].apply(lambda x: x.split('---') if (len(x.split('---')) > 1) else x)
    df['WIN_POSTAL_CODE'] = df['WIN_POSTAL_CODE'].apply(lambda x: x.split('---') if (len(x.split('---')) > 1) else x)

    df['CAE_NAME'] = df['CAE_NAME'].apply(lambda x: x.split('---') if (len(x.split('---')) > 1) else x)
    df['CAE_TOWN'] = df['CAE_TOWN'].apply(lambda x: x.split('---') if (len(x.split('---')) > 1) else x)
    df['CAE_ADDRESS'] = df['CAE_ADDRESS'].apply(lambda x: x.split('---') if (len(x.split('---')) > 1) else x)
    df['CAE_POSTAL_CODE'] = df['CAE_POSTAL_CODE'].apply(lambda x: x.split('---') if (len(x.split('---')) > 1) else x)
    return df


def clean_conglomerates(df):
    newrows = []
    oldrows_cleaned = []
    count = 0
    for i, r in df.iterrows():
        if (r['B_AWARDED_TO_A_GROUP'] == 'Y') & (r['B_MULTIPLE_CAE'] == 'N'):
            winners = zip(r['WIN_NAME'], r['WIN_TOWN'], r['WIN_ADDRESS'], r['WIN_POSTAL_CODE'])
            winners = list(winners)
            for winner in winners:
                newrow = r.copy()
                newrow['WIN_NAME'] = winner[0]
                newrow['WIN_TOWN'] = winner[1]
                newrow['WIN_ADDRESS'] = winner[2]
                newrow['WIN_POSTAL_CODE'] = winner[3]
                count += 1
                newrows.append(newrow)
        elif (r['B_MULTIPLE_CAE'] == 'Y') & (r['B_AWARDED_TO_A_GROUP'] == 'N'):
            caes = zip(r['CAE_NAME'], r['CAE_TOWN'], r['CAE_ADDRESS'], r['CAE_POSTAL_CODE'])
            caes = list(caes)
            for cae in caes:
                newrow = r.copy()
                newrow['CAE_NAME'] = cae[0]
                newrow['CAE_TOWN'] = cae[1]
                newrow['CAE_ADDRESS'] = cae[2]
                newrow['CAE_POSTAL_CODE'] = cae[3]
                count += 1
                newrows.append(newrow)
        elif (r['B_MULTIPLE_CAE'] == 'Y') & (r['B_AWARDED_TO_A_GROUP'] == 'Y'):
            winners_caes = zip(r['WIN_NAME'], r['WIN_TOWN'], r['WIN_ADDRESS'], r['WIN_POSTAL_CODE'], r['CAE_NAME'],
                               r['CAE_TOWN'], r['CAE_ADDRESS'], r['CAE_POSTAL_CODE'])
            winners_caes = list(winners_caes)
            for winner_cae in winners_caes:
                newrow = r.copy()
                newrow['WIN_NAME'] = winner_cae[0]
                newrow['WIN_TOWN'] = winner_cae[1]
                newrow['WIN_ADDRESS'] = winner_cae[2]
                newrow['WIN_POSTAL_CODE'] = winner_cae[3]
                newrow['CAE_NAME'] = winner_cae[4]
                newrow['CAE_TOWN'] = winner_cae[5]
                newrow['CAE_ADDRESS'] = winner_cae[6]
                newrow['CAE_POSTAL_CODE'] = winner_cae[7]
                count += 1
                newrows.append(newrow)

        if (r['B_MULTIPLE_CAE'] == 'N') & (r['B_AWARDED_TO_A_GROUP'] == 'N'):
            oldrows_cleaned.append(r.copy())

    df_splitted = pd.DataFrame(newrows).reset_index(drop=True)
    df_oldrows_cleaned = pd.DataFrame(oldrows_cleaned).reset_index(drop=True)
    df_new = df_oldrows_cleaned.append(df_splitted).reset_index(drop=True)
    print('old len:', len(df))
    print('splitted len:', len(df_splitted))
    print('old rows len:', len(df_oldrows_cleaned))
    print(len(df) - len(df_oldrows_cleaned))
    return df_new


import re

'''
Preprocess a string.
:parameter
    :param text: string - name of column containing text
    :param lst_stopwords: list - list of stopwords to remove
    :param flg_stemm: bool - whether stemming is to be applied
    :param flg_lemm: bool - whether lemmitisation is to be applied
:return
    cleaned text
'''

import glob

stopwords_dict = dict()
for file in glob.glob('../inputs/*ST.txt'):
    with open(file, encoding="ISO-8859-1") as f:
        lines = f.read()
    stopwords_dict[file[10:-6]] = lines.split('\n')


def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and
    # characters and then strip
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    ## Tokenize (convert from string to list)
    lst_text = text.split()
    # remove numerals
    lst_text = [word for word in lst_text if word.isalpha()]
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in
                    lst_stopwords]

    #     ## Stemming (remove -ing, -ly, ...)
    #     if flg_stemm == True:
    #         ps = nltk.stem.porter.PorterStemmer()
    #         lst_text = [ps.stem(word) for word in lst_text]

    #     ## Lemmatisation (convert the word into root word)
    #     if flg_lemm == True:
    #         lem = nltk.stem.wordnet.WordNetLemmatizer()
    #         lst_text = [lem.lemmatize(word) for word in lst_text]

    ## back to string from list
    text = " ".join(lst_text)
    return text

def detect_lang_remove_stop(text):
    isReliable, textBytesFound, details, vectors = cld2.detect(
    text, returnVectors=True)
    full_text = ''
    for lang_slice in vectors:
        lang_detected = lang_slice[2].lower()
        print(lang_detected)
        text_slice =text[lang_slice[0]:lang_slice[1]]
        stopwords = stopwords_dict.get(lang_detected)
        text_slice_clean = utils_preprocess_text(text_slice, lst_stopwords = stopwords)
        full_text+=text_slice_clean
    return full_text

df['CAE_all'] =  df['CAE_NAME_notice']+df['CAE_NATIONALID_notice']+df['CAE_ADDRESS_notice']+df['CAE_TOWN_notice']

categorical = ['CAE_all', 'MAIN_ACTIVITY_notice','YEAR_notice','ID_TYPE_notice','CAE_TYPE_notice','TYPE_OF_CONTRACT_notice','TAL_LOCATION_NUTS_notice',
               'EU_INST_CODE_notice','FRA_ESTIMATED_notice','LOTS_SUBMISSION','TOP_TYPE_notice','B_ACCELERATED_notice','CRIT_CODE_notice']
binary = ['CANCELLED_notice', 'B_MULTIPLE_CAE_notice','CORRECTIONS_notice','B_ON_BEHALF_notice','B_INVOLVES_JOINT_PROCUREMENT_notice','B_AWARDED_BY_CENTRAL_BODY_notice','B_MULTIPLE_COUNTRY_notice',
          'B_AWARDED_BY_CENTRAL_BODY_notice','B_FRA_AGREEMENT_notice','B_FRA_SINGLE_OPERATOR','B_DYN_PURCH_SYST_notice','B_GPA_notice','B_VARIANTS','B_EU_FUNDS_notice','B_RENEWALS',
         'B_ELECTRONIC_AUCTION_notice', 'B_LANGUAGE_ANY_EC','B_RECURRENT_PROCUREMENT']
time = ['YEAR_notice','DT_DISPATCH_notice','CONTRACT_START','CONTRACT_COMPLETION','DT_APPLICATIONS','DT_AWARD']
to_delete = ['ISO_COUNTRY_CODE_ALL_notice','XSD_VERSION_notice']
to_OHE = categorical +binary