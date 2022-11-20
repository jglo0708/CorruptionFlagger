import pandas as pd
import time
import requests
from bs4 import BeautifulSoup
import logging
import pycld2 as cld2
import json


def lang_detection(text):
    isReliable, textBytesFound, details = cld2.detect(
        text
    )
    return details[0][1] == 'en'


def extract_relevant_text(soup):
    all_dvs = soup.findAll("div", {"class": "grseq"})
    text = ''
    for dv in all_dvs:
        if 'Section II: Object' in dv.text:
            for dv_part in dv:
                if dv_part.find("p") is not None:
                    if 'Title:' in dv_part.text:
                        text_titles = dv_part.find("p").text.strip()
                        text += text_titles + '|'
                    if 'Short description:' in dv_part.text:
                        text_s_description = dv_part.find("p").text.strip()
                        text += text_s_description + '|'
                    if 'Description of the procurement:' in dv_part.text:
                        text_descriptions = dv_part.find("p").text.strip()
                        text += text_descriptions + '|'
                    if 'Award criteria' in dv_part.text:
                        text_descriptions = dv_part.find("p").text.strip()
                        text += text_descriptions + '|'

        elif 'Section IV: Procedure' in dv.text:
            for dv_part in dv:
                if dv_part.find("p") is not None:
                    if 'Description:' in dv_part.text:
                        text_procurement_description = dv_part.find("p").text.strip()
                        text += text_procurement_description + '|'
        else:
            continue
    text = text.split('|')
    text = set(text)
    text = ' | '.join(text)
    return text


def get_soup(url):
    notice_url = 'https://' + url
    page = requests.get(notice_url)
    soup = BeautifulSoup(page.text, 'html.parser')
    time.sleep(0.3)
    return soup


def get_content(url):
    notice_url = 'https://' + url
    page = requests.get(notice_url)

    time.sleep(0.3)
    return page.content


def extract_nuts_contractor(soup, contractor_winner):
    all_dvs = soup.findAll("div", {"class": "grseq"})
    for dv in all_dvs:
        if 'Name and address of the contractor' in dv.text:
            for dv_part in dv:
                for dv_part1 in dv_part:
                    for dv_part2 in dv_part1:
                        if 'Official name:' in dv_part2:
                            contractor = dv_part2.split("Official name: ")[1]

            if contractor == contractor_winner:
                contractor_nuts = dv.find("span", {"class": "nutsCode"})
                contractor_nuts = contractor_nuts.text.split()[0]
                return contractor_nuts
            else:
                continue


def extract_nuts_contracting_auth(soup):
    all_dvs = soup.findAll("div", {"class": "grseq"})
    nuts = ''
    for dv in all_dvs:
        if ('Section I: Contracting authority' in dv.text) | ('Section I: Contracting entity' in dv.text):
            nuts_contracting = dv.find("span", {"class": "nutsCode"})
            nuts += nuts_contracting.text.split()[0]

    return nuts


cols = (21, 27, 35, 39, 51, 56, 61, 63, 79, 81, 84, 85, 91, 98, 108, 111)
mixed_types_dict = {x: "string" for x in cols}
logging.basicConfig(filename='../outputs/download_soup.log', level=logging.DEBUG)
logging.info('read file')
with open('../inputs/extracted_text.json') as json_file:
    extracted_text = json.load(json_file)

# sample = pd.read_csv('../inputs/sample_big.csv')
sample = pd.read_csv('../inputs/sample_big.csv', index_col=0, dtype=mixed_types_dict)
logging.info('download soup')

for i, r in sample.iterrows():
    if (i % 1000) == 0:
        with open("../inputs/extracted_text.json", "w") as outfile:
            json.dump(extracted_text, outfile)
        with open('../inputs/extracted_text.json') as infile:
            extracted_text = json.load(infile)
    try:
        if r['ID_NOTICE_CN'] in extracted_text:
            pass
        else:
            soup = get_soup(r['TED_NOTICE_URL_notice'])
            extracted_text[r['ID_NOTICE_CN']] = extract_relevant_text(soup)
    except:
        with open("extracted_text.json", "w") as outfile:
            json.dump(extracted_text, outfile)

# sample['soup'] = sample.apply(lambda x: get_soup(x['TED_NOTICE_URL_notice']),axis=1)
# sample['NUTS_CAE'] = sample.apply(lambda x: extract_nuts_contracting_auth(x['soup']),axis=1)
# # sample['NUTS_WIN'] = sample.apply(lambda x: extract_nuts_contractor(x['soup'], x['WIN_NAME']),axis=1)
# sample.to_csv('../inputs/sample_big_with_noticenuts.csv')

with open("extracted_text.json", "w") as outfile:
    json.dump(extracted_text, outfile)

# with open('../outputs/test_eng_documents.txt', 'a') as f:
#     dfAsString = sample[sample['EN_TEXT']==True]['TEXT'].to_string(header=False, index=False)
#     f.write(dfAsString)
#
# with open('../outputs/test_noneng_documents.txt', 'a') as f:
#     dfAsString = sample[sample['EN_TEXT']==False]['TEXT'].to_string(header=False, index=False)
#     f.write(dfAsString)
