'''
Match class with DBpedia entity using wordnet ID
input: IMSC.json (unseen classes and their impressive seen classes), words.txt (class wordnet ids and names)
output: matched dbpedia entity (with entity name and entity uri)
'''

import requests
import re
import xml.etree.ElementTree as ET
import json


# load class names
def getName(file):
    file_object = open(file, 'rU')
    wnid_word = dict()  # wnid - class name
    try:
        for line in file_object:
            line = line[:-1]
            wn, name = line.split('\t')
            wnid_word[wn] = name.strip()
    finally:
        file_object.close()
    return wnid_word

# lookup entities from DBPedia
def lookup_resources(name, cat):
    entityURIs = list()
    entityNames = list()

    name_items = list()
    name_brackets = re.findall('\((.*?)\)', name)
    for name_bracket in name_brackets:
        name = name.replace('(%s)' % name_bracket, '')
    name = name.strip()

    if len(name) > 2:
        name_items.append(name)
    for name_bracket in name_brackets:
        if len(name_bracket) > 2:
            name_items.append(name_bracket.strip())
    for name_item in name_items:
        try:
            lookup_url = 'http://lookup.dbpedia.org/api/search/KeywordSearch?MaxHits=2&QueryClass=%s&QueryString=%s' \
                         % (cat, name_item)
            lookup_res = requests.get(lookup_url)
            root = ET.fromstring(lookup_res.content)
            for child in root:
                entity_name = child[0].text  # the first value is label name
                entityNames.append(entity_name)

                entity_uri = child[1].text  # the second value is URI
                entityURIs.append(entity_uri)


        except UnicodeDecodeError:
            pass
    return entityNames, entityURIs






if __name__ == '__main__':
    # load class
    IMSC_file = '../data/X_ZSL/IMSC.json'
    IMSCs = json.load(open(IMSC_file, 'r'))

    class_list = list()
    # each item is an unseen class with its impressive seen classes list
    for unseen, seens in IMSCs.items():
        class_list.append(unseen)
        class_list.extend(seens)
    print('total classes: ', len(class_list))
    # class_list = ['n02127482', 'n02123394', 'n02123597', 'n02439033', 'n02430045', 'n02374451', 'n02391049', 'n02331046', 'n02363005', 'n02342885', 'n02330245', 'n02355227', 'n02064816', 'n02065726', 'n02068974', 'n02071294', 'n02411705', 'n02419796']

    # load class <wnid, name> dict
    wnid_name_file = '../data/AZSL-G/materials/words.txt'
    wnid2name = getName(wnid_name_file)

    # look up DBpedia entity
    items = list()

    for i, wnid in enumerate(class_list):
        text_name = wnid2name[wnid]  # get class name (text name)
        names = text_name.split(',')
        # entitys = []
        entity = ''
        flag = 0  # if get entity?
        for name in names:
            entityNames, entityURIs = lookup_resources(name, 'animal')
            if len(entityURIs):
                uri = entityURIs[0]
                entity_name = entityNames[0]

                line = wnid + "\t" + text_name + "\t" + entity_name + "\t" + uri
                print(i, " ", line)
                items.append(line)
                break

    print("total:", len(items))

    # save_wnid_dbEntity_file = '../data/X_ZSL/wnid-dbEntity.txt'
    ## store matched results
    # wr_fp = open(save_wnid_dbEntity_file, 'w')
    # for l in items:
    #     wr_fp.write('%s\n' % l)
    # wr_fp.close()


