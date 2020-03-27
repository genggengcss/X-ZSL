'''
get abstract text of dbpedia entity (querying "")
'''
import urllib
import os
from urllib.parse import quote
from urllib.request import Request
from urllib.request import urlopen
import json
from SPARQLWrapper import SPARQLWrapper, JSON

def loadUri(file):
    with open(file, 'r') as fp:
        lines = fp.readlines()
        wnid2Uri = dict()
        for line in lines:
            line = line.strip().split('\t')
            uri = line[3]
            wnid2Uri[line[0]] = line[3]  # 'wnid': 'entity uri'
    return wnid2Uri




# Abstract = []
#
# for i, uri in enumerate(URI):
#
#     download_file = "Abstract.txt"
#     with open(download_file, 'w') as fw:
#         query = "SELECT ?a WHERE { <" + str(uri) + "> <http://dbpedia.org/ontology/abstract> ?a }"
#         f = "&format=application%2Fsparql-results%2Bjson"
#         escapeQuery = quote(query)
#         requestURL = endpointURL + "?query=" + escapeQuery + f
#         request = urlopen(requestURL)
#         print(request.read())
#         abstract = request.read().decode('utf-8')
#         fw.write(str(uri) + '\t' + str(abstract) + '\n')
#         print(i, uri)

sparql = SPARQLWrapper("http://dbpedia.org/sparql")
def queryAbstractText(uri):


    query = "SELECT ?a WHERE { <" + str(uri) + "> <http://dbpedia.org/ontology/abstract> ?a }"
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    print(results)


    for result in results["results"]["bindings"]:
        if result["a"]["xml:lang"] == 'en':
            print(result["a"]["value"])




if __name__ == '__main__':
    # load class
    IMSC_file = '../data/X_ZSL/IMSC.json'
    IMSCs = json.load(open(IMSC_file, 'r'))

    class_list = list()
    for unseen, seens in IMSCs.items():
        class_list.append(unseen)
        class_list.extend(seens)
    print('total classes: ', len(class_list))

    # load entity
    Entity_file = '../data/X_ZSL/wnid-dbEntity.txt'
    wnid_entity = loadUri(Entity_file)

    for wnid in class_list:
        if wnid in wnid_entity:
            entity_uri = wnid_entity[wnid]
            queryAbstractText(entity_uri)

