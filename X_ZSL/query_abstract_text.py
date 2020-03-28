'''
get abstract text of dbpedia entity (SPQRAL query with the property dbo:abstract)
'''

import json
from SPARQLWrapper import SPARQLWrapper, JSON


sparql = SPARQLWrapper("http://dbpedia.org/sparql")


def loadUri(file):
    with open(file, 'r') as fp:
        lines = fp.readlines()
        wnid2Uri = dict()
        for line in lines:
            line = line.strip().split('\t')
            wnid2Uri[line[0]] = line[3]  # 'wnid': 'entity uri'
    return wnid2Uri

def queryAbstractText(uri):


    query = "SELECT ?a WHERE { <" + str(uri) + "> <http://dbpedia.org/ontology/abstract> ?a }"
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    # print(results)


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

