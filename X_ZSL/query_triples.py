'''
query triples with triple pattern
input: IMSC.json (unseen classes and their impressive seen classes), wnid-dbEntity.txt (matched entities)
'''

import json
from SPARQLWrapper import SPARQLWrapper, JSON

# some invalid relations
stopRelations = ['http://dbpedia.org/ontology/wikiPageWikiLink', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
                 'http://dbpedia.org/property/statusSystem', 'http://dbpedia.org/ontology/wikiPageRedirects',
                 'http://dbpedia.org/property/infraordoAuthority', 'http://dbpedia.org/ontology/binomialAuthority',
                 'http://dbpedia.org/ontology/wikiPageExternalLink', 'http://dbpedia.org/property/imageWidth',
                 'http://dbpedia.org/property/imageCaption', 'http://dbpedia.org/property/ordoAuthority']
sparql = SPARQLWrapper("http://dbpedia.org/sparql")


def loadUri(file):
    with open(file, 'r') as fp:
        lines = fp.readlines()
        wnid2Uri = dict()
        wnid2Name = dict()
        for line in lines:
            # each line consists of [wnid, class name, entity name, entity URI]
            line = line.strip().split('\t')
            wnid2Uri[line[0]] = line[3]  # 'wnid': 'entity uri'
            wnid2Name[line[0]] = line[1]
    return wnid2Uri, wnid2Name



# pattern 1: (s,r,u)
def queryPattern1(unseen, seen):
        query = "SELECT ?r WHERE { <" + str(seen) + "> ?r <" + str(unseen) + ">}"
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        # print("query 1:", results)
        if results["results"]["bindings"]:
            for result in results["results"]["bindings"]:
                query_r = result["r"]["value"]
                if query_r in stopRelations:
                    continue
                print("------ > pattern 1: (%s, %s, %s)" % (seen, query_r, unseen))

# pattern 2: (u,r,s)
def queryPattern2(unseen, seen):
    query = "SELECT ?r WHERE { <" + str(unseen) + "> ?r <" + str(seen) + ">}"
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    # print("query 2:", results)
    if results["results"]["bindings"]:
        for result in results["results"]["bindings"]:
            query_r = result["r"]["value"]
            if query_r in stopRelations:
                continue
            print("------ > pattern 2: (%s, %s, %s)" % (unseen, query_r, seen))

# pattern3 : (u, r1, t) & (s, r2, t) and pattern4: (u, p, v) & (s, p, v)
def queryPattern3(unseen, seen):
    query = "SELECT ?r1 ?r2 ?t WHERE { <" + str(unseen) + "> ?r1 ?t. <" + str(seen) + "> ?r2 ?t.}"
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    # print("query 3:", results)
    if results["results"]["bindings"]:
        for result in results["results"]["bindings"]:
            # print(result)
            r1 = result["r1"]["value"]
            r2 = result["r2"]["value"]
            if r1 in stopRelations or r2 in stopRelations:
                continue
            if r1 != r2:
                print("------ > pattern 3: r1 != r2: ", r1)
            t = result["t"]["value"]
            print("------ > pattern 3: (%s, %s, %s) & (%s, %s, %s)" % (unseen, r1, t, seen, r2, t))

# pattern5: (u, r1, t) & (t, r2, s) or (s, r1, t) & (t, r2, u)
def queryPattern5(unseen, seen):
    query = "SELECT ?r1 ?r2 ?t WHERE { <" + str(unseen) + "> ?r1 ?t. ?t ?r2 <" + str(seen) + ">.}"
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    # print("query 3:", results)
    if results["results"]["bindings"]:
        for result in results["results"]["bindings"]:
            # print(result)
            r1 = result["r1"]["value"]
            r2 = result["r2"]["value"]
            if r1 in stopRelations or r2 in stopRelations:
                continue
            t = result["t"]["value"]
            print("------ > pattern 5: (%s, %s, %s) & (%s, %s, %s)" % (unseen, r1, t, t, r2, seen))

    query2 = "SELECT ?r1 ?r2 ?t WHERE { <" + str(seen) + "> ?r1 ?t. ?t ?r2 <" + str(unseen) + ">.}"
    sparql.setQuery(query2)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    # print("query 3:", results)
    if results["results"]["bindings"]:
        for result in results["results"]["bindings"]:
            # print(result)
            r1 = result["r1"]["value"]
            r2 = result["r2"]["value"]
            if r1 in stopRelations or r2 in stopRelations:
                continue
            t = result["t"]["value"]
            print("------ > pattern 5: (%s, %s, %s) & (%s, %s, %s)" % (seen, r1, t, t, r2, unseen))


if __name__ == '__main__':
    # load class
    IMSC_file = '../data/X_ZSL/IMSC.json'
    IMSCs = json.load(open(IMSC_file, 'r'))

    # load matched entities
    Entity_file = '../data/X_ZSL/wnid-dbEntity.txt'
    wnid_entity, wnid_name = loadUri(Entity_file)

    # each item is an unseen class with its impressive seen classes list
    for unseen, seens in IMSCs.items():
        if unseen in wnid_entity:
            print("unseen >", wnid_name[unseen])
            for seen in seens:
                if seen in wnid_entity:
                    queryPattern1(wnid_entity[unseen], wnid_entity[seen])
                    queryPattern2(wnid_entity[unseen], wnid_entity[seen])
                    queryPattern3(wnid_entity[unseen], wnid_entity[seen])
                    queryPattern5(wnid_entity[unseen], wnid_entity[seen])



