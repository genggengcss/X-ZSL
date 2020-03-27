'''
query triples with triple pattern
'''

import json
from SPARQLWrapper import SPARQLWrapper, JSON

# invalid relations
stopRelations = ['http://dbpedia.org/ontology/wikiPageWikiLink', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
                 'http://dbpedia.org/property/statusSystem', 'http://dbpedia.org/ontology/wikiPageRedirects',
                 'http://dbpedia.org/property/infraordoAuthority', 'http://dbpedia.org/ontology/binomialAuthority',
                 'http://dbpedia.org/ontology/wikiPageExternalLink', 'http://dbpedia.org/property/imageWidth',
                 'http://dbpedia.org/property/imageCaption']

def loadUri(file):
    with open(file, 'r') as fp:
        lines = fp.readlines()
        wnid2Uri = dict()
        wnid2Name = dict()
        for line in lines:
            line = line.strip().split('\t')
            uri = line[3]
            wnid2Uri[line[0]] = line[3]  # 'wnid': 'entity uri'
            wnid2Name[line[0]] = line[1]
    return wnid2Uri, wnid2Name


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

# rule: (s,r,u)
def queryRule1(unseen, seen):
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
                print("------ > rule1: (%s, %s, %s)" % (seen, query_r, unseen))

# rule: (u,r,s)
def queryRule2(unseen, seen):
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
            print("------ > rule2: (%s, %s, %s)" % (unseen, query_r, seen))

# rule3: (u, r1, t) & (s, r2, t) and rule4: (u, p, v) & (s, p, v)
def queryRule3(unseen, seen):
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
                print("------ > rule3: r1 != r2: ", r1)
            t = result["t"]["value"]
            print("------ > rule3: (%s, %s, %s) & (%s, %s, %s)" % (unseen, r1, t, seen, r2, t))

# rule5: (u, r1, t) & (t, r2, s) or (s, r1, t) & (t, r2, u)
def queryRule5(unseen, seen):
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
            print("------ > rule5: (%s, %s, %s) & (%s, %s, %s)" % (unseen, r1, t, t, r2, seen))

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
            print("------ > rule5: (%s, %s, %s) & (%s, %s, %s)" % (seen, r1, t, t, r2, unseen))


if __name__ == '__main__':
    # load class
    IMSC_file = '../data/X_ZSL/IMSC.json'
    IMSCs = json.load(open(IMSC_file, 'r'))



    # load entity
    Entity_file = '../data/X_ZSL/wnid-dbEntity.txt'
    wnid_entity, wnid_name = loadUri(Entity_file)



    for unseen, seens in IMSCs.items():
        if unseen in wnid_entity:
            print("unseen >", wnid_name[unseen])
            for seen in seens:
                if seen in wnid_entity:
                    # queryRule1(wnid_entity[unseen], wnid_entity[seen])
                    # queryRule2(wnid_entity[unseen], wnid_entity[seen])
                    queryRule3(wnid_entity[unseen], wnid_entity[seen])
                    # queryRule5(wnid_entity[unseen], wnid_entity[seen])



