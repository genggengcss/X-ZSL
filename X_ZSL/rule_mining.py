'''
We give some examples of unseen class and its impressive seen classes to illustrate the Evidence Mining algorithm which extracts knowledge from Attribute Graph.
Including:
unseen  seen
dolphin killer+whale
horse   zebra
rat (hamster, beaver, mouse)
'''
import json
from itertools import chain, combinations
from collections import defaultdict


def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet, fredSetList):
        """calculates the support for items in the itemSet and returns a subset
       of the itemSet each of whose elements satisfies the minimum support"""
        _itemSet = set()
        localSet = defaultdict(int)

        for item in itemSet:
                for transaction in transactionList:
                        if item.issubset(transaction[0]):
                                freqSet[item] += 1
                                fredSetList[item].append(transaction[1])
                                localSet[item] += 1

        for item, count in localSet.items():
                support = float(count)/len(transactionList)

                if support >= minSupport:
                        _itemSet.add(item)

        return _itemSet


def joinSet(itemSet, length):
        """Join a set with itself and returns the n-element itemsets"""
        return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])


def getItemSetTransactionList(data_iterator):
    transactionList = list()
    # transactionLabel = dict()
    itemSet = set()
    for att, cls in data_iterator.items():
        # transactionLabel[cls] = att
        transaction = frozenset(cls)
        transactionList.append((transaction, att))

        for item in transaction:
            itemSet.add(frozenset([item]))              # Generate 1-itemSets
    return itemSet, transactionList


def runApriori(data, minSupport, minConfidence):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - rules ((pretuple, posttuple), confidence)
    """
    itemSet, transactionList = getItemSetTransactionList(data)

    freqSet = defaultdict(int)
    freqSetList = defaultdict(list)
    largeSet = dict()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport

    assocRules = dict()
    # Dictionary which stores Association Rules

    oneCSet = returnItemsWithMinSupport(itemSet,
                                        transactionList,
                                        minSupport,
                                        freqSet, freqSetList)

    currentLSet = oneCSet
    k = 2
    while(currentLSet != set([])):
        largeSet[k-1] = currentLSet
        currentLSet = joinSet(currentLSet, k)
        currentCSet = returnItemsWithMinSupport(currentLSet,
                                                transactionList,
                                                minSupport,
                                                freqSet, freqSetList)
        currentLSet = currentCSet
        k = k + 1

    def getSupport(item):
            """local function which Returns the support of an item"""
            return float(freqSet[item])/len(transactionList)
    def getSupportList(item):
            """local function which Returns the support of an item"""
            return freqSetList[item]
    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item))
                           for item in value])

    toRetRules = []
    for key, value in list(largeSet.items())[1:]:
        for item in value:
            _subsets = map(frozenset, [x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)
                if len(remain) > 0:
                    confidence = getSupport(item)/getSupport(element)
                    if confidence >= minConfidence:
                        toRetRules.append(((tuple(element), tuple(remain)),
                                           confidence, getSupportList(item)))
    return toRetItems, toRetRules


def printResults(items, rules):
    """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
    for item, support in sorted(items, key=lambda(item, support): support):
        print("item: %s , %.3f" % (str(item), support))

    print ("\n------------------------ RULES:")

    for rule in rules:
        pre, post = rule[0]
        if len(post) == 1 and str(post[0]) == unseen_name:
            print("Rule: {%s} ==> {%s} , confidence value : %.3f" % (str(pre), str(post), rule[1]))
            print("Support attributes (common attributes): ", set(rule[2]))
            print("\n")


def dataFromFile(fname):
        """Function which reads from the file and yields a generator"""
        file_iter = open(fname, 'rU')
        for line in file_iter:
                line = line.strip().rstrip(',')  # Remove trailing comma
                record = frozenset(line.split(','))
                yield record

def extract_attributes(graph, cls):
    att_list = list()
    name = ''
    for triple in graph:
        if triple['o']['id'] == cls:
            name = triple['o']['name']
            att_list.append(triple['v'])
    return att_list, name

if __name__ == '__main__':
    # load class
    example_class = {"n02331046": ["n02363005", "n02342885", "n02330245"]}  # rat & (hamster, beaver, mouse)
    # example_class = {'n02374451': ['n02391049']}  # horse & zebra
    # example_class = {'n02068974': ['n02071294']}  # dolphin & killer+whale
    unseen_name = ''

    '''
    extract attributes from Attribute Graph
    '''
    # load attribute graph
    AttributeGraph = '../data/X_ZSL/AttributeGraph.json'
    AG = json.load(open(AttributeGraph, 'r'), encoding='utf-8')
    graph = AG['@graph']

    cls_att_dict = dict()
    for unseen, seens in example_class.items():
        att_list, name = extract_attributes(graph, unseen)
        unseen_name = name
        if '' in att_list:
            att_list.remove('')  # remove the empty value
        cls_att_dict[name] = att_list
        print("unseen att num:", len(att_list))
        for seen in seens:
            att_list, name = extract_attributes(graph, seen)
            if '' in att_list:
                att_list.remove('')  # remove the empty value
            cls_att_dict[name] = att_list
            print("seen att num:", len(att_list))
    # print(cls_att_dict)

    '''
    construct transaction datatset:  < attribute - class list>
    '''
    att_cls_dict = defaultdict(list)
    for cls, atts in cls_att_dict.items():
        for att in atts:
            att_cls_dict[att].append(cls)
    # print(att_cls_dict)

    '''
    mine rule using Apriori algorithm
    '''
    minSupport = 0.1
    minConfidence = 0.3

    items, rules = runApriori(att_cls_dict, minSupport, minConfidence)

    printResults(items, rules)


