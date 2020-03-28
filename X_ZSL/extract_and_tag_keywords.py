'''
We provide an example of extracting keywords, labeling pos tags and labeling ner tags
The processing procedure is as follows:
1. Extract keywords from abstract text using gensim toolkit;
2. Get POS tag and NER tag for each word in text (save in dict);
3. Label each keyword with POS tag and NER tag;
4. Generate natural language sentence based on tagged keywords.
'''

from gensim.summarization import keywords
import spacy
from spacy import displacy
import en_core_web_sm
nlp = en_core_web_sm.load()




def ner_tag(doc):
    words_ner_dict = dict([(X.text, X.label_) for X in doc.ents])
    return words_ner_dict

def pos_tag(doc):
    words_pos_dict = dict([(x.orth_, x.pos_) for x in [y
                                      for y
                                      in doc
                                      if not y.is_stop and y.pos_ != 'PUNCT']])
    return words_pos_dict



if __name__ == '__main__':
    # an abstract text example
    text = 'Zebras (/ˈzɛbrə/ ZEB-rə or /ˈziːbrə/ ZEE-brə) are several species of African equids (horse family) united by their distinctive black and white striped coats. Their stripes come in different patterns, unique to each individual. They are generally social animals that live in small harems to large herds. Unlike their closest relatives the horses and donkeys, zebras have never been truly domesticated. There are three species of zebras: the plains zebra, the Grévy\'s zebra and the mountain zebra. The plains zebra and the mountain zebra belong to the subgenus Hippotigris, but Grévy\'s zebra is the sole species of subgenus Dolichohippus. The latter resembles an ass, to which it is closely related, while the former two are more horse-like. All three belong to the genus Equus, along with other living equids. The unique stripes of zebras make them one of the animals most familiar to people. They occur in a variety of habitats, such as grasslands, savannas, woodlands, thorny scrublands, mountains, and coastal hills. However, various anthropogenic factors have had a severe impact on zebra populations, in particular hunting for skins and habitat destruction. Grévy\'s zebra and the mountain zebra are endangered. While plains zebras are much more plentiful, one subspecies, the quagga, became extinct in the late 19th century – though there is currently a plan, called the Quagga Project, that aims to breed zebras that are phenotypically similar to the quagga in a process called breeding back.'

    # extract keywords (return 10 keywords by default)
    keywords = keywords(text, words=10, lemmatize=True).split('\n')
    print("keywords:", keywords)

    # label each word in text with pos tag and label named entities
    doc = nlp(text)
    words_pos_dict = pos_tag(doc)
    words_ner_dict = ner_tag(doc)


    '''
    label each keyword with a pos tag and if keywords is a name entity
    return: tuple: (keyword, pos tag, ner tag)
    '''
    print("------ keyword labeling ------ ")
    print("%s | %s | %s" % ("keywords", "POS tag", "NER tag"))
    keyword_tag_list = list()
    for word in keywords:
        pos_tag, ner_tag = '', ''
        if len(word.split()) >= 2:
            for wd in word.split():
                if wd in words_pos_dict:
                    pos_tag += words_pos_dict[wd] + " "
        else:
            if word in words_pos_dict:
                pos_tag += words_pos_dict[word]

        if word in words_ner_dict:
            ner_tag = words_ner_dict[word]

        print("%s | %s | %s" % (word, pos_tag, ner_tag))
        keyword_tag_list.append((word, pos_tag, ner_tag))


    '''
    generate explanations with tagged keywords
    '''
    NamedEntityTag = ['LOC', 'GPE']
    print("------ generate explantions ------ ")
    for keyword in keyword_tag_list:
        if keyword[1] == "ADJ":
            print("They are both %s animals." % keyword[0])

        if keyword[1] == "NOUN":
            if keyword[2] and keyword[2] in NamedEntityTag:
                print("They both live in %s." % keyword[0])
                continue
            print("They are both have %s." % keyword[0])

        if keyword[1] == "ADJ NOUN":
            print("They are similar in %s." % keyword[0])
