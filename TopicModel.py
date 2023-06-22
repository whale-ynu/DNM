'''
@author: 
Hao Wu (haowu@ynu.edu.cn)

@references:
'''

import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import NMF

class TopicModel(object):
    
    def __init__(self, path):
        '''
        :param path: the path for all WSDL documents
        '''
        self.path = path
        self.wsdlfiles = sorted([os.path.join(self.path, fn) for fn in os.listdir(self.path)])
        print("Total %d WSDL documents!" % len(self.wsdlfiles))
    
    def factorize(self, num_topics, _max_df=0.5, _max_features=10000):
        '''
        The document-term matrix is constructed with the weights of term frequency and factored into a term-feature and a feature-document matrix. 
        :param num_topics: the dimensionality of latent topics
        :param _max_df: ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words)
        :param _max_features: If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
        '''    
        print("Parsing WSDL documents...")
        vectorizer = TfidfVectorizer(input='filename', stop_words='english', decode_error='ignore', max_df=0.5, max_features=10000)
        dtm = vectorizer.fit_transform(self.wsdlfiles).toarray()
        self.vocab = np.array(vectorizer.get_feature_names())
        print("Factorizing document-term matrix...")
        self.nmf = NMF(n_components=num_topics, random_state=1)
        doc2topic = self.nmf.fit_transform(dtm)
        # row-based normalization for document-topic distribution 
        sum = np.sum(doc2topic, axis=1, keepdims=True);
        for i in range(len(sum)): 
            if sum[i] == 0: sum[i] = 1 
        self.doc_topics = doc2topic / sum
        
    def getDocumentTopics(self, wsdlfilename):
        '''
        :param wsdlfilename: the basename of a WSDL document銆�
        return: the topic distribution of the given WSDL document
        '''        
        return self.doc_topics[self.wsdlfiles.index(os.path.join(self.path, wsdlfilename))] 
      
    def showTopics(self, num_top_words=50):
        '''
        :param num_top_words: the number of words to be shown
        ''' 
        self.topic_words = []
        for topic in self.nmf.components_:
            word_idx = np.argsort(topic)[::-1][0:num_top_words]
            self.topic_words.append([self.vocab[i] for i in word_idx])
        for t in range(len(self.topic_words)):
            print("Topic {}: {}".format(t, ' '.join(self.topic_words[t][:num_top_words])))
            
    def getSortedDocumentTopics(self, doc_topic):
        '''
        :param doc_topic: the number of words to be shown
        ''' 
        dict=[[doc_topic[i],i] for i in range(len(doc_topic))] 
        dict.sort(reverse=True) 
        return dict
    
    def cosineSimilarity(self, a, b):
        '''
        calculate the cosine similarity between two topic vectors, a and b
        ''' 
        sum1 = 0
        sum2 = 0
        sum3 = 0
        for k in range(len(a)):
            if a[k] > 0 and b[k] > 0:
                sum1 = sum1 + a[k] * b[k]
                sum2 = sum2 + a[k] * a[k]
                sum3 = sum3 + b[k] * b[k]
        if sum1 == 0 or sum2 == 0 or sum3==0: 
            return 0
        else: return sum1 / (sum2+sum3)

if __name__ == '__main__':
   tm = TopicModel("_wsdl/")
   tm.factorize(50, 0.95, 10000)
   tm.showTopics(50)
   print(tm.getDocumentTopics('0.wsdl'))
   print(tm.getSortedDocumentTopics(tm.getDocumentTopics('0.wsdl')))
   print(tm.getSortedDocumentTopics(tm.getDocumentTopics('1.wsdl')))
   print(tm.getSortedDocumentTopics(tm.getDocumentTopics('31.wsdl')))
   print(tm.getSortedDocumentTopics(tm.getDocumentTopics('4000.wsdl')))
   print(tm.getSortedDocumentTopics(tm.getDocumentTopics('3313.wsdl')))
