'''
Author: Julie.Syu
LastEditTime: 2021-06-03
Description: train embedding & tfidf
'''
import pandas as pd
import numpy as np
from gensim import models
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import jieba
from gensim.models import LdaMulticore
#from gensim.models import LdaModel
from features import label2idx
import gensim
import config
from gensim.models import Word2Vec,FastText
from gensim import models 
from gensim.models import KeyedVectors
import re
import pickle 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import transformers
import torch
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig

# class SingletonMetaclass(type):
#     '''
#     @description: singleton
#     '''
#     def __init__(self, *args, **kwargs):
#         self.__instance = None
#         super().__init__(*args, **kwargs)
#
#     def __call__(self, *args, **kwargs):
#         if self.__instance is None:
#             self.__instance = super(SingletonMetaclass,
#                                     self).__call__(*args, **kwargs)
#             return self.__instance
#         else:
#             return self.__instance


#class Embedding(metaclass=SingletonMetaclass):
class Embedding:
    def __init__(self):
        '''
        @description: This is embedding class. Maybe call so many times. we need use singleton model.
        In this class, we can use tfidf,lda, word2vec,doc2v, fasttext, bert,xlnet, word embedding
        @param {type} None
        @return: None
        '''

        #######################################################################
        # load stop words #
        #######################################################################

        
        f = open(config.chstopwords_obj,'rb')
        self.stopWords = [x.strip() for x in pickle.load(f)]

    def load_data(self, path):
          
#         '''
#         @description:Load all data, then do word segmentation
#         @param {type} None
#         @return:None
#         '''
        data = pd.read_csv(path,encoding='utf-8').dropna().reset_index(drop=True)        
        self.train = data['text'].tolist()



    def trainer(self):
        '''
        @description: Train tfidf,  word2vec, fasttext
        @param {type} None
        @return: None
        '''
        #######################################################################
        # TfIdf #
        #######################################################################
        #count_vect : instantiated by tfidfVectorizer
        #https://blog.csdn.net/blmoistawinde/article/details/80816179
        count_vect = TfidfVectorizer(stop_words=self.stopWords,min_df=5,max_df=500,ngram_range=(1,4) )
        #list of string
        self.tfidf = count_vect.fit(self.train)


        #######################################################################
        # Word2Vec, FastText, Doc2Vec #
        #######################################################################
        #instantiated w2v and build vocabulary and trained
        self.train_v = [sample.split() for sample in self.train] #list of token list
        #print("check",self.train)
        #https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial
        #https://radimrehurek.com/gensim/models/word2vec.html
        self.w2v=Word2Vec(min_count=1,
                     window=5,
                     vector_size=600,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=4,
                     sg=1, 
                     hs=0,

                     ns_exponent=0.75,
                     compute_loss=True,
                        cbow_mean=1) #0:cbow 1:skip
        
        self.w2v.build_vocab(self.train_v)
        self.w2v.train(self.train_v,total_examples=self.w2v.corpus_count,epochs=15,report_delay=2)
                 
                 
#         from gensim.models import Phrases

#         # Train a bigram detector.
#         bigram_transformer = Phrases(common_texts)

#         # Apply the trained MWE detector to a corpus, using the result to train a Word2vec model.
#         model = Word2Vec(bigram_transformer[common_texts], min_count=1)

        ######fasttext
        self.fast = FastText( window=5,min_count=1,vector_size=600,workers=4,alpha=0.03,min_alpha=0.0007,sg=1,hs=0,seed=1,word_ngrams =1,sample=6e-5, negative=20, ns_exponent=0.75,cbow_mean=1)
        self.train_f=[[w for w in s if w!=' '] for s in self.train] 

        self.fast.build_vocab(self.train_f)
        self.fast.train(self.train_f,total_examples=self.fast.corpus_count,epochs=15,report_delay=2,compute_loss=True)
        
        #######sentense embedding
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(self.train_v)]
        self.sentense= Doc2Vec(dm=1,vector_size=600, window=3, min_count=2,seed=0,hs=0,workers=4,alpha=0.03,min_alpha=0.0007, sample=6e-5,                                  negative=20, ns_exponent=0.75)
        self.sentense.build_vocab(documents)
        self.sentense.train(documents,total_examples=len(documents),epochs=15,report_delay=2,compute_loss=True)
        

        #######################################################################
        #  LDA model#
        #######################################################################

        self.id2word = gensim.corpora.Dictionary(self.train_v)
        corpus = [self.id2word.doc2bow(text) for text in self.train_v]
        self.lda = LdaMulticore(corpus=corpus,
                         id2word=self.id2word,
                         random_state=100,
                         num_topics=30,
                         passes=10,
                         chunksize=100,#1000
                         batch=False,
                         alpha='asymmetric',
                         decay=0.5,
                         offset=64,
                         eta=None,
                         eval_every=0,
                         iterations=100,
                         gamma_threshold=0.001,
                         minimum_probability=0,
                         per_word_topics=True)

    def saver(self):
        '''
        @description: save all model
        @param {type} None
        @return: None
        '''
        #save tfidf
        joblib.dump(self.tfidf, './model/tfidf')
        # Store just the words + their trained embeddings.(model.wv)

        #save w2v
        self.w2v.wv.save_word2vec_format('./model/w2v_600_sg.bin',binary=False)
        #save fasttext
        self.fast.wv.save_word2vec_format('./model/fast_600_sg.bin',binary=False)
        #save doc2v
        self.sentense.save('./model/sentense_600_dm.bin')#, doctag_vec=True, word_vec=True, binary=False)
        #save lda
        self.lda.save('./model/lda_30')

    def load(self):
        '''
        @description: Load all embedding model
        @param {type} None
        @return: None
        '''
        #load tfidf
        self.tfidf = joblib.load('./model/tfidf')
        #only load the words+trained embeddings
        #load w2v
        self.w2v = models.KeyedVectors.load_word2vec_format('./model/w2v_600_sg.bin', binary=False)
        #ps:Word2Vec.load()  load the whole model
        #load fast,doc2v,lda
        self.fast = models.KeyedVectors.load_word2vec_format('./model/fast_600_sg.bin', binary=False)
        self.sentense=Doc2Vec.load('./model/sentense_600_dm.bin')#, binary=False)
        self.lda = models.ldamodel.LdaModel.load('./model/lda_30')
        
        #######Bert embedding 'bert-base-chinese'
        self.Bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.Bert = BertModel.from_pretrained('bert-base-chinese',output_hidden_states=True)
        self.Bert.eval()
        #######XLnet
        self.xlnet_tokenizer = XLNetTokenizer.from_pretrained("hfl/chinese-xlnet-mid")
        self.xlnet = XLNetModel.from_pretrained("hfl/chinese-xlnet-mid",output_hidden_states=True)
        self.xlnet.eval()
if __name__ == "__main__":
    em = Embedding()
    em.load_data(config.train_embed_file)#train data or other external corpus for embedding 
    em.trainer()
    em.saver()
