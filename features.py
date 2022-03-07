'''
Author: Julie.Syu
LastEditTime: 2021-06-03
Description: Feature engineering.
'''
import numpy as np
import pandas as pd
import joblib
import string,pickle
import jieba.posseg as pseg
import jieba
import json
import os
from numpy import dot
from numpy.linalg import norm
import transformers
import torch
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig

def label2idx(data):

    if os.path.exists('./data/label2id.json'):
        labelToIndex = json.load(open('./data/label2id.json',
                                      encoding='utf-8'))
        print("1.labelToIndex",labelToIndex)
    else:
        label = data['label'].unique()
        labelToIndex = dict(zip(label, list(range(len(label)))))
        with open('./data/label2id.json', 'w', encoding='utf-8') as f:
            json.dump({k: v for k, v in labelToIndex.items()}, f)
    return labelToIndex


def get_tfidf(tfidf, data):
    #https://ithelp.ithome.com.tw/articles/10228481

    f = open('./data/stopwords.pickle','rb')
    stopWords = [x.strip() for x in pickle.load(f)]
        

    text = data['text'].apply(lambda x: " ".join([w for w in x.split() if w not in stopWords and w != '']))

    data_tfidf = pd.DataFrame(
        tfidf.transform(
            text.tolist()).toarray())
    data_tfidf.columns = ['tfidf' + str(i) for i in range(data_tfidf.shape[1])]
    data = pd.concat([data, data_tfidf], axis=1)

    return data

def array2df(data, col):
    return pd.DataFrame.from_records(
        data[col].values,
        columns=[col + "_" + str(i) for i in range(len(data[col].iloc[0]))])

def transformer_embedding(sentece,model,tokenizer):
    tokenized_sentence = tokenizer(sentece, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**tokenized_sentence)
    return outputs[0]
    
    
def get_embedding_feature(data, embedding_model,name,emb,transformertoken=None):
    '''
    @description: , word2vec -> max/mean, word2vec n-gram(2, 3, 4) -> max/mean, label embedding->max/mean
    @param {type}
    data, input data set
    @return:
    data, data set
    '''
    #embeding_feature_Y
    labelToIndex = label2idx(data)

    if os.path.exists('./data/'+emb+'_label_embedding_600sg_'+name+'_.pkl'):
        w2v_label_embedding=joblib.load('./data/'+emb+'_label_embedding_600sg_'+name+'_.pkl')

    else:
        if emb in ["bert","xlnet"]:
            w2v_label_embedding = np.array([
                np.mean([
                    torch.mean(transformer_embedding(key,embedding_model,transformertoken),axis=1).detach().numpy()
                    
                ],
                        axis=0) for key in labelToIndex.keys()
            ])[:,0,:]
        elif emb=="doc":
            w2v_label_embedding = np.array([
                np.mean([
                    np.array(embedding_model.infer_vector([key]))
                    
                ],
                        axis=0) for key in labelToIndex.keys()
            ])
        else:
            w2v_label_embedding = np.array([
                np.mean([
                    embedding_model[word] for word in key if word in embedding_model.index_to_key
                ],
                        axis=0) for key in labelToIndex.keys()
            ])



        if name!="predict":
            joblib.dump(w2v_label_embedding,'./data/'+emb+'_label_embedding_600sg_'+name+'_.pkl',protocol=pickle.HIGHEST_PROTOCOL)
    #print("w2v_label_embedding",w2v_label_embedding.shape)
    #embeding_feature_X
    if os.path.exists( './data/'+emb+'generate_feature_aggregation_600sg_'+name+'_.pkl'):
        tmp=joblib.load('./data/'+emb+'generate_feature_aggregation_600sg_'+name+'_.pkl')
    else:

        tmp = data['text'].apply(lambda x: pd.Series(
            generate_feature(x, embedding_model,emb, w2v_label_embedding,transformertoken)))
        if name!="predict":
            joblib.dump(tmp, './data/'+emb+'generate_feature_aggregation_600sg_'+name+'_.pkl',protocol=pickle.HIGHEST_PROTOCOL)
 
    
    
    tmp = pd.concat([array2df(tmp, col) for col in tmp.columns], axis=1)

    data = pd.concat([data, tmp], axis=1)
    if name!="predict":
        joblib.dump(data, './data/'+emb+'feature_complete_600sg_'+name+'_.pkl',protocol=pickle.HIGHEST_PROTOCOL)
    return data


def wam(sentence, embedding_model, emb,transformertoken=None,method='mean', aggregate=True):
    '''
    @description: generate sentence embedding by word average model
    @param {type}
    sentence: sentence
    w2v_model: word2vec model
    method： aggregation by mean or max
    aggregate: aggregate or not
    @return:
    '''
    
    
    #######################################################################
    #          generate embedding array
    #######################################################################

    if emb=="w2v":

        arr= np.array([embedding_model[word] for word in sentence.split() if word in embedding_model.index_to_key])
    elif emb=="fast":

        arr= np.array([embedding_model[word] for word in sentence if word in embedding_model.index_to_key])
    elif emb=="doc":

        arr= np.array(embedding_model.infer_vector([word for word in sentence.split()]))
    elif emb in ["bert","xlnet"]:
        arr=transformer_embedding(sentence,embedding_model,transformertoken)[0,:,:].numpy()

      
 

    if not aggregate:
        return arr
    if len(arr) > 0:
        #######################################################################
        #          aggregation method
        #######################################################################
        # embedding Average
        if method == 'mean':
            return np.mean(np.array(arr),axis=0) #np.mean(arr,axis=0) can work 

        # embedding max
        elif method == 'max':
            return np.max(np.array(arr),axis=0)
        else:
            raise NotImplementedError
    else:
        return np.zeros(300)


def rename_column(data, suffix):
    data.columns += suffix
    return data


def generate_feature(sentence, embedding_model,emb, label_embedding,transformertoken=None):
        '''
        @description: word2vec -> max/mean, word2vec n-gram(2, 3, 4) -> max/mean, label embedding->max/mean
        @param {type}
        data， input data, DataFrame
        label_embedding, all label embedding  ex: 12 labels, dim:300 -> [12*300]
        model_name, w2v means word2vec
        @return: data, DataFrame
        '''


        # get word embedding  and appending without aggregation
              

        w2v = wam(sentence, embedding_model,emb,transformertoken, aggregate=False)  # [seq_len * 300]

        if len(w2v) < 1:
            return {
                'w2v_label_mean': np.zeros(300),
                'w2v_label_max': np.zeros(300),
                'w2v_mean': np.zeros(300),
                'w2v_max': np.zeros(300),
                'w2v_2_mean': np.zeros(300),
                'w2v_3_mean': np.zeros(300),
                'w2v_4_mean': np.zeros(300),
                'w2v_2_max': np.zeros(300),
                'w2v_3_max': np.zeros(300),
                'w2v_4_max': np.zeros(300)
            }
    
        if emb=="doc":
            w2v_label_mean = Find_Label_embedding(w2v, label_embedding,emb, method=None)

            w2v_mean = np.array(w2v)

            return{ 
                'w2v_label_mean': w2v_label_mean,
                'w2v_mean': w2v_mean}
        else:
            
            w2v_label_mean = Find_Label_embedding(w2v, label_embedding,emb, method='mean')
            w2v_label_max = Find_Label_embedding(w2v, label_embedding,emb, method='max')

        # aggregate by mean/max
            w2v_mean = np.mean(np.array(w2v), axis=0)

            w2v_max = np.max(np.array(w2v), axis=0)

            # aggregate by sliding windows
            w2v_2_mean = Find_embedding_with_windows(w2v, 2, method='mean')

            w2v_3_mean = Find_embedding_with_windows(w2v, 3, method='mean')

            w2v_4_mean = Find_embedding_with_windows(w2v, 4, method='mean')

            w2v_2_max = Find_embedding_with_windows(w2v, 2, method='max')

            w2v_3_max = Find_embedding_with_windows(w2v, 3, method='max')

            w2v_4_max = Find_embedding_with_windows(w2v, 4, method='max')
            #print("down w2v_label_mean",w2v_label_mean.shape,w2v_label_mean)#(300,)
            return {
                'w2v_label_mean': w2v_label_mean,
                'w2v_label_max': w2v_label_max,
                'w2v_mean': w2v_mean,
                'w2v_max': w2v_max,
                'w2v_2_mean': w2v_2_mean,
                'w2v_3_mean': w2v_3_mean,
                'w2v_4_mean': w2v_4_mean,
                'w2v_2_max': w2v_2_max,
                'w2v_3_max': w2v_3_max,
                'w2v_4_max': w2v_4_max
            }


def softmax(x):
    '''
    @description: calculate softmax
    @param {type}
    x, ndarray of embedding
    @return: softmax result
    '''
    return np.exp(x) / np.exp(x).sum(axis=0)


def Find_Label_embedding(example_matrix, label_embedding,emb, method='mean'):
    '''
    @description: 根据论文《Joint embedding of words and labels》获取标签空间的词嵌入
    @param {type}
    example_matrix(np.array 2D): denotes words embedding of input
    label_embedding(np.array 2D): denotes the embedding of all label
    @return: (np.array 1D) the embedding by join label and word
    '''

    # Joint Embeddings of Words and Labels-- cosin similiarity of each label and doc embedding
    if emb!="doc":
        similarity_matrix =dot(label_embedding,example_matrix.T)/(norm(label_embedding)*norm(example_matrix))
    
    else:
        similarity_matrix =dot(label_embedding,example_matrix)/(norm(label_embedding)*norm(example_matrix))

    ##########################
    # get attention embedding
    ##########################
    # max-pooling/mean-pooling
    #
    #
    if method == 'mean':
        similarity_matrix=np.mean(similarity_matrix, axis=0)
    elif method == 'max':
        similarity_matrix=np.max(similarity_matrix, axis=0)
    
    #get the attention score by softmax
    attention=softmax(similarity_matrix) #attention(text_len,)
    #print("attention",attention.shape)

    # multiply doc embedding matrix with attention score
    attention_embedding = np.transpose([attention])*example_matrix    #example_matrix (text_len,300) attention(text_len,) np.transpose([attention])(text_len,1)

        
    #print("attention_embedding",attention_embedding.shape) #(text_len,300)
    if method == 'mean':
        return np.mean(attention_embedding, axis=0) 
    elif method == 'max':
        return np.max(attention_embedding, axis=0)
    else:
        res=np.mean(attention_embedding, axis=0) #np.transpose([np.mean(attention_embedding, axis=0)]).T
        #print("atten",res.shape)
        return res


def Find_embedding_with_windows(embedding_matrix, window_size=2,
                                method='mean'):
    '''
    @description: generate embedding use window
    @param {type}
    embedding_matrix, input sentence's embedding
    window_size, 2, 3, 4
    method, max/ mean
    @return: ndarray of embedding
    '''

    result_list = []


    for k1 in range(len(embedding_matrix)):

        #print("example len",len(embedding_matrix),"k1 + window_size",k1 + window_size)
        if int(k1 + window_size) > len(embedding_matrix):

            result_list.append(np.mean(np.array([embedding_matrix[k1+i] for i in range(len(embedding_matrix)-k1)]),axis=0))
        else:

            result_list.append(np.mean(np.array([embedding_matrix[k1+i] for i in range(window_size)]),axis=0))
                
            
    if method == 'mean':
        return np.mean(result_list, axis=0)
    else:
        return np.max(result_list, axis=0)


def get_lda_features_helper(lda_model, document):
    '''
    @description: Transforms a bag of words document to features.
    It returns the proportion of how much each topic was
    present in the document.
    @param {type}
    lda_model: lda_model
    document, input
    @return: lda feature
    '''

    topic_importances=lda_model.get_document_topics(document)
    #print("topic_importances",topic_importances)
    topic_importances = np.array(topic_importances)
    #print("shape",topic_importances.shape,topic_importances[:, 1])
    return topic_importances[:, 1]#[:1]


def get_lda_features(data, LDAmodel):
    if isinstance(data.iloc[0]['text'], str):
        data['text'] = data['text'].apply(lambda x: x.split())
    data['bow'] = data['text'].apply(
        lambda x: LDAmodel.id2word.doc2bow(x))
    data['lda'] = list(
        map(lambda doc: get_lda_features_helper(LDAmodel, doc), data['bow']))
    cols = [x for x in data.columns if x not in ['lda', 'bow']]
    #print("lda",data.columns)
    #print(data["lda"])
    return pd.concat([data[cols], array2df(data, 'lda')], axis=1)


def tag_part_of_speech(data):
    '''
    @description: tag part of speech, then calculate the num of noun, adj and verb
    @param {type}
    data, input data
    @return:
    noun_count,num of noun
    adjective_count, num of adj
    verb_count, num of verb
    '''
    words = [tuple(x) for x in list(pseg.cut(data))]
    noun_count = len(
        [w for w in words if w[1] in ('NN', 'NNP', 'NNPS', 'NNS')])
    adjective_count = len([w for w in words if w[1] in ('JJ', 'JJR', 'JJS')])
    verb_count = len([
        w for w in words if w[1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')
    ])
    return noun_count, adjective_count, verb_count


ch2en = {
    '！': '!',
    '？': '?',
    '｡': '.',
    '（': '(',
    '）': ')',
    '，': ',',
    '：': ':',
    '；': ';',
    '｀': ','
}


def get_basic_feature_helper(text):
    '''
    @description: get_basic_feature, length, capitals number, num_exclamation_marks, num_punctuation, num_question_marks, num_words, num_unique_words .etc
    @param {type}
    df, dataframe
    @return:
    df, dataframe
    '''
    if isinstance(text, str):
        text = text.split()

    queryCut = [i if i not in ch2en.keys() else ch2en[i] for i in text]

    # num of words
    num_words =len(text)
    
    # num pf capitals
    capitals = sum(1 for c in queryCut if c.isupper())
    # the percentage of capitals
    caps_vs_length = capitals / num_words
    # num_of exclamation_marks
    num_exclamation_marks = queryCut.count('!')

    # num_of question_marks
    num_question_marks =queryCut.count('?')

    ## num_of punctuation
    num_punctuation = sum(queryCut.count(w) for w in string.punctuation)


    num_symbols = sum(queryCut.count(w) for w in "*&$%")


    num_unique_words = len(set(w for w in queryCut))

    words_vs_unique = num_unique_words / num_words

    nouns, adjectives, verbs = tag_part_of_speech("".join(text))

    nouns_vs_length = nouns / num_words

    adjectives_vs_length = adjectives / num_words

    verbs_vs_length = verbs / num_words

    count_words_title = len([w for w in queryCut if w.istitle()])

    mean_word_len = np.mean([len(w) for w in queryCut])
    return {
        'num_words': num_words,
        'capitals': capitals,
        'caps_vs_length': caps_vs_length,
        'num_exclamation_marks': num_exclamation_marks,
        'num_question_marks': num_question_marks,
        'num_punctuation': num_punctuation,
        'num_symbols': num_symbols,
        'num_unique_words': num_unique_words,
        'words_vs_unique': words_vs_unique,
        'nouns': nouns,
        'adjectives': adjectives,
        'verbs': verbs,
        'nouns_vs_length': nouns_vs_length,
        'adjectives_vs_length': adjectives_vs_length,
        'verbs_vs_length': verbs_vs_length,
        'count_words_title': count_words_title,
        'mean_word_len': mean_word_len
    }


def get_basic_feature(data):
    tmp = data['text'].apply(
        lambda x: pd.Series(get_basic_feature_helper(x)))
    return pd.concat([data, tmp], axis=1)