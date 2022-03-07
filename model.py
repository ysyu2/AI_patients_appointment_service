'''
Author: Julie.Syu
LastEditTime: 2021-06-03
Description: building model-sampling,parameter tuning
'''

import json,os,copy
import jieba,config
import joblib
import lightgbm as lgb
import pandas as pd
import sklearn.metrics as metrics
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestClassifier
from embedding import Embedding
from features import (get_basic_feature, get_embedding_feature,get_lda_features, get_tfidf,rename_column)
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from random import randint
import pickle 
import transformers
import torch
from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import ClusterCentroids
from scipy.stats import uniform
from data import bert_clean,basic_clean
import numpy as np
import jieba

#define the user token
user_cut==True
if user_cut:
	jieba.load_userdict(config.user_dict)

#define sampling strategy
samplingstrategy_bls_train={'呼吸内科':800,'血液肿瘤科':800,'内分泌科': 800, '脑血管及神经内科':800,'家医科':800,'小儿内科':800,'妇产科':800,'心血管内科':800,'肝胆肠胃科': 800,'耳鼻喉科': 800,'皮肤科':800}
samplingstrategy_smote_train='auto'
samplingstrategy_aday_train='auto'
samplingstrategy_down_train={'风湿免疫科':700,'骨科':700,'泌尿科':700,'眼科':700}

samplingstrategy_bls_test={'呼吸内科':80,'血液肿瘤科':80,'内分泌科': 80, '脑血管及神经内科':80,'家医科':80,'小儿内科':80,'妇产科':80,'心血管内科':80,'肝胆肠胃科': 80,'耳鼻喉科': 80,'皮肤科':80}
samplingstrategy_smote_test='auto'
samplingstrategy_aday_test='auto'
samplingstrategy_down_test={'风湿免疫科':70,'骨科':70,'泌尿科':70,'眼科':70}

#define feature set
emb_list=['fast']  #the key name of self.feature_select_col
basic_list=['tfidf'] #lda,tfidf,POS

#define classifier
cls=lgb.LGBMClassifier(random_state=20,n_estimators=300,learning_rate=0.06,reg_alpha=6)

	
class Classifier:
    #bs_gen:basic embbedding; Transformer_gen:Bert/Xlnet
    def __init__(self,Transformer_gen=True,bs_gen=True, train_mode=False) -> None:
        f = open(config.chstopwords_obj,'rb')

        self.stopWords = [
            x.strip() for x in pickle.load(f)
        ]
        self.embedding = Embedding()
        self.embedding.load()
        self.labelToIndex = json.load(
            open(config.label2id, encoding='utf-8'))
        

        
		 
        if train_mode:
            if Transformer_gen:
                self.train_bert = pd.read_csv(config.train_dataBert_file,encoding='utf-8').dropna().reset_index(drop=True)
                print("init self.train_bert",self.train_bert)
                self.dev_bert = pd.read_csv(config.eval_dataBert_file,encoding='utf-8').dropna().reset_index(drop=True)
                print("init self.dev_bert",self.dev_bert)
                self.test_bert = pd.read_csv(config.test_dataBert_file,encoding='utf-8').dropna().reset_index(drop=True)
                print("init self.test_bert",self.test_bert)
                print("BERT data")
            if bs_gen:
                self.train_bs = pd.read_csv(config.train_dataclean_file,encoding='utf-8').dropna().reset_index(drop=True)
                print("init self.train_bs",self.train_bs)
                self.dev_bs = pd.read_csv(config.eval_dataclean_file,encoding='utf-8').dropna().reset_index(drop=True)
                print("init self.dev_bs",self.dev_bs)
                self.test_bs = pd.read_csv(config.test_dataclean_file,encoding='utf-8').dropna().reset_index(drop=True)
                print("init self.test_bs",self.test_bs)
                print("basic")
        self.exclusive_col = ['text', 'label']
        self.feature_select_col={
                "mean_all":['w2v_label_mean','w2v_mean','w2v_2_mean','w2v_3_mean','w2v_4_mean'],
                "max_all":['w2v_label_max','w2v_max','w2v_2_max','w2v_3_max','w2v_4_max'],
                "label_mean":['w2v_label_mean'],
                "label_max":['w2v_label_max'],
                "v_mean":['w2v_mean'],
                "v_max":['w2v_max'],
                "v_label_mean":["w2v_mean" ,'w2v_label_mean' ],
                "v_label_max":["w2v_max" ,'w2v_label_max'],
                "v_max_label_mean":["w2v_max",'w2v_label_mean'],
				"v_mean_label_max":["w2v_mean",'w2v_label_max'],
                "v_label_mean_max":["w2v_mean","w2v_max",'w2v_label_mean','w2v_label_max'],
                "vs_max_label_mean":['w2v_label_mean' ,'w2v_max' ,'w2v_2_max','w2v_3_max' ,'w2v_4_max'],
				"vs_mean_label_max":['w2v_label_max' ,'w2v_mean' ,'w2v_2_mean','w2v_3_mean' ,'w2v_4_mean']
        }
        self.predict_bert=None
        self.predict_bs=None

    def feature_engineer(self,data1=None,data2=None,name="",emb_list=[],basic_list=[],for_train=True):
	#data1: for Bert/Xlnet; data2:for basic embedding
        print("data1",data1,type(data1),data1.shape)
        print("__")
        print("data2",data2,type(data2),data2.shape)
        final=pd.DataFrame()
        if for_train:
            head=data1.iloc[:,:2]
        else:
            head=data1.iloc[:,:1]
        
        #print("fe be4",data.columns)
        print("head",head)
        for emb in emb_list:
            
            if os.path.exists('./data/'+emb+'feature_complete_600sg_'+name+'_.pkl'):
                tmp=joblib.load('./data/'+emb+'feature_complete_600sg_'+name+'_.pkl')
                print("emb from load",tmp)

            else:
                if emb=="bert":
                    print("in feature engineer",emb)
                    tmp=get_embedding_feature(data1, self.embedding.Bert,name,emb="bert",transformertoken=self.embedding.Bert_tokenizer)
                elif emb=="xlnet":
                    print("in feature engineer",emb)
                    tmp=get_embedding_feature(data1, self.embedding.xlnet,name,emb="xlnet",transformertoken=self.embedding.xlnet_tokenizer)
                elif emb =="w2v":
                    print("in feature engineer",emb)
                    tmp = get_embedding_feature(data2, self.embedding.w2v,name,emb="w2v",transformertoken=None)
                elif emb=="doc":
                    print("in feature engineer",emb)
                    tmp = get_embedding_feature(data2, self.embedding.sentense,name,emb="doc",transformertoken=None)
                elif emb=="fast":
                    print("in feature engineer",emb)
                    tmp = get_embedding_feature(data2, self.embedding.fast,name,emb="fast",transformertoken=None)
            
            if emb in ["bert","xlnet"]:
                selected=self.feature_select_col["v_label_max"]
            elif emb in ["w2v","fast"]:
                selected=self.feature_select_col["max_all"]
                
            elif emb in ["doc"]:
                selected=self.feature_select_col["label_mean"]
            selected=[i for i in tmp.columns for s in selected if s in i]
                    
            tmp=tmp[selected]
            tmp=rename_column(tmp, emb)
            print("emb",emb,tmp.columns,len(tmp.columns))
            final = pd.concat([final, tmp], axis=1)
        

            
        
        for b in basic_list:
            if b=="lda":
                tmp = get_lda_features(data2, self.embedding.lda)
            elif b=="tfidf":
                tmp = get_tfidf(self.embedding.tfidf, data2)
                print('tfidf',tmp.columns,tmp)
            elif b=="POS":
                tmp = get_basic_feature(data2)
            tmp=tmp[[i for i in tmp.columns if i not in self.exclusive_col]]
            
            print("tmp",b,tmp.columns)
            final = pd.concat([final, tmp], axis=1)
        
        data = pd.concat([head,final], axis=1)
        print("fe after",data.columns)
        
        return data
    
    def Grid_Train_model(self,model,Train_feature,Train_label):

        parameters={
        'boosting_type':["gbdt","dart","goss",'df'],
        'max_depth':[-1,3],
         'random_state':[0,10,100],
        'learning_rate':[0.001,0.01,0.06,0.1,0.5,1.5,2,2.5,3],
        'n_estimators':[50,100,500,600,1000],
        'subsample':[0.5,1],
        'colsample_bytree':[0.5,1],
        'reg_alpha':[0,0.05,1,3,6,10],
        'reg_lambda':[0,0.05,1,5,8,10],
        'objective':['binary','multiclass']
        
        }

        gsearch=GridSearchCV(model, parameters,scoring='f1_macro',cv=3,n_jobs=-1,verbose=10,return_train_score=False)
        gsearch.fit(Train_feature,Train_label)

        print("best patameters set:{}".format(gsearch.best_params_))
        print("Train score:",gsearch.score_samples(Train_feature))
        return gsearch
    def Bay_Train_model(self,model,Train_feature,Train_label):
        parameters={
        'learning_rate':Real(1e-4, 1e+1, prior='log-uniform'),
        'max_depth':Integer(-1,10),
        'random_state':Integer(0,1),
        'n_estimators':Integer(100,10000),
        'reg_alpha':Real(0,10),
        'reg_lambda':Real(0,10)}
                    
        bsearch=BayesSearchCV(model,search_spaces=parameters,
                          scoring='f1_macro',cv=3,n_jobs=-1,random_state=0,verbose=10,return_train_score=True)
        bsearch.fit(Train_feature,Train_label)

        print("best patameters set:{}".format(bsearch.best_params_))
        print("Train score:",bsearch.score_samples(Train_feature))
        return bsearch
    
    def Rand_Train_model(self,model,Train_feature,Train_label):

        parameters={
       
        'random_state':[0,100,500,1000,2000],
        'max_depth': [15,10,5,3,-1],
        'learning_rate':uniform(1e-4, 1e+1),
        'n_estimators':[80,100,200,300,500,600,800],
        'reg_alpha':uniform(0,10),
        'reg_lambda':uniform(0,10),

        }
        rsearch=RandomizedSearchCV(model, parameters,scoring='f1_macro',cv=10,n_jobs=-1,n_iter=100,verbose=10,random_state=10,return_train_score=True)
        rsearch.fit(Train_feature,Train_label)

        print("best patameters set:{}".format(rsearch.best_params_))
        print("Train score:",rsearch.best_score_)
	
        return rsearch

    def trainer(self,cls,emb_list,basic_list,grid_search,bay_search,rand_search,fit_bestp,smote,blsmote,aday,down_cluster,balanced):
        name="train"
        train = self.feature_engineer(self.train_bert,self.train_bs,name,emb_list,basic_list,for_train=True)
        train=train.dropna(axis=0).reset_index(drop=True)
        

        name="test"
        dev = self.feature_engineer(self.dev_bert,self.dev_bs,name,emb_list,basic_list,for_train=True)
        dev=dev.dropna(axis=0).reset_index(drop=True)
        #print("dev",dev[-50:])

        cols = [x for x in train.columns if x not in self.exclusive_col] #['text',  'label']
        X_train = train[cols]
        y_train = train['label']

        X_test = dev[cols]
        y_test = dev['label']
        print("orig_len",len(X_train))

        ###synthentic dataset (oversample,downsample,balanced)
        if smote:
            
            X_train, y_train = SMOTE(random_state=42,sampling_strategy=samplingstrategy_smote_train,k_neighbors=5).fit_resample(X_train, y_train)
            X_test, y_test = SMOTE(random_state=42,sampling_strategy=samplingstrategy_smote_test,k_neighbors=5).fit_resample(X_test, y_test)
            print("smote_len",len(X_train),len(X_test))

        if aday:
            
            X_train, y_train = syn.fit_resample(ADASYN(random_state=42,sampling_strategy=samplingstrategy_aday_train,k_neighbors=5),X_train, y_train)
            X_test, y_test = syn.fit_resample(ADASYN(random_state=42,sampling_strategy=samplingstrategy_aday_test,k_neighbors=5),X_test, y_test)
            print("aday_len",len(X_train),len(X_test))
        if blsmote:
            
            X_train, y_train = BorderlineSMOTE(random_state=100,sampling_strategy=samplingstrategy_bls_train,k_neighbors=5).fit_resample(X_train, y_train)
            X_test, y_test = BorderlineSMOTE(random_state=100,sampling_strategy=samplingstrategy_bls_test,k_neighbors=5).fit_resample(X_test, y_test)
            print("bls_len",len(X_train),len(X_test))
			
        if down_cluster:
            
            X_train, y_train = ClusterCentroids(random_state=42,sampling_strategy=samplingstrategy_down_train,voting="hard").fit_resample(X_train, y_train)
            X_test, y_test = ClusterCentroids(random_state=42,sampling_strategy=samplingstrategy_down_test,voting="hard").fit_resample(X_test, y_test)
            print("down_len",len(X_train),len(X_test))

        
        ###tune parameter
        
        if grid_search:
            
            optim_cls=self.Grid_Train_model(cls,X_train,y_train)
       
        elif bay_search:
            
            optim_cls=self.Bay_Train_model(cls,X_train,y_train)
            
        elif rand_search:
            
            optim_cls=self.Rand_Train_model(cls,X_train,y_train)
            

            
        ###initiate multilabel
        mlb = MultiLabelBinarizer()  
        

        y_train=y_train.apply(lambda row: [i.strip() for i in row.strip().split('；') if len(i)>0])
        y_test=y_test.apply(lambda row: [i.strip() for i in row.strip().split('；') if len(i)>0])
		
        #print("test label",y_test[-50:])
        print("check_____________")
        #print("train",train)
        #print("train label",y_train[-50:])
        print("check",X_train.columns)
        y_train = mlb.fit_transform([r for i,r in y_train.iteritems() ])
        y_test = mlb.transform([r for i,r in y_test.iteritems() ])
        c=list(mlb.classes_)		
        self.ix2label={str(k):v for k,v in zip(range(len(c)),c)}
        with open(config.id2label, 'w', encoding='utf-8') as f:
            json.dump(self.ix2label, f)
        print("___")
        print(y_test[:5])
        
        print('X_train: ', X_train.shape,'y_train: ', y_train.shape)
        

        ### train and fit
        
        if fit_bestp:
            best_p=optim_cls.best_params_
            print("best_parameter_result",optim_cls.cv_results_,"best_parameter",best_p)
            
            cls=optim_cls

            #cls= LogisticRegression()#random_state=0,max_iter=400,solver='newton-cg',C=1)
            #0.06, learning_rate=0.04516695623193222,max_depth=6,reg_alpha=6.2270975262943695,                                       reg_lambda=6.466799410716572)
        print("cls",cls)
        self.clf_BR = BinaryRelevance(cls,require_dense = [False, True])
        #print('X_train: ',X_train[-50:])
        #print('X_test: ',X_test[-50:])
        print(X_test[:5])
        print(y_test[:5])
        self.clf_BR.fit(X_train,y_train)
        prediction = self.clf_BR.predict(X_test)
		
        res=[]
        prediction1=prediction.toarray()
        for row in prediction1 :
            r=[]
            #print(row)
            for i in range(len(row)):
                if row[i]>0:
                   r.append(self.ix2label[str(i)])
            res.append(r)
        res=pd.DataFrame({'predict':res})
        res=pd.concat([self.dev_bs,res],axis=1)
        res.to_csv('predict_check_eval.csv')
        
        res=[]
        prediction_train=self.clf_BR.predict(X_train)
        prediction1=prediction_train.toarray()
        for row in range(len(prediction1)) :
            r=[]
            if sum(prediction1[row])==0:
               max_idx= np.argmax(self.clf_BR.predict_proba(X_train.iloc[row]).toarray()[0])
               r.append(self.ix2label[str(max_idx)])		   

            else:
               for i in range(len(prediction1[row])):
                    if prediction1[row][i]>0:
                       r.append(self.ix2label[str(i)])
					

            res.append(r)
        res=pd.DataFrame({'predict':res})
        res=pd.concat([self.train_bs,res],axis=1)
        res.to_csv('predict_check_train.csv')
		
        print("test",metrics.accuracy_score(y_test, prediction))
        print("train",metrics.accuracy_score(y_train,prediction_train ))
        
    def save(self):
        joblib.dump(self.clf_BR, config.model)

    def load(self):
        print("be4 load")
        self.model = joblib.load(config.model)
        print("after",self.model)
        return self.model
    def predict(self, text,model,emb_list,basic_list):
        self.ix2label = json.load(
            open(config.id2label, encoding='utf-8'))
        
        df = pd.DataFrame([[text]], columns=['text'])
        
        df['text'] = df['text'].apply(lambda x: " ".join(
            [w for w in jieba.lcut(x) if w not in self.stopWords and w != '']))
        df1=copy.deepcopy(df)
        print("be4 df",df)

        self.predict_bert=bert_clean(df)
        print("bert df",self.predict_bert,self.predict_bert.shape)
        
        
        name="predict"
        self.predict_bs=basic_clean(df1)
        print("basic df",self.predict_bs,self.predict_bs.shape)
        df=self.feature_engineer(data1=self.predict_bert,data2=self.predict_bs,name=name,emb_list=emb_list,basic_list=basic_list,
                                     for_train=False)
    
        print("after featureenguneer",df.head(),df.shape)
        print("column",df.columns)
        cols = [x for x in df.columns if x not in self.exclusive_col]
        #print("after",cols)
        print(df[cols])

        pred = model.predict(df[cols]).toarray()[0]
        print(type(pred),pred)
        print(self.ix2label)    
        res=[]
        r=[]
        
        if sum(pred)==0:
           pred_=np.where(model.predict_proba(df[cols]).toarray()[0]>0.09, 1, 0)
           print(model.predict_proba(df[cols]).toarray()[0])
           if sum(pred_)!=0:
              for i in range(len(pred_)):
                  if pred_[i]>0:
                     r.append(self.ix2label[str(i)])
           else:
               max_idx= np.argmax(model.predict_proba(df[cols]).toarray()[0])
               r.append(self.ix2label[str(max_idx)])		   

        else:
           print(pred)
           for i in range(len(pred)):
                if pred[i]>0:
                    r.append(self.ix2label[str(i)])
		
        res.append(r)
        return res


if __name__ == "__main__":
    bc = Classifier(Transformer_gen=True,bs_gen=True,train_mode=True)
    bc.trainer(cls=cls,emb_list=emb_list,basic_list=basic_list,grid_search=False,bay_search=False,rand_search=False,
               fit_bestp=False,smote=False,blsmote=True,aday=False,down_cluster=False,balanced=False)
    bc.save()
