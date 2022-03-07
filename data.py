'''
Author: Julie Syu
LastEditTime: 2021-06-03
Desciption: Process data.
'''
import config
import pickle,copy
import pandas as pd
import re,os,json

id2label = {}
with open(config.label_ids_file, 'r',encoding='utf-8') as txt:
    for line in txt:
        ID, label = line.strip().split('\t')
        id2label[ID] = label


#create train,eval,test file
filelist = [config.train_data_file, config.eval_data_file, config.test_data_file]
while True:
    if all([os.path.exists(f) for f in filelist]):
        break
    else:
        for filepath in [config.train_raw_file,config.eval_raw_file,config.test_raw_file]:
            samples = []
            with open(filepath, 'r',encoding='utf-8') as txt:
                for line in txt:
                    ID, text = line.strip().split('\t')
                    label = id2label[ID]
                    sample = label+'\t'+text
                    samples.append(sample)

            if 'train' in filepath:    
                outfile = config.train_data_file
            if 'eval' in filepath:
                outfile = config.eval_data_file
            if 'test' in filepath:
                outfile = config.test_data_file



            with open(outfile, 'w',encoding='utf-8') as csv:
                csv.write('label\ttext\n')
                for sample in samples:
                    csv.write(sample)
                    csv.write('\n')
#stop words
with open(config.chstopwords,'r',encoding='utf-8') as txt:
    stop=[]
    for l in txt:
        l=l.split("\n")[0]
        stop.append(l)
    stop.append(l)
    new=['//item.yiyaojd.com/NUM.html',
         '//itemmjdcom/product/NUMhtml',
          '//itemyiyaojdcom/NUMhtml',
'b...',
'//item.m.jd.com/product/NUM.html',
'-NUM/NUM-NUM/NUM-NUM/NUM-NUM/NUM-NUM/NUM-NUM/NUM-NUM/NUM-NUM/NUM-NUM/NUM-NUM/NUM',
'_',
'`',
'∵',
'^_^',
'≧',
'╯',
'▂╰',
'〇',
'〈',
'「',
'{',
'|',
'~',
'~d',
'°',
'Д',
'–',
'―',
'――',
'…',
'o^^o',
'wwwwwsdffffgggfgggdsd-vgfdssff-ghggh','谢谢','谢谢','多谢','NUM','感觉','医院','你好','您好','医生','药','现在','以前','之前','最近','时间','请','请问',"||","都","很","问题",""," ","特别","效果"
        ,"后","会","去","有点","需要","处方","还","症状","经常",'有时候', '比较', '有时', '情况','再','左右', '容易','导致','应该','今天', '已经','本人', '已', '身体','建议','我的',  '后来', '原因']

    stop=stop+new

with open(config.chstopwords_obj , 'wb') as fp:
    pickle.dump(stop, fp)

    


####basic clean
def basic_clean(data):
    #          remove stopwords #
    f = open(config.chstopwords_obj,'rb')
    stopWords = [x.strip() for x in pickle.load(f)]
    def clean(str_):
        str_=re.sub(r"^\*+","",str_)
        str_=re.sub(r"^\!+","",str_)
        str_=re.sub(r"^\`+","",str_)

        str_=re.sub(r"^\'+","",str_)
        str_=re.sub(r"\^+","",str_)
        str_=re.sub(r"^\_+","",str_)
        str_=re.sub(r"^\!+","",str_)
        str_=re.sub(r"^\-+","",str_)
        str_=re.sub(r"\.+","",str_)
        str_=re.sub(r"^\?+","",str_)
        return str_.strip()
    data["text"]=data["text"].str.split(' ').apply(lambda x: " ".join([clean(k).strip() for k in x if ((clean(k) not in
    stopWords) and ("NUM" not in clean(k))) and (clean(k)!="")]))
    
    return data

####### clean for BERT

def bert_clean(data):
    #          remove stopwords #
    f = open(config.chstopwords_obj,'rb')
    stopWords = [x.strip() for x in pickle.load(f)]
    stopWords.remove(",")
    stopWords.remove("||")
    #print("BERT")
    def clean(str_):
                #str_=re.sub(r"^\||+",",",str_)
                #str_=re.sub(r"^\!+","",str_)
                str_=re.sub(r"^\`+","",str_)
                str_=re.sub(r"^\，+",",",str_)

                str_=re.sub(r"^\'+","",str_)
                str_=re.sub(r"\^+","",str_)
                str_=re.sub(r"^\_+","",str_)
                #str_=re.sub(r"^\!+","",str_)
                str_=re.sub(r"^\-+","",str_)
                #str_=re.sub(r"\.+","",str_)
                #str_=re.sub(r"^\?+","",str_)
                return str_.strip()
    data["text"]=data["text"].str.split(' ').apply(lambda x: " ".join([clean(k).strip() for k in x if ((clean(k) not in stopWords) and ("NUM" not in clean(k))) and (clean(k)!="")]))
    #print("in Bert",data)
    data["text"]=data["text"].apply(lambda x: "".join(",".join(x.split("||")).split(" ")))
    
    return data

            

                
filelist = [config.train_dataclean_file, config.eval_dataclean_file, config.test_dataclean_file,config.train_dataBert_file, config.eval_dataBert_file, config.test_dataBert_file]
while True:
    if all([os.path.exists(f) for f in filelist]):
        break
    else:
        for path in [config.train_data_file,config.eval_data_file,config.test_data_file]:
                
                
                ##clean for basic
                data = pd.read_csv(path, sep='\t')
                data = data.fillna("")
                outputpath=path[:-4]+"_clean"+path[-4:]    
                bs=basic_clean(data)
                bs= bs[['text', 'label']]
                bs.columns=["text_bs","label_bs"]
                
                ##clean for bert
                data = pd.read_csv(path, sep='\t')
                data = data.fillna("")
                outputpath2=path[:-4]+"_Bert"+path[-4:]    
                bert=bert_clean(data)  
                bert = bert[['text', 'label']]
                bert.columns=["text_bert","label_bert"]
                
                a=pd.concat([bs,bert],axis=1)
                mask = a.eq('').any(axis=1)
                print("mask",mask)
                
                a=a[~mask]

                bs=a[["text_bs","label_bs"]]
                bs.columns=["text","label"]
                print("after basic",bs[bs["text"]==''])
                print("bs",bs)
                bs.to_csv(outputpath,index=False,encoding='utf-8')    
                
                bert=a[["text_bert","label_bert"]]
                bert.columns=["text","label"]
                
                print("after bert",bert[bert["text"]==''])
                print("bert",bert)
                bert.to_csv(outputpath2,index=False,encoding='utf-8')  