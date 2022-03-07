'''
Author: Julie Syu
LastEditTime: 2021-06-03
Desciption: Configuration
'''


train_raw_file = './text_data/-+-¦-+¦+-²+¦+»/-+-¦-+¦+-²+¦+»/data_cat10_annotated_train.txt'
eval_raw_file = './text_data/-+-¦-+¦+-²+¦+»/-+-¦-+¦+-²+¦+»/data_cat10_annotated_eval.txt'
test_raw_file = './text_data/-+-¦-+¦+-²+¦+»/-+-¦-+¦+-²+¦+»/data_cat10_annotated_test.txt'
rawdata_index_to_label=False

label_ids_file = './text_data/id_label.txt'
train_data_file = './text_data/train.csv'
eval_data_file = './text_data/eval.csv'
test_data_file = './text_data/test.csv'



train_dataclean_file = './text_data/train_clean.csv'
eval_dataclean_file = './text_data/eval_clean.csv'
test_dataclean_file = './text_data/test_clean.csv'
train_embed_file= './text_data/train_clean.csv'

train_dataBert_file = './text_data/train_Bert.csv'
eval_dataBert_file = './text_data/eval_Bert.csv'
test_dataBert_file = './text_data/test_Bert.csv'

label2id='./data/label2id.json'
id2label='./data/id2label.json'

chstopwords_bert= './data/stopwordsbert.pickle'

chstopwords = './data/stopwords.txt'
chstopwords_obj = './data/stopwords.pickle'

model='./model/clf_BR_word2v_bert.pkl'


user_dict='./user.txt'