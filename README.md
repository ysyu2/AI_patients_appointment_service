# AI_patients_appointment_service
Feeling sick?? No idea to book appointment in which hospital departments??  As a foreigner living in a mandarin country might encounter language or culture barrier when booking medical appointment.  Input your symptoms in your native language, boom! This intelligent service can help you which departments to go.  


[Demo Video](https://user-images.githubusercontent.com/50165431/157267118-f146ea00-581b-41f7-8ecd-b27c8089be2e.mp4)



## Project Objective
The aim of this personal side-project is to develop a general multilabel text classification framework and can be implemented in any corpus. 
The framework includes preprocessing, embedding and feature engineering, model training(with resampling and parameter tunning) and deployment on web.

The feature engineering technque is leveraging by this paper [Joint Embedding of Words and Labels for Text Classification](https://arxiv.org/abs/1805.04174) introducing an attention framework that measures the compatibility of embeddings between textsequences and labeles. Further, I implmented Average/Max pooling with different windows size  to generate lists of embedding array. This project can be seen as an extension to features embedding techniques.


## The Framework

### data preparation and semi-auto labeling
Data labelling is exhausted and costly. The data labeling task is leveraged by the state-of-the-art algorithms,[confident learning](https://arxiv.org/abs/1911.00068), to find label errors and the [cleanlab](https://github.com/cleanlab/cleanlab) is implmented in this project. Therefore, the initiate manully labeling set is 500 and iterativly expands to 5000+ labeling set. 
![image](https://user-images.githubusercontent.com/50165431/157873751-a8271e80-3961-41f2-af6e-0bf23151e51c.png)

### Multi-labeling classification 
![image](https://user-images.githubusercontent.com/50165431/157068252-02ed7d60-4062-4449-ae1e-56b291ce82b3.png)

![image](https://user-images.githubusercontent.com/50165431/157069551-a1d64194-0874-4be8-9215-b182ef9065db.png)

## Performance accross different feature engineering

| F1-score      |fasttext with "vs_mean_label_max" + tfidf|fasttext with "vs_mean_label_max"|fasttext with "mean_all"|fasttext with "max_all"|
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Test set   | 0.91          | 0.89          | 0.83          | 0.79          |
| Train set      | 0.95          | 0.91          | 0.89          | 0.85          |


ps: "vs_mean_label_max" is the feature concatenate with ['w2v_label_max' ,'w2v_mean' ,'w2v_2_mean','w2v_3_mean' ,'w2v_4_mean']

## Core Package Components
1.__config.py__ - _the configuration of file and model location ex: trainig/eval/test/stopwords dataset._ <br />
2.__data.py__ - _clean dataset, remove stopwords and save tokenized string with string label._<br />
3.__embedding.py__ - _train and save embedding & tfidf & LDA._<br />
4.__features.py__ - _feature engineering techniques and label-word join embedding._<br />
5.__model.py__ - _building model with sampling and parameter tuning function._<br />
6.__app.py__ - _deploy on flask and runing on web application._<br />

