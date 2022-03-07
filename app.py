'''
Author: Julie.Syu
LastEditTime: 2021-06-03
Description: Application
'''
from flask import Flask, request, jsonify,render_template
import json

from model import *
import googletrans


#######################################################################
#Initialize translator and load classifier model#
#######################################################################

bc=Classifier()
clf=bc.load()
translator = googletrans.Translator()

#######################################################################
#Initialize flask     #
#######################################################################

app = Flask(__name__)




#set up route
@app.route('/', methods=['GET',"POST"])
def gen_ans():
    '''
    @description: RestfulAPI to get the input symptoms and return the predicted label 
    @param {type}
    @return: json displayed on the html
    '''
    print(3)
    #######################################################################
    #          predict the input text and return response
    #######################################################################
    #
    #
    print(1, request)
    print(2, [i for i in request.form.values()])
    if request.method=='GET':
        #return('<form action="/predict" method="post"><h1>Please tell us your symptons:<br /></h1><textarea name="text" #rows="20" cols="50">Enter your request here.</textarea><br /><input type="submit" value="Send" /></form>')
        return render_template('template1.html')
    elif request.method=='POST':
        print(3, [i for i in request.form.values()])
        print(request.form["text"])
        text_orig = request.form["text"]
        lang_orig=translator.detect(text_orig).lang
        text=translator.translate(text_orig, dest='zh-CN').text
		
        label=bc.predict(text,clf,emb_list=emb_list,basic_list=basic_list)
        result = {}
        print(lang_orig,text,label)
        if lang_orig!='zh-CN':
                label_lang=[translator.translate(i, dest=lang_orig).text for i in label[0]]
                result = {"label":[label_lang],"labelinchinese": label,}

			
        else:
                result = {"label": label}

 
        return render_template('template2.html',name='test',result=result)#json.dumps(result, ensure_ascii=False, indent=4))


#python -m flask run
if __name__ == '__main__':
    print(4)
    app.run()#host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    # http://localhost:5000/
