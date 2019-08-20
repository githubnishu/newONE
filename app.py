from werkzeug.utils import secure_filename
import urllib.request
from flask import Flask, render_template, redirect, url_for, flash, jsonify
from flask import request
from flask_cors import CORS, cross_origin
from html2text import html2text
import re
import requests
import gensim
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.externals import joblib
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from heapq import nlargest as _nlargest
from functools import wraps
import pyodbc


clf = joblib.load('svmClf.pkl')
ocs = joblib.load('ocsOutlier.pkl')
calibrated_svc = joblib.load('svcProb.pkl')
count_vect = joblib.load('countVectorizer.pkl')
tfidf_transformer = joblib.load('tfidfTransformer.pkl')

names = []
types = []
company_doccount = []
resultlabeled1 = []
resultlabeled1_proba = []
resultlabeled1_new = []
resultlabeled2 = []
resultlabeled2_proba = []
resultlabeled2_new = []
resultlabeled3 = []
resultlabeled3_proba = []
resultlabeled3_new = []


app = Flask(__name__, template_folder='templates')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
loader = True
isLoaded = False





def returncompanyname(value, comp_obj):
    for i in range(len(comp_obj)):
        if(comp_obj[i]['company_id'] == str(value)):
            return(comp_obj[i]['company_text'])


def paragraphs(file, separator=None):
    if not callable(separator):
        def separator(line):return line == '\n'
    paragraph = []
    for line in file:
        if separator(line):
            if paragraph:
                yield ''.join(paragraph)
                paragraph = []
        else:
            paragraph.append(line)
    yield ''.join(paragraph)
    return paragraph


lemmatizer = WordNetLemmatizer()


def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


def get_close_matches_indexes(word, possibilities, n=3, cutoff=0.6):
    if not n > 0:
        raise ValueError("n must be > 0: %r" % (n,))
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("cutoff must be in [0.0, 1.0]: %r" % (cutoff,))
    result = []
    s = SequenceMatcher()
    s.set_seq2(word)
    for idx, x in enumerate(possibilities):
        s.set_seq1(x)
        if s.real_quick_ratio() >= cutoff and \
           s.quick_ratio() >= cutoff and \
           s.ratio() >= cutoff:
            result.append((s.ratio(), idx))

    result = _nlargest(n, result)
    return [x for score, x in result]


@app.route('/')
@cross_origin()
def index():
    return(render_template('index.html', loader=loader))



@app.route('/find-labels', methods=["GET", "POST"])
def find_labels():
    
    servererror = []
    arr_link = []
    tag = []
    obj1 = []
    names.clear()
    types.clear()
    resultlabeled1.clear()
    resultlabeled1_proba.clear()
    resultlabeled2.clear()
    resultlabeled2_proba.clear()
    resultlabeled3.clear()
    resultlabeled3_proba.clear()
    arr_link.clear()
    tag.clear()
    obj1.clear()
    company_doccount.clear()
    db_throw = []
    # if request.method == 'POST':
    comp_obj = request.json.get('company_obj')
    year = request.json.get('year')
    doc_obj = request.json.get('doc_obj')

    conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                     'Server=13.71.31.138;'
                     'Database=Query_LoggerDB;'
                     'UID=satender;'
                     'PWD=Irg$App$@Cloud2019')
        
  
    for i in range(len(comp_obj)):
        for j in range(len(doc_obj)):
            comp = returncompanyname(comp_obj[i]['company_id'], comp_obj).replace(' ', '')
            dt = doc_obj[j]['doc_type']
            
            curr = conn.cursor()     
            # query3 = "SELECT * FROM [Query_LoggerDB].[dbo].[Comp_Result] WHERE [Year]=%s AND [DocType]=%s" % int(year) % str(dt)
            curr.execute("SELECT * FROM [Query_LoggerDB].[dbo].[Comp_Result] WHERE [Year]=? AND [DocType]=? AND [Company]=?", (int(year) , str(dt), str(comp)))
            temp = curr.fetchall()
            
            if(len(temp) == 0):
                db_throw.append({"comp_text": returncompanyname(comp_obj[i]['company_id'], comp_obj),"comp_id": comp_obj[i]['company_id'], "dt": doc_obj[j]['doc_type']})

            for rows in temp:
                predicted_Label = rows[5].replace('[', '').replace(']', '').replace("'", '')
                
                if (predicted_Label == 'Authority to set the number of board seats'):
                    resultlabeled1.append(rows[4] + "split" + "Company Name : " + \
                        rows[1] + "split" + "Document Type: " + \
                        rows[3] + "split" + rows[6])
                    resultlabeled1_proba.append(rows[7])
                    
                    
                elif (predicted_Label == 'Voting standard for director elections'):
                    resultlabeled2.append(rows[4] + "split" + "Company Name : " + \
                        rows[1] + "split" + "Document Type: " + \
                        rows[3] + "split" + rows[6])
                    resultlabeled2_proba.append(rows[7])
                elif (predicted_Label == 'Filling of newly created board seats'):
                    resultlabeled3.append(rows[4] + "split" + "Company Name : " + \
                        rows[1] + "split" + "Document Type: " + \
                        rows[3] + "split" + rows[6])
                    resultlabeled3_proba.append(rows[7])
    
    
    new_comp_obj = []
    new_doc_obj = []
    for i in range(len(comp_obj)):
        for j in range(len(db_throw)):
            if(comp_obj[i]["company_id"] == db_throw[j]["comp_id"]):
                new_comp_obj.append({'company_id':db_throw[j]["comp_id"], 'company_text':db_throw[j]["comp_text"]})
            
    for i in range(len(doc_obj)):
        for j in range(len(db_throw)):
            if(doc_obj[i]["doc_type"] == db_throw[j]["dt"]):
                new_doc_obj.append({'doc_type':db_throw[j]["dt"]})
    
    
    if(len(db_throw)==0):
        return("true")
    
    doc_obj = new_doc_obj
    comp_obj = new_comp_obj
    

    base_url = "http://insights.irganalytics.com/"
    for i in range(len(comp_obj)):
        company_doccount.append(0)
        for j in range(len(doc_obj)):
            if doc_obj[j]['doc_type'] == "Proxy":
                generated_link = base_url + "Proxy/" + year + \
                    "/" + comp_obj[i]['company_id'] + ".htm"
                arr_link.append(generated_link)
            else:
                generated_link = base_url + "BoardProxy/" + \
                    doc_obj[j]['doc_type'] + "/" + year + \
                    "/" + comp_obj[i]['company_id'] + ".htm"
                arr_link.append(generated_link)

            names.append(returncompanyname(
                comp_obj[i]['company_id'], comp_obj).replace(' ', ''))
            types.append(doc_obj[j]['doc_type'])

            try:
                page = requests.get(generated_link)
                page.raise_for_status()
            except requests.exceptions.HTTPError as err:
                servererror.append(err+'NOT FOUND!! COMPANY-%s DOCUMENT-%s \n' % (returncompanyname(
                    comp_obj[i]['company_id'], comp_obj).replace(' ', ''), doc_obj[j]['doc_type']))
    
    
    if(len(servererror) != 0):
        return(jsonify(servererror))
    else:
        for i in range(len(arr_link)):
            page = requests.get(arr_link[i])
            page = page.text
            soup = BeautifulSoup(page)
            para = soup.findAll('p')
            font = soup.findAll('div')
            
            if (len(font) > len(para)):
                para = font
            for p in para:
                obj1.append(
                    ' '.join(gensim.utils.simple_preprocess(p.text)))
                tag.append(p.get('id'))

            txt = html2text(page)
            txt1 = re.sub('(?<![\r\n])(\r?\n|\r)(?![\r\n])', ' ', txt)
            obj = paragraphs(txt1)
            
            for item in obj:
                item1 = ' '.join(gensim.utils.simple_preprocess(item))
                if len(item1) > 150:
                   
                    # item2 = lemmatize_sentence(item1)
                    
                    p_count = count_vect.transform([item1])
                    p_tfidf = tfidf_transformer.transform(p_count)
                    
                    if (ocs.predict(p_tfidf)) == 1:
                        tagindex = get_close_matches_indexes(
                            item1, obj1, n=1)
                        returntaglink = arr_link[i]
                        if (len(tagindex)) > 0:
                            returntag = tag[tagindex[0]]
                            returntaglink = arr_link[i]+'#'+returntag

                        value = returntaglink + "split" + "Company Name : " + \
                            str(names[i]) + "split" + "Document Type: " + \
                            str(types[i]) + "split" + item1
                    
                        predicted_Label = clf.predict(p_tfidf)
                        predicted_Proba = calibrated_svc.predict_proba(p_tfidf)
                        
                        curr = conn.cursor()                           
                        query2 = """INSERT INTO [Query_LoggerDB].[dbo].[Comp_Result]([Company],[Year],[DocType],[Link],[Class_Label],[Result],[Probability]) VALUES (?,?,?,?,?,?, ?);"""


                        if (predicted_Label == 'Authority to set the number of board seats'):
                            resultlabeled1.append(value)
                            resultlabeled1_proba.append(
                                predicted_Proba[0][0])
                            curr.execute(query2, (str(names[i]),int(year), str(types[i]), str(returntaglink), str(predicted_Label), str(item1), predicted_Proba[0][0]))
                            conn.commit()
                        elif (predicted_Label == 'Voting standard for director elections'):
                            resultlabeled2.append(value)
                            resultlabeled2_proba.append(
                                predicted_Proba[0][2])
                            curr.execute(query2, (str(names[i]), int(year), str(types[i]),str(returntaglink), str(predicted_Label), str(item1), predicted_Proba[0][2] ))
                            conn.commit()
                        elif (predicted_Label == 'Filling of newly created board seats'):
                            resultlabeled3.append(value)
                            resultlabeled3_proba.append(
                                predicted_Proba[0][1])
                            curr.execute(query2, (str(names[i]), int(year), str(types[i]),str(returntaglink), str(predicted_Label), str(item1), predicted_Proba[0][1] ))
                            conn.commit()
        return("true")





@app.route('/find-labelsresults', methods=["POST", "GET"])
@cross_origin()
def find_labelsresults():
    resultlabeled1_new.clear()
    resultlabeled2_new.clear()
    resultlabeled3_new.clear()

    c_labels = request.form.getlist('c_labels')
    c_numbers = request.form.getlist('c_numbers')

    var = ['No Result For this Class Label']
    if (c_labels[0] == '1'):
        if (len(resultlabeled1) == 0):
            return(jsonify(var))
        else:
            if c_numbers[0] == 'all':
                return(jsonify(resultlabeled1))
            else:
                j = sorted(range(len(resultlabeled1_proba)), key=lambda i: resultlabeled1_proba[i], reverse=True)[
                    :min(int(c_numbers[0]), len(resultlabeled1_proba))]
                for i in j:
                    resultlabeled1_new.append(resultlabeled1[i])
                return(jsonify(resultlabeled1_new))
    elif (c_labels[0] == '2'):
        if (len(resultlabeled2) == 0):
            return(jsonify(var))
        else:
            if c_numbers[0] == 'all':
                return(jsonify(resultlabeled2))
            else:
                j = sorted(range(len(resultlabeled2_proba)), key=lambda i: resultlabeled2_proba[i], reverse=True)[
                    :min(int(c_numbers[0]), len(resultlabeled2_proba))]
                for i in j:
                    resultlabeled2_new.append(resultlabeled2[i])
                return(jsonify(resultlabeled2_new))
    elif (c_labels[0] == '3'):
        if (len(resultlabeled3) == 0):
            return(jsonify(var))
        else:
            if c_numbers[0] == 'all':
                return(jsonify(resultlabeled3))
            else:
                j = sorted(range(len(resultlabeled3_proba)), key=lambda i: resultlabeled3_proba[i], reverse=True)[
                    :min(int(c_numbers[0]), len(resultlabeled3_proba))]
                for i in j:
                    resultlabeled3_new.append(resultlabeled3[i])
                return(jsonify(resultlabeled3_new))


if __name__ == "__main__":
    app.run(debug=True)
