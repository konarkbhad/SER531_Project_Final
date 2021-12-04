#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask.templating import render_template
import nltk
from flask import Flask

app = Flask(__name__)


# In[2]:



#nltk.download('punkt')


# load english language model


# In[3]:


# import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


# In[4]:


#nltk.download("stopwords")
from nltk.corpus import stopwords


# In[5]:


text = '''Harry James Potter was a half-blood wizard, one of the most famous wizards of modern times. 
He was born on 31 July, 1980. He was the only child and son of James and Lily Potter, both  members of the original Order of the Phoenix. '''


# In[6]:


sent_tokenize(text)


# In[7]:


sentance = sent_tokenize(text)


# In[8]:


words = word_tokenize(text)
words


# In[9]:


stop_words = set(stopwords.words("english"))
stop_words


# In[10]:


filtered_word =text.split(' ')
filtered_word


# In[11]:


from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')


# In[12]:


lematizer = WordNetLemmatizer()
lem_word = [lematizer.lemmatize(word) for word in words]


# In[13]:


small = ' '.join(lem_word)


# In[14]:


#get_ipython().system('pip install spacy')


# In[15]:


import spacy


# In[16]:


#get_ipython().system('spacy download en_core_web_sm')


# In[17]:


nlp = spacy.load('en_core_web_sm')


# In[18]:


details = nlp(small)


# In[19]:


for token in details:
    print (token, token.tag_, token.pos_)


# In[20]:


from spacy import displacy


# In[21]:


displacy.render(details, style='dep', jupyter=True)


# In[22]:



piano_doc = nlp(text)
for ent in piano_doc.ents:
     print(ent.text,
           ent.label_, spacy.explain(ent.label_))


# In[23]:


import nltk
from nltk import pos_tag
from nltk import RegexpParser
#nltk.download('averaged_perceptron_tagger')


# In[24]:


text2 = 'In Euro 1992, Germany reached the final, but lost 0–2 to Denmark'
import re
def preproc(text):
    text = re.sub(r'[^\w\s]', '', text)
    #tokens = nltk.word_tokenize(text)
    #token = [word for word in tokens if word.casefold() not in stop_words]
    #' '.join(token)
    return (text)


# In[25]:


text1 = preproc(text)
text1


# In[26]:


stop_words = set(stopwords.words("english"))
tokens = nltk.word_tokenize(text)
tags = nltk.pos_tag(tokens)
grammar = "NP: {<DT>?<JJ>*<NN>}"
cp = nltk.RegexpParser(grammar)
res = cp.parse(tags)
print(res)


# In[27]:


#res.draw()


# In[ ]:





# In[28]:


#print(res)


# In[29]:


my_doc=nlp(text)

for token in my_doc:
    print(token.text,'---',token.dep_)


# In[30]:


displacy.render(my_doc ,style='dep', jupyter=True)


# In[31]:


tokens = nltk.word_tokenize(text)
sent = nltk.pos_tag(tokens)


# In[32]:


sent


# In[33]:


from nltk.chunk import RegexpParser
chunker = RegexpParser(r'''
NP:
{<DT><.*>*<NN.*>}
<NN.*>}{<.>
<.*>}{<DT>
<NN.*>{}<NN.*>
''')
chunk = chunker.parse(sent)


# In[34]:


hig = chunk.height()


# In[35]:


tree = []
for s in chunk.subtrees(lambda t: t.height()<hig+1):
    tree.append(s.leaves())


# In[36]:


top = tree[0]
top1 = top.copy()
all = tree[1:]


# In[37]:


all1 = [food for sublist in all for food in sublist]
for i in all1:
    top1.remove(i)


# In[38]:


top1


# In[39]:


chunk_1 = ' '.join([i[0] for i in top1 if i[0].casefold() not in stop_words])
chunk_1


# In[40]:


all_chunks = []
for j in all:
    all_chunks.append(' '.join([i[0] for i in j if i[0].casefold() not in stop_words]))


# In[41]:


all_chunks


# In[42]:


final = [chunk_1]
final.append(all_chunks)
final


# In[43]:



# Loading Libraries
from nltk.chunk.regexp import ChunkString, ChunkRule,SplitRule
from nltk.tree import Tree
from nltk.chunk.regexp import MergeRule, SplitRule
  
# Chunk String
chunk_string = ChunkString(Tree('S', sent))
print ("Chunk String : ", chunk_string)
  
# Applying Chunk Rule
ur = ChunkRule('<DT><.*>*<NN.*>', 'chunk determiner to noun')
ur.apply(chunk_string)
print ("\nApplied ChunkRule : ", chunk_string)

  
# Splitting
sr1 = SplitRule('<NN.*>', '<.>', 'split after noun')
sr1.apply(chunk_string)
print ("\nSplitting Chunk String : ", chunk_string)
  
sr2 = SplitRule('<.*>', '<DT>', 'split before determiner')
sr2.apply(chunk_string)
print ("\nFurther Splitting Chunk String : ", chunk_string)
  
# Merging
mr = MergeRule('<NN.*>', '<NN.*>', 'merge nouns')
mr.apply(chunk_string)
print ("\nMerging Chunk String : ", chunk_string)
  
# Back to Tree
#chunk_string.to_chunkstruct(Tree)


# ## Final Function.. Need to pass the text 

# In[ ]:





# In[ ]:





# In[44]:


# import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk import pos_tag
from nltk import RegexpParser
from nltk.tokenize import sent_tokenize, word_tokenize


# In[45]:


#nltk.download('punkt')


# In[46]:


#nltk.download('averaged_perceptron_tagger')


# In[47]:


from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')


# In[48]:


#nltk.download("stopwords")
from nltk.corpus import stopwords


# In[49]:


stop_words = set(stopwords.words("english"))
filtered_word = [word for word in words if word.casefold() not in stop_words]


# In[50]:


text = '''Harry James Potter was a half-blood wizard, one of the most famous wizards of modern times. 
He was born on 31 July, 1980. He was the only child and son of James and Lily Potter, both  members of the original Order of the Phoenix. '''


# In[51]:


text2 = 'In Euro 1992, Germany reached the final, but lost 0–2 to Denmark'


# In[52]:


import re
def preproc(text):
    text = re.sub(r'[^\w\s]', '', text)
    #tokens = nltk.word_tokenize(text)
    #token = [word for word in tokens if word.casefold() not in stop_words]
    #' '.join(token)
    return (text)


# In[53]:


text1 = preproc(text)
text1


# In[54]:


text3 = preproc(text2)
text3


# In[55]:


from nltk.chunk import RegexpParser
chunker = RegexpParser(r'''
        NP:
        {<DT><.*>*<NN.*>}
        <NN.*>}{<.>
        <.*>}{<DT>
        <NN.*>{}<NN.*>
        ''')
def extraction(text):
    final_1 = []
    sentance = nltk.sent_tokenize(text)
    for sentance in sentance:
        tree = []
        tokens = nltk.word_tokenize(sentance)
        sent = nltk.pos_tag(tokens)
        
        
        chunk = chunker.parse(sent)
        hig = chunk.height()
        for s in chunk.subtrees(lambda t: t.height()<hig+1):
            tree.append(s.leaves())
        top = tree[0]
        top1 = top.copy()
        all = tree[1:]
        all1 = [food for sublist in all for food in sublist]
        for i in all1:
            top1.remove(i)
        chunk_1 = ' '.join([i[0] for i in top1 if i[0].casefold() not in stop_words])
        final_1.append(chunk_1)
        for j in all:
            final_1.append(' '.join([i[0] for i in j if i[0].casefold() not in stop_words]))
    return final_1


# In[56]:
print('.....................')
output_1 = extraction(text1)
print(extraction(text1))

print('.....................')
output_2 = extraction(text3)


# In[57]:


print(extraction(text3))


# In[ ]:

from flask import jsonify, request

@app.route("/text1")
def test():
    return jsonify(output_1)

@app.route("/text2")
def test2():
    return jsonify(output_2)    

# @app.route("/home")
# def home():
#     return render_template('home.html')

def do_something(text1):
    out = extraction(text1)
    return jsonify(out)
   
@app.route('/test')
def home():
    return render_template('home.html')

@app.route('/join', methods=['GET','POST'])
def my_form_post():
    text1 = request.form['text1']
    word = request.args.get('text1')
    out = do_something(text1)
    result = {
        "output": out
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

if __name__ == "__main__":
    app.run()
