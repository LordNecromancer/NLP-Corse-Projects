import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math
import pickle
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

import numpy as np


def clean_data(d):
    res=''
    symbols = '{}()[]""\'\'``\\.,:;+-*/&|!...<>=~\'s$'
    stopword = set(stopwords.words('english'))
    tokenized = word_tokenize(d.lower())
    temp = [w for w in tokenized if not w in stopword and
             not w in symbols and not w.isdigit() and len(w) > 1]
    if len(temp) > 0:
        res=' '.join(temp)
    return res

def read_and_clean_docs(args,is_clean=0):
    docs = {}
    rel={}
    cur = 501
    args=reversed(args)
    num=0

    for d in args:
        f=open('lisa/'+d)
        r=f.readline()

        while r:
            r=r.replace('  ',' ')
            if len(r.split(' '))==2 and r.split(' ')[0]=='Document':

                docs[r.strip().split(' ')[1]]=''
                cur=r.strip().split(' ')[1]
                print(cur)
            elif len(r.split(' '))==2 and r.split(' ')[0]=='Query':
                cur=r.strip().split(' ')[1]
                r=f.readline()
                r=f.readline().strip()
                r=r.lower()
                v=r.split(' ')
                while v[-1]!='-1':
                    r=f.readline()
                    r=r.lower()
                    v.extend(r.strip().split(' '))
                docs[cur]=v


            elif len(r.strip().split(' '))==1 and r.strip().isdigit():
                docs[r.strip().split(' ')[0]] = ''
                cur = r.strip().split(' ')[0]


            else:
                if is_clean==0:
                    r=clean_data(r)
                docs[cur]+=r+' '
            r=f.readline()

    return docs


def calculate_tf(docs):
    tf={}
    for k,v in docs.items():
        temp={}
        for d in v.split(' '):
            if d in temp:
                temp[d]+=1
            else:
                temp[d]=1
        tf[k]=temp
    return tf

def calculate_idf(docs,words):
    idf={}
    c=1
    for word in words:
        c+=1
        print(c)
        temp=0
        for n,doc in docs.items():
            if word in doc.split(' '):
                temp+=1
        temp=math.log2(len(docs)/float(temp))
        idf[word]=temp
   # print('sdfdfadfdfds',len(idf))
    return idf


def search_queries(que,tf,idf):
    res={}
    for k,v in que.items():
        t=calculate_tf({k:v})[k]
       # print('tttttttttttttt',t)

        sum=0
        wij2=0
        wik2=0
        res[k]={}
        for k2, v2 in tf.items():
            for k1,v1 in t.items():
                #if k1 in v2 and k1 in idf:
                    #print(k,k2,k1,idf[k1],v2[k1],v1)
               # elif k=='35' and k2=='738':
                   # print('ooooooo',k,k2,k1,v1,k1 in v2,k1 in idf)
                    #print(k1)
                   # print(v2)


                if k1 not in v2 or k1 not in idf:
                    wij=0
                else:
                    wij=v2[k1]*idf[k1]
                if k1 not in idf:
                    wik=0
                else:
                    wik=v1*idf[k1]
                sum+=wij*wik
                wij2+=wij**2
                wik2+=wik**2
            wij2=math.sqrt(wij2)
            wik2=math.sqrt(wik2)
            if (wij2*wik2)>0:
                sum=sum/(wij2*wik2)
            else:
                print(sum)
            res[k][k2]=sum
        #print(t)
    return res


def calculate_precision(acc,res,rel):
    correct = 0
    all = 0
    accs=[]

    for a in acc:
        for k,v in res.items():
            v=dict(list({k: v for k, v in reversed(sorted(v.items(), key=lambda item: item[1]))}.items())[:a])
            #print(k,v)
            for k1,v1 in v.items():
              #  if k1 not in index:
                    #print('1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111')
                if k in rel  and k1 in rel[k]:
                    correct+=1
                all+=1
        print(correct,all)
        accs.append({a:correct/all})
    return accs

def calculate_recall(acc,res,rel):
    correct = 0
    all = 0
    accs=[]
    for a in acc:
        for k,v in res.items():
            v=dict(list({k: v for k, v in reversed(sorted(v.items(), key=lambda item: item[1]))}.items())[:a])
            print(k,v)
            if k in rel:
                all+=len(rel[k])-1
            for k1,v1 in v.items():
                if k in rel  and k1 in rel[k]:
                    #print(k,k1)
                    correct+=1
        accs.append({a:correct/all})

    return accs

def get_index(docs):
    ind = {}
    num = 0
    for q, u in docs.items():
        ind[q] = num
        num += 1
    return ind


def create_tfidf_matrix(docs,tf,idf):
    print(len(tf),len(idf))
    m=[[0]*len(docs) for x in range(len(idf))]
    ind=get_index(docs)
    lendocs=0
    for d,doc in docs.items():
        print(d)
        c = 0
        for k,v in list(idf.items()):
            ktf=0
            if d in tf and k in tf[d]:
                ktf=tf[d][k]
            #print(c,int(d),len(m),len(m[0]))
            m[c][ind[d]]=ktf*v
            c+=1

    return m,ind




#general use after the first run
#args2=['lisa_cleaned.txt']
#docs=read_and_clean_docs(args2,1)





#task 2


args=['lisa0.001','lisa1.001','lisa2.001','lisa3.001','lisa4.001','lisa5.001','lisa0.501','lisa1.501','lisa2.501','lisa3.501','lisa4.501','lisa5.501','lisa5.627','lisa5.850']
docs=read_and_clean_docs(args)
print(docs)

#task 1
f=open('1.txt','a')
f.write(docs['100']+'\n'+'\n')
f.write(docs['101'])


unique={}
for k,v in list(docs.items()):
    for d in v.split(' '):
        if d in unique:
            unique[d]+=1
        else:
            unique[d]=1
unique={k: v for k, v in sorted(unique.items(), key=lambda item: item[1])}
print('unique',len(unique))


tf=calculate_tf(docs)
#run once since it is heavy
'''''
idf=calculate_idf(docs,unique)
f=open('idf_saved.pkl','wb')
pickle.dump(idf,f)
'''''

f=open('idf_saved.pkl','rb')
idf=pickle.load(f)

#task 3
que=read_and_clean_docs(['lisa.que'])
rel=read_and_clean_docs(['lisa.rel'])
unique2={}
for k,v in list(que.items()):
    for d in v.split(' '):
        if d in unique2:
            unique2[d]+=1
        else:
            unique2[d]=1
unique={k: v for k, v in sorted(unique.items(), key=lambda item: item[1])}
print('unique',len(unique))

print('unique2',len(unique2))

results=search_queries(que,tf,idf)

prec=calculate_precision([5,10,20,40],results,rel)
print('precision: '+ str(prec))

recall=calculate_recall([5,10,20,40],results,rel)
print('recall: '+str(recall))

for i in range(4):
    p=list(prec[i].values())[0]
    r=list(recall[i].values())[0]
    f=2*p*r/(p+r)

    print('F-measure '+str(i)+' : '+ str(f))

#task 4
#run once since it's really heavy
'''''
matrix=create_tfidf_matrix(docs,tf,idf)

f=open('tf_idf_matrix.pkl','wb')
pickle.dump(matrix,f)
'''''

f=open('tf_idf_matrix.pkl','rb')
matrix=pickle.load(f)
#print(len(matrix),len(matrix[0]))
matrix=np.transpose(matrix)
svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42)
s=svd.fit_transform(matrix)
#print(len(matrix),len(matrix[0]))
#print(len(s),len(s[0]))
q=que
print(q)
index=get_index(docs)
count=0
colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
for k,v in q.items():
    if count>=10:
        break

    vector=[0]*len(idf)
    tfk=calculate_tf({k:v})[k]
    c = 0
    for k1 ,v1 in list(idf.items()):
        ktf = 0
        if k1 in tfk :
            ktf = tfk[k1]
        # print(c,int(d),len(m),len(m[0]))
        vector[c] = ktf * v1
        c += 1
    vector=np.transpose(vector)
    r=results[k]
    r = dict(list({k2: v2 for k2, v2 in reversed(sorted(r.items(), key=lambda item: item[1]))}.items())[:10])
    print(r)
    vector=vector.reshape(1,-1)
    print(vector)
    vector=svd.transform(vector)
    print(vector)

    x=[vector[0][0]]
    y=[vector[0][1]]
    plt.scatter(x,y,marker='*',color=colors[count])
    x=[]
    y=[]
    for k2,v2 in r.items():
        temp=s[index[k2]]
        print(len(s),len(s[0]))
        print(temp,len(temp),index[k2])
        x.append(temp[0])
        y.append(temp[1])




    plt.scatter(x,y,color=colors[count])
    count += 1
plt.show()



