
import numpy as np


def preprocess(f,alltags):
    sents=[[]]
    tags=[[]]
    words={}

    l=f.readline()
    c=0
    while l:
        #print(sents[-1])
        #print(tags[-1])
        c+=1
        if len(l)<2:
            l = f.readline()
            continue
        w,t=l.split()


        if w not in words:
            words[w]=1
        if t not in alltags:
            alltags.append(t)
        if w in '!?:;.':

           #print(sents[-1],tags[-1])
            sents[-1].append(w)
            tags[-1].append(t)

            sents.append(['<S>'])
            tags.append(['start'])

        else:
            sents[-1].append(w)
            tags[-1].append(t)
        l=f.readline()
    return words,sents,tags

def preprocess_test(f,alltags):
    sents = [[]]
    words = {}

    l = f.readline()
    c = 0
    while l:
        l=l.replace('\n','')

      #  print(l,len(l))
        # print(sents[-1])
        # print(tags[-1])

        print(c)

        w= l




        if w in '!?:;.':
            print("sdgsdgsdgsdgsd")

            # print(sents[-1],tags[-1])
            sents[-1].append(w)
            sents.append(['<S>'])
            c += 1

        else:
            if w not in words:
                words[w] = 1
            sents[-1].append(w)
            c += 1
        l = f.readline()
    return words, sents

def get_bigram_count(sents):
    bicount={}
    for sent in sents:
        for i in range(len(sent)-1):
            if (sent[i],sent[i+1]) in bicount:
                bicount[(sent[i],sent[i+1])]+=1
            else:
                bicount[(sent[i], sent[i + 1])]=1

    return bicount

def get_unigram_count(sents):
    unicount={}
    for  sent in sents:
        for i in sent:
            if i in unicount:
                unicount[i]+=1
            else:
                unicount[i]=1
    return unicount



def calculate_transition_count(tags):
    transition_probs={}
    bicount=get_bigram_count(tags)
    unicount=get_unigram_count(tags)
    #print(unicount,bicount)
    return unicount,bicount
def calculate_emission_counts(sents,tags):
    emission_count={}
    for i in range(len(sents)):
        for j in range(len(sents[i])):
            if (sents[i][j],tags[i][j]) in emission_count:
                emission_count[(sents[i][j], tags[i][j])]+=1
            else:
                emission_count[(sents[i][j], tags[i][j])] = 1
    return emission_count

def run_viterbi(data,words,emission,unicount,bicount,alltags,k):
    m = [[0] * len(data) for x in range(len(alltags))]
    b = [[0] * len(data) for x in range(len(alltags))]
   # print(data)
#    data.pop(-1)

    for i in range(len(alltags)):
        if ('start',alltags[i]) in bicount:
            p_transit=bicount[('start',alltags[i])]/unicount['start']
        else:
            p_transit=0
            #print('aaaaaaaaaaaaaaaa',data)

        if len(data)>=1 and (data[0],alltags[i]) in emission:
            p_emission=emission[(data[0],alltags[i])]/unicount[alltags[i]]
        else:
            p_emission=0

        #print('kkkkkkkkkk',p_transit,('start',alltags[i]) in bicount,p_emission)

        if p_transit==0:
            p_transit=k
        else:
            p_transit=(bicount[('start',alltags[i])]+k)/(unicount['start']+k*words)

        if p_emission==0:
            p_emission=k
        else:
            p_emission=(emission[(data[0],alltags[i])]+k)/(unicount[alltags[i]]+k*words)
        temp=p_transit*p_emission
        m[i][0]=temp
        b[i][0]=-1
       # print('jjjjjjjjjjj',p_transit,('start',alltags[i]) in bicount,p_emission)



    for i in range(1,len(data)):
        for j in range(len(alltags)):
            #print(len(b),len(b[0]),i,j,len(alltags))
            maximum=0
            for u in range(len(alltags)):

                if (alltags[u], alltags[j]) in bicount:
                    p_transit = bicount[(alltags[u], alltags[j])] / unicount[alltags[u]]
                else:
                    p_transit=0
                if (data[i], alltags[j]) in emission:
                    p_emission = emission[(data[i], alltags[j])] / unicount[alltags[j]]
                else:
                    p_emission=0
                #print('mmmmmm',p_transit,p_emission,m[j][i-1])

                if p_transit == 0:
                    p_transit = k
                else:
                    p_transit = (bicount[(alltags[u], alltags[j])] + k) / (unicount[alltags[u]] + k * words)

                if p_emission == 0:
                    p_emission = k
                else:
                    p_emission = (emission[(data[i], alltags[j])] + k) / (unicount[alltags[j]] + k * words)
                temp = p_transit * p_emission
                if m[u][i-1]*temp>maximum:
                    #print(j,i,u,p_transit,p_emission,m[j][i-1])
                    maximum=m[u][i-1]*temp
                    b[j][i]=u
            m[j][i]=maximum
    #print(m)
    #print(b)
    maxim=0
    for j in range(len(alltags)):
        if m[j][len(data)-1]>m[maxim][len(data)-1]:
            maxim=j
    tags=[alltags[maxim]]
    prev=b[maxim][len(data)-1]
    i=len(data)-1
    while prev!=-1 :
        tags.append(alltags[prev])
        i=i-1
        prev=b[prev][i]
    return list(reversed(tags))






def validate(data,data_tag,words,emission,unicount,bicount,alltags,k,isTest=0):

    if isTest==1:
        file=open('y_test.txt','a')
        c=0

        for i,d in enumerate(data):
            best = run_viterbi(d, words, emission, unicount, bicount, alltags, k)
            #print(len(data),len(d),d)
            #print(best)
            for j in range(len(d)):



                if isTest==1 and best[j]=='start':
                    continue
                c += 1
                print(c)
                file.write(best[j]+'\n')



    else:


        correct=0
        all=0
        for i,d in enumerate(data):
            best=run_viterbi(d,words,emission,unicount,bicount,alltags,k)
            #print(d)
            #print(best)
            for j in range(len(d)):
                #print(best[j] , data_tag[i][j],best[-j])
                if best[j]==data_tag[i][j]:
                    correct+=1
                all+=1
        return correct/all











alltags=['start']
f=open('Train.txt','r',encoding="utf8")
train_words,train,train_tags=preprocess(f,alltags)

f=open('Val.txt','r',encoding="utf8")
val_words,val,val_tag=preprocess(f,alltags)



f=open('Test.txt','r',encoding="utf8")
test_words,test=preprocess_test(f,alltags)
not_in_train=[]
for w in val_words:
    if w not in train_words:
        not_in_train.append(w)
print('number of words not in train but available in val and test: ',len(not_in_train))


#alltags.insert(0,'start')

unicount,bicount=calculate_transition_count(train_tags)
emission_count=calculate_emission_counts(train,train_tags)
validate(test,{},len(train_words)+len(not_in_train),emission_count,unicount,bicount,alltags,10**-8,1)

print(1)
#print(validate(val,val_tag,len(train_words)+len(not_in_train),emission_count,unicount,bicount,alltags,10**-8))
'''''
print(2)
print(validate(val,val_tag,len(train_words)+len(not_in_train),emission_count,unicount,bicount,alltags,10**-5))
print(3)
print(validate(val,val_tag,len(train_words)+len(not_in_train),emission_count,unicount,bicount,alltags,10**-4))
print(4)
print(validate(val,val_tag,len(train_words)+len(not_in_train),emission_count,unicount,bicount,alltags,10**-3))
print(5)
print(validate(val,val_tag,len(train_words)+len(not_in_train),emission_count,unicount,bicount,alltags,10**-2))
print(6)
print(validate(val,val_tag,len(train_words)+len(not_in_train),emission_count,unicount,bicount,alltags,10**-1))
print(7)
print(validate(val,val_tag,len(train_words)+len(not_in_train),emission_count,unicount,bicount,alltags,10**-0))
'''




