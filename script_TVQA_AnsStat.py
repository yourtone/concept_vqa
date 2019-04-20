# script: check train answer statistics

import json
from operator import itemgetter

data=json.load(open('dataTVQA/TextVQA_0.5_train.json','r'))
pairs=data['data']
print('pairs length: {}'.format(len(pairs)))
anslist=list(map(itemgetter('answers'), pairs))
print('anslist length: {}'.format(len(anslist)))
alist = eval('[%s]'%repr(anslist).replace('[', '').replace(']', ''))
print('alist length: {}'.format(len(alist)))
dict={}
for ans in alist:
    dict[ans]=dict.get(ans,0)+1

print('dict length: {}'.format(len(dict)))
sdict=sorted(dict.items(),key=itemgetter(1),reverse=True)
print('Top 10 answer statistics:\n\t{}'.format(repr(sdict[:10])))