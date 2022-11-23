from utils import *
from ngram_process import graph_generated


d = 100  # embedding dimension
tr = 0  # training ratio

root = 'data/sina_douban/'
anchorfile = root + 'sina_douban_anchor.txt'
namefile1 = root + 'sina_douban_all_sina.txt'
namefile2 = root + 'sina_douban_all_douban.txt'
an_length = 3088

edgefile1 = root + 'graph/sina_douban_all_sina_DENA.txt'
edgefile2 = root + 'graph/sina_douban_all_douban_DENA.txt'

embedding_file1 = root + 'emb/sina_douban_all_sina_emb_DENA.txt'
embedding_file2 = root + 'emb/sina_douban_all_douban_emb_DENA.txt'

print('training ratio=', tr, 'dim=', d)
trainfile = root + str(tr) + '_anchor_train.txt'
testfile = root + str(tr) + '_anchor_test.txt'

anchor_split(anchorfile=anchorfile, train_file=trainfile, test_file=testfile, train_ratio=tr)
graph_generated(namefile1, namefile2, edgefile1, edgefile2)
cosine_matrix, test_length = run_emb(edgefile1, edgefile2, trainfile, testfile,
                                     namefile1, namefile2, embedding_file1,embedding_file2,
                                     an_length=an_length,rep_dim=d, epoch=500)

# 30 means outputting pre@1~pre@30
pre = evaluate(cosine_matrix, 30, an_length=test_length)
print('pre@k=', pre)
