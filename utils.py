import graph
import numpy as np
import ngram_process as ngp
import time
import embedding
import random
from sklearn.metrics.pairwise import cosine_similarity


def run_emb(inputfile1, inputfile2, train_file, test_file, namefile1, namefile2,
            embedding_file1, embedding_file2, an_length, rep_dim=100, epoch=300):
    t1 = time.time()
    edges=graph_for_learning(inputfile1,inputfile2,train_file)
    g=graph.Graph()
    g.read_edgelist(edges)
    print('node number:',g.G.number_of_nodes(),'edge number:',g.G.number_of_edges())
    del edges

    test_sn1 = []
    test_sn2 = []
    f = open(test_file, 'r', encoding='utf-8')
    while 1:
        l = f.readline().rstrip('\n')
        if l == '':
            break
        name1, name2 = l.split(' ,')
        test_sn1.append(name1)
        test_sn2.append(name2)
    f.close()
    test_length=len(test_sn1)
    f1 = open(namefile1, 'r', encoding='utf-8')
    i = 0
    while 1:
        l = f1.readline().rstrip('\n')
        if l == '':
            break
        if i >= an_length:
            test_sn1.append(l+'1')
        i = i+1
    f1.close()
    f2 = open(namefile2, 'r', encoding='utf-8')
    i=0
    while 1:
        l = f2.readline().rstrip('\n')
        if l == '':
            break
        if i >= an_length:
            test_sn2.append(l+'2')
        i = i+1
    f2.close()

    model = embedding.Embedding(g, namefile1, namefile2, rep_dim=rep_dim, epoch=epoch)
    print("Embedding Running time is", time.time() - t1)
    emb_vec = model.emb_vec

    namelist1 = ngp.read_name(namefile1)
    namelist2 = ngp.read_name(namefile2)
    nameemb1 = {}
    for name in namelist1:
        vec = emb_vec[name+'1']
        nameemb1[name] = vec

    model.save_embeddings(embedding_file1, nameemb1)

    nameemb2 = {}
    for name in namelist2:
        vec = emb_vec[name+'2']
        nameemb2[name] = vec

    model.save_embeddings(embedding_file2, nameemb2)

    sn1_emb=[]
    sn2_emb=[]
    for name in test_sn1:
        if name in emb_vec.keys():
            vec = emb_vec.get(name)
            sn1_emb.append(vec)
        else:
            print(name)

    for name in test_sn2:
        if name in emb_vec.keys():
            vec = emb_vec.get(name)
            sn2_emb.append(vec)
        else:
            print(name)
    cosine_matrix = cosine_similarity(np.array(sn1_emb), np.array(sn2_emb))
    return cosine_matrix, test_length


def graph_for_learning(inputfile1, inputfile2, trainfile):
    edges = []
    f1 = open(inputfile1, 'r', encoding='utf-8')
    while 1:
        l = f1.readline().rstrip('\n')
        if l == '':
            break
        src, dst = l.split(' ,')
        edges.append([src, dst])
    f1.close()
    f2 = open(inputfile2, 'r', encoding='utf-8')
    while 1:
        l = f2.readline().rstrip('\n')
        if l == '':
            break
        src, dst = l.split(' ,')
        edges.append([src, dst])
    f2.close()
    if trainfile != '':
        ft = open(trainfile, 'r', encoding='utf-8')
        while 1:
            l = ft.readline().rstrip('\n')
            if l =='':
                break
            src, dst = l.split(' ,')
            edges.append([src, dst])
        ft.close()

    edges=list(set([tuple(t) for t in edges]))
    print('Edge length:', len(edges))
    return edges


def evaluate(matrix, k, an_length):
    pre = [0] * k
    t = 0.5

    for i in range(an_length):
        sort1 = np.argsort(-matrix[i])
        for j in range(len(sort1)):
            if sort1[j] == i and j < k:
                for p in range(j, k):
                    pre[p] = pre[p]+1
        sort2 = np.argsort(-matrix.T[i])
        for j in range(len(sort2)):
            if sort2[j] == i and j<k:
                for p in range(j, k):
                    pre[p] = pre[p]+1
    pre_n = np.array(pre)/(2 * an_length)
    return pre_n


def anchor_split(anchorfile, train_file, test_file, train_ratio):
    anchor = []
    fa=open(anchorfile, 'r', encoding='utf-8')
    while 1:
        l = fa.readline().rstrip('\n')
        if l == '':
            break
        name1, name2 = l.split(' ,')
        anchor.append([name1, name2])
    fa.close()
    train_list = random.sample(anchor, int(train_ratio*len(anchor)))
    test_list = []
    for an in anchor:
        if an not in train_list:
            test_list.append(an)
    ftr = open(train_file, 'w', encoding='utf-8')
    for train in train_list:
        ftr.write("{}\n".format(' ,'.join([x for x in train])))
    ftr.close()
    fte = open(test_file, 'w', encoding='utf-8')
    for test in test_list:
        fte.write("{}\n".format(' ,'.join([x for x in test])))
    fte.close()
    print('anchors have been split')
