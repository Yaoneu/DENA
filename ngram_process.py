# -*- coding=utf-8 -*-
import pypinyin
import pymongo
import sys


def n_gram_DEAN(sn_name):
    # database
    client = pymongo.MongoClient('localhost', 27017)
    db = client['chinese_character']
    characters = db['characters']

    n_gram_dict = {}
    stroke_n_gram_dict = {}
    pinyin_n_gram_dict = {}
    word_dict = {}
    for n in sn_name:
        name_n_gram = n_gram_list(n)
        n_gram_dict[n] = name_n_gram
        for word in name_n_gram:
            if len(word) > 1 and word not in word_dict.keys():
                chinese_list = [char for char in word if u'\u4e00' <= char <= u'\u9fff']
                if chinese_list:
                    word_dict[word] = chinese_list
        chinese_char = []
        for char in n:
            if u'\u4e00' <= char <= u'\u9fff':
                chinese_char.append(char)
        if len(chinese_char) > 0:
            for c in chinese_char:
                if c in stroke_n_gram_dict.keys():
                    pass
                else:
                    structure, structure_code, stroke, stroke_code = structure_stroke_query(c, characters)
                    stroke_n_gram = n_gram_list(stroke)
                    stroke_n_gram_dict[c] = stroke_n_gram
                if c in pinyin_n_gram_dict.keys():
                    pass
                else:
                    pinyin = pypinyin.pinyin(c, style=pypinyin.STYLE_TONE, heteronym=True)
                    pinyin_n_gram = []
                    for py in pinyin[0]:
                        pinyin_n_gram = pinyin_n_gram+n_gram_list(py)
                    pinyin_n_gram = list(set(pinyin_n_gram))
                    pinyin_n_gram_dict[c] = pinyin_n_gram
    return n_gram_dict, stroke_n_gram_dict, pinyin_n_gram_dict, word_dict


def structure_stroke_query(char, characters):
    # char=char.encode('utf-8')
    information = characters.find({'_id': char})
    if information.count() > 1:
        print('error')
        sys.exit()
    else:
        info = information[0]
        structure = info['structure']
        structure_code = info['structure_code']
        stroke = info['stroke']
        stroke_code = info['stroke_code']
        return structure, structure_code, stroke, stroke_code


def n_gram_list(ngram_string):
    length = len(ngram_string)
    if length == 1:
        return [ngram_string]
    n_gram = []
    for i in range(2, length+1):
        for j in range(length+1-i):
            n_gram.append(ngram_string[j:j+i])
    return list(set(n_gram))


def ngram_graph_DEAN(n_gram_dict, stroke_n_gram_dict, pinyin_n_gram_dict, word_dict, edgefile, sn='1'):
    edge_list = []
    for key in n_gram_dict.keys():
        ngram_list = n_gram_dict.get(key)
        for n in ngram_list:
            edge_list.append([key+sn, n])

    for key in stroke_n_gram_dict.keys():
        ngram_list = stroke_n_gram_dict.get(key)
        for n in ngram_list:
            edge_list.append([key, n])

    for key in pinyin_n_gram_dict.keys():
        ngram_list = pinyin_n_gram_dict.get(key)
        for n in ngram_list:
            edge_list.append([key, n])

    for key in word_dict:
        ngram_list = word_dict.get(key)
        for n in ngram_list:
            edge_list.append([key, n])

    f = open(edgefile, 'w', encoding='utf-8')
    for edge in edge_list:
        f.write("{} ,{}\n".format(edge[0], edge[1]))
    f.close()


def read_name(name_file):
    name = []
    f = open(name_file, 'r', encoding='utf-8')
    while 1:
        l = f.readline().rstrip('\n')
        if l == '':
            break
        name.append(l)
    f.close()
    return name


def graph_generated(namefile1, namefile2, edgefile1, edgefile2):
    namelist1 = read_name(namefile1)
    namelist1 = list(set(namelist1))
    n_gram_dict, stroke_n_gram_dict, pinyin_n_gram_dict, word_dict = n_gram_DEAN(namelist1)
    ngram_graph_DEAN(n_gram_dict, stroke_n_gram_dict, pinyin_n_gram_dict, word_dict, edgefile1, sn='1')

    namelist2 = read_name(namefile2)
    namelist2 = list(set(namelist2))
    n_gram_dict, stroke_n_gram_dict, pinyin_n_gram_dict, word_dict = n_gram_DEAN(namelist2)
    ngram_graph_DEAN(n_gram_dict, stroke_n_gram_dict, pinyin_n_gram_dict, word_dict, edgefile2, sn='2')
    print('display name graph generated')
