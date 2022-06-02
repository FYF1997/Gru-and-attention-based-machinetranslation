from __future__ import  unicode_literals, print_function, division
import math
import re
import time
import unicodedata

import jieba
import torch
from machine_translation.logger import logger
from  torch.autograd import  Variable

use_cuda = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 25
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category((c)) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r"\1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

class Lang:
    def __init__(self, name):
        self.name = name
        self.need_cut = self.name == 'cmn'
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 2

    def addSentences(self,sentence):
        if self.need_cut:
            sentence = cut(sentence)

        for word in sentence.split(' '):
            if len(word) > 0:
                self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
def cut(sentence, use_jieba = False):
    if use_jieba:
        return ''.join(jieba.cut(sentence))
    else:
        words = [word for word in sentence]
        return ''.join(words)

import jieba.posseg as pseg

def tag(sentence):
    words = pseg.cut(sentence)
    result = ''
    for w in words:
        result = result + w.word + "/" + w.flag + ""
    return result

def readLangs(lang1, lang2, reverse = False):
    logger.info("Reading lines...")
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding = 'utf-8').read().strip().split('\n')


    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return  input_lang, output_lang, pairs


eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    return  len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse = False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    logger.info("read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    logger.info("Trimmed to %s sentence pairs" % len(pairs))
    logger.info("Counting words...")
    for pair in pairs:
        input_lang.addSentences(pair[0])
        output_lang.addSentences(pair[1])
    logger.info("Counted words:")
    logger.info('%s, %d' % (input_lang.name, input_lang.n_words))
    logger.info('%s, %d' % (output_lang.name, output_lang.n_words))
    return input_lang, output_lang, pairs

def indexFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ') if len(word) > 0]

def variableFromSentence(lang, sentence):
    if lang.need_cut:
        sentence = cut(sentence)
    indexes = indexFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variableFromPair(input_lang, output_lang, pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s' % (asMinutes(s), asMinutes(rs))

if __name__ == "__main__":
    s = 'Fans of Belgium cheer prior to the 2018 FIFA World Cup Group G match between Belgium and Tunisia in Moscow, Russia, June 23, 2018.'
    s = '结婚的和尚未结婚的和尚'
    s = "买张下周三去南海的飞机票，海航的"
    s = "过几天天天天气不好。"

    a = cut(s, use_jieba=True)
    print(a)
    print(tag(s))