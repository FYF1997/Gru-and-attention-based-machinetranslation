import pickle
import matplotlib.pyplot as plt
import torch
from machine_translation.logger import logger
from machine_translation.train import evaluate,evaluateAndShowAtten,evaluateRandomly

input = 'eng'
output = 'cmn'
logger.info(('%s -> %s') % (input, output))
input_lang = pickle.load(open('./data/%s_%s_input_lang.pkl' % (input, output), "rb"))
output_lang = pickle.load(open('./data/%s_%s_output_lang.pkl' % (input, output), "rb"))
pairs = pickle.load(open('./data/%s_%s_pairs.pkl' % (input, output), 'rb'))
logger.info('lang loaded.')

# 加载训练好的编码器和解码器
encoder1 = torch.load(open('./data/%s_%s_encoder1.model' % (input, output), 'rb'))
attn_decoder1 = torch.load(open('./data/%s_%s_attn_decoder1.model' % (input, output), 'rb'))
logger.info('model loaded.')

# 对单句进行评估并绘制注意力图像
def evaluateAndShowAttention(sentence):
    evaluateAndShowAtten(input_lang, output_lang, sentence, encoder1, attn_decoder1)

evaluateAndShowAttention("他们肯定会相恋的。")
evaluateAndShowAttention("我现在正在学习。")

# 语料中的数据随机选择评估
evaluateRandomly(input_lang, output_lang, pairs, encoder1, attn_decoder1)

output_words, attentions = evaluate(input_lang, output_lang,
                                    encoder1, attn_decoder1, "我是中国人。")
plt.matshow(attentions.numpy())