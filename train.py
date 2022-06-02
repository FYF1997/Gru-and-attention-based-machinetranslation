import random
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from machine_translation.logger import logger
from machine_translation.process import *
from torch import nn
from torch import optim
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

def evaluate(input_lang, output_lang, encoder, decoder, sentence, max_length = MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden
    decoder_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs

        )

        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoder_words.append('<EOS>')
            break
        else:
            decoder_words.append(output_lang.index2word[ni])
        decoder_input = Variable(torch.LongTensor[[ni]])
        decoder_input = decoder_input.cuda() if use_cuda else  decoder_input

    return decoder_words, decoder_attentions[:di + 1]

teacher_forcing_ratio = 0.5

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length = MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden
        )
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: 以目标作为下一个输入
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: 网络自己预测的输出为下一个输入
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

        # 反向传播
    loss.backward()

    # 网络状态更新
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base = 0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
def trainIters(input_lang, output_lang, pairs, encoder, decoder, n_iters, print_every=1000, plot_every=100,
               learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [variableFromPair(input_lang, output_lang, random.choice(pairs))
                      for i in range(n_iters)]
    # 损失函数定义
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            logger.info('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                               iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


def evaluateRandomly(input_lang, output_lang, pairs, encoder, decoder, n=10):
    '''
    从语料中随机选取句子进行评估
    '''
    for i in range(n):
        pair = random.choice(pairs)
        logger.info('> %s' % pair[0])
        logger.info('= %s' % pair[1])
        output_words, attentions = evaluate(input_lang, output_lang, encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        logger.info('< %s' % output_sentence)
        logger.info('')


def showAttention(input_sentence, output_words, attentions):
    try:
        # 添加绘图中的中文显示
        plt.rcParams['font.sans-serif'] = ['STSong']  # 宋体
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        # 使用 colorbar 初始化绘图
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap='bone')
        fig.colorbar(cax)

        # 设置x，y轴信息
        ax.set_xticklabels([''] + input_sentence.split(' ') +
                           ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + output_words)

        # 显示标签
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()
    except Exception as err:
        logger.error(err)


def evaluateAndShowAtten(input_lang, ouput_lang, input_sentence, encoder1, attn_decoder1):
    output_words, attentions = evaluate(input_lang, ouput_lang,
                                        encoder1, attn_decoder1, input_sentence)
    logger.info('input = %s' % input_sentence)
    logger.info('output = %s' % ' '.join(output_words))
    # 如果是中文需要分词
    if input_lang.name == 'cmn':
        print(input_lang.name)
        input_sentence = cut(input_sentence)
    showAttention(input_sentence, output_words, attentions)