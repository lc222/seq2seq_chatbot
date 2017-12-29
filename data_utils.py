from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import nltk

import pickle
import random

padToken, goToken, eosToken, unknownToken = 0, 1, 2, 3

class Batch:
    #batch类，里面包含了encoder输入，decoder输入，decoder标签，decoder样本长度mask
    def __init__(self):
        self.encoderSeqs = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.weights = []

def loadDataset(filename):
    '''
    读取样本数据
    :param filename: 文件路径，是一个字典，包含word2id、id2word分别是单词与索引对应的字典和反序字典，
                    trainingSamples样本数据，每一条都是QA对
    :return: word2id, id2word, trainingSamples
    '''
    dataset_path = os.path.join(filename)
    print('Loading dataset from {}'.format(dataset_path))
    with open(dataset_path, 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        word2id = data['word2id']
        id2word = data['id2word']
        trainingSamples = data['trainingSamples']
    return word2id, id2word, trainingSamples

def createBatch(samples, en_de_seq_len):
    '''
    根据给出的samples（就是一个batch的数据），进行padding并构造成placeholder所需要的数据形式
    :param samples: 一个batch的样本数据，列表，每个元素都是[question， answer]的形式，id
    :param en_de_seq_len: 列表，第一个元素表示source端序列的最大长度，第二个元素表示target端序列的最大长度
    :return: 处理完之后可以直接传入feed_dict的数据格式
    '''
    batch = Batch()
    #根据样本长度获得batch size大小
    batchSize = len(samples)
    #将每条数据的问题和答案分开传入到相应的变量中
    for i in range(batchSize):
        sample = samples[i]
        batch.encoderSeqs.append(list(reversed(sample[0])))  # 将输入反序，可提高模型效果
        batch.decoderSeqs.append([goToken] + sample[1] + [eosToken])  # Add the <go> and <eos> tokens
        batch.targetSeqs.append(batch.decoderSeqs[-1][1:])  # Same as decoder, but shifted to the left (ignore the <go>)
        # 将每个元素PAD到指定长度，并构造weights序列长度mask标志
        batch.encoderSeqs[i] = [padToken] * (en_de_seq_len[0] - len(batch.encoderSeqs[i])) + batch.encoderSeqs[i]
        batch.weights.append([1.0] * len(batch.targetSeqs[i]) + [0.0] * (en_de_seq_len[1] - len(batch.targetSeqs[i])))
        batch.decoderSeqs[i] = batch.decoderSeqs[i] + [padToken] * (en_de_seq_len[1] - len(batch.decoderSeqs[i]))
        batch.targetSeqs[i] = batch.targetSeqs[i] + [padToken] * (en_de_seq_len[1] - len(batch.targetSeqs[i]))

    #--------------------接下来就是将数据进行reshape操作，变成序列长度*batch_size格式的数据------------------------
    encoderSeqsT = []  # Corrected orientation
    for i in range(en_de_seq_len[0]):
        encoderSeqT = []
        for j in range(batchSize):
            encoderSeqT.append(batch.encoderSeqs[j][i])
        encoderSeqsT.append(encoderSeqT)
    batch.encoderSeqs = encoderSeqsT

    decoderSeqsT = []
    targetSeqsT = []
    weightsT = []
    for i in range(en_de_seq_len[1]):
        decoderSeqT = []
        targetSeqT = []
        weightT = []
        for j in range(batchSize):
            decoderSeqT.append(batch.decoderSeqs[j][i])
            targetSeqT.append(batch.targetSeqs[j][i])
            weightT.append(batch.weights[j][i])
        decoderSeqsT.append(decoderSeqT)
        targetSeqsT.append(targetSeqT)
        weightsT.append(weightT)
    batch.decoderSeqs = decoderSeqsT
    batch.targetSeqs = targetSeqsT
    batch.weights = weightsT

    return batch

def getBatches(data, batch_size, en_de_seq_len):
    '''
    根据读取出来的所有数据和batch_size将原始数据分成不同的小batch。对每个batch索引的样本调用createBatch函数进行处理
    :param data: loadDataset函数读取之后的trainingSamples，就是QA对的列表
    :param batch_size: batch大小
    :param en_de_seq_len: 列表，第一个元素表示source端序列的最大长度，第二个元素表示target端序列的最大长度
    :return: 列表，每个元素都是一个batch的样本数据，可直接传入feed_dict进行训练
    '''
    #每个epoch之前都要进行样本的shuffle
    random.shuffle(data)
    batches = []
    data_len = len(data)
    def genNextSamples():
        for i in range(0, data_len, batch_size):
            yield data[i:min(i + batch_size, data_len)]

    for samples in genNextSamples():
        batch = createBatch(samples, en_de_seq_len)
        batches.append(batch)
    return batches

def sentence2enco(sentence, word2id, en_de_seq_len):
    '''
    测试的时候将用户输入的句子转化为可以直接feed进模型的数据，现将句子转化成id，然后调用createBatch处理
    :param sentence: 用户输入的句子
    :param word2id: 单词与id之间的对应关系字典
    :param en_de_seq_len: 列表，第一个元素表示source端序列的最大长度，第二个元素表示target端序列的最大长度
    :return: 处理之后的数据，可直接feed进模型进行预测
    '''
    if sentence == '':
        return None
    #分词
    tokens = nltk.word_tokenize(sentence)
    if len(tokens) > en_de_seq_len[0]:
        return None
    #将每个单词转化为id
    wordIds = []
    for token in tokens:
        wordIds.append(word2id.get(token, unknownToken))
    #调用createBatch构造batch
    batch = createBatch([[wordIds, []]], en_de_seq_len)
    return batch
