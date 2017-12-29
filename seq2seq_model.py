import tensorflow as tf
from seq2seq import embedding_attention_seq2seq
class Seq2SeqModel():

    def __init__(self, source_vocab_size, target_vocab_size, en_de_seq_len, hidden_size, num_layers,
                 batch_size, learning_rate, num_samples=1024,
                 forward_only=False, beam_search=True, beam_size=10):
        '''
        初始化并创建模型
        :param source_vocab_size:encoder输入的vocab size
        :param target_vocab_size: decoder输入的vocab size，这里跟上面一样
        :param en_de_seq_len: 源和目的序列最大长度
        :param hidden_size: RNN模型的隐藏层单元个数
        :param num_layers: RNN堆叠的层数
        :param batch_size: batch大小
        :param learning_rate: 学习率
        :param num_samples: 计算loss时做sampled softmax时的采样数
        :param forward_only: 预测时指定为真
        :param beam_search: 预测时是采用greedy search还是beam search
        :param beam_size: beam search的大小
        '''
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.en_de_seq_len = en_de_seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.num_samples = num_samples
        self.forward_only = forward_only
        self.beam_search = beam_search
        self.beam_size = beam_size
        self.global_step = tf.Variable(0, trainable=False)

        output_projection = None
        softmax_loss_function = None
        # 定义采样loss函数，传入后面的sequence_loss_by_example函数
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w = tf.get_variable('proj_w', [hidden_size, self.target_vocab_size])
            w_t = tf.transpose(w)
            b = tf.get_variable('proj_b', [self.target_vocab_size])
            output_projection = (w, b)
            #调用sampled_softmax_loss函数计算sample loss，这样可以节省计算时间
            def sample_loss(logits, labels):
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(w_t, b, labels=labels, inputs=logits, num_sampled=num_samples, num_classes=self.target_vocab_size)
            softmax_loss_function = sample_loss

        self.keep_drop = tf.placeholder(tf.float32)
        # 定义encoder和decoder阶段的多层dropout RNNCell
        def create_rnn_cell():
            encoDecoCell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
            encoDecoCell = tf.contrib.rnn.DropoutWrapper(encoDecoCell, input_keep_prob=1.0, output_keep_prob=self.keep_drop)
            return encoDecoCell
        encoCell = tf.contrib.rnn.MultiRNNCell([create_rnn_cell() for _ in range(num_layers)])

        # 定义输入的placeholder，采用了列表的形式
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.decoder_targets = []
        self.target_weights = []
        for i in range(en_de_seq_len[0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None, ], name="encoder{0}".format(i)))
        for i in range(en_de_seq_len[1]):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None, ], name="decoder{0}".format(i)))
            self.decoder_targets.append(tf.placeholder(tf.int32, shape=[None, ], name="target{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None, ], name="weight{0}".format(i)))

        # test模式，将上一时刻输出当做下一时刻输入传入
        if forward_only:
            if beam_search:#如果是beam_search的话，则调用自己写的embedding_attention_seq2seq函数，而不是legacy_seq2seq下面的
                self.beam_outputs, _, self.beam_path, self.beam_symbol = embedding_attention_seq2seq(
                    self.encoder_inputs, self.decoder_inputs, encoCell, num_encoder_symbols=source_vocab_size,
                    num_decoder_symbols=target_vocab_size, embedding_size=hidden_size,
                    output_projection=output_projection, feed_previous=True)
            else:
                decoder_outputs, _ = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    self.encoder_inputs, self.decoder_inputs, encoCell, num_encoder_symbols=source_vocab_size,
                    num_decoder_symbols=target_vocab_size, embedding_size=hidden_size,
                    output_projection=output_projection, feed_previous=True)
                # 因为seq2seq模型中未指定output_projection，所以需要在输出之后自己进行output_projection
                if output_projection is not None:
                    self.outputs = tf.matmul(decoder_outputs, output_projection[0]) + output_projection[1]
        else:
            # 因为不需要将output作为下一时刻的输入，所以不用output_projection
            decoder_outputs, _ = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                self.encoder_inputs, self.decoder_inputs, encoCell, num_encoder_symbols=source_vocab_size,
                num_decoder_symbols=target_vocab_size, embedding_size=hidden_size, output_projection=output_projection,
                feed_previous=False)
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss(
                decoder_outputs, self.decoder_targets, self.target_weights, softmax_loss_function=softmax_loss_function)

            # Initialize the optimizer
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
            self.optOp = opt.minimize(self.loss)

        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, session, encoder_inputs, decoder_inputs, decoder_targets, target_weights, go_token_id):
        #传入一个batch的数据，并训练性对应的模型
        # 构建sess.run时的feed_inpits
        feed_dict = {}
        if not self.forward_only:
            feed_dict[self.keep_drop] = 0.5
            for i in range(self.en_de_seq_len[0]):
                feed_dict[self.encoder_inputs[i].name] = encoder_inputs[i]
            for i in range(self.en_de_seq_len[1]):
                feed_dict[self.decoder_inputs[i].name] = decoder_inputs[i]
                feed_dict[self.decoder_targets[i].name] = decoder_targets[i]
                feed_dict[self.target_weights[i].name] = target_weights[i]
            run_ops = [self.optOp, self.loss]
        else:
            feed_dict[self.keep_drop] = 1.0
            for i in range(self.en_de_seq_len[0]):
                feed_dict[self.encoder_inputs[i].name] = encoder_inputs[i]
            feed_dict[self.decoder_inputs[0].name] = [go_token_id]
            if self.beam_search:
                run_ops = [self.beam_path, self.beam_symbol]
            else:
                run_ops = [self.outputs]

        outputs = session.run(run_ops, feed_dict)
        if not self.forward_only:
            return None, outputs[1]
        else:
            if self.beam_search:
                return outputs[0], outputs[1]