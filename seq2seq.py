from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import copy

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

Linear = rnn_cell_impl._Linear  # pylint: disable=protected-access,invalid-name

def _extract_beam_search(embedding, beam_size, num_symbols, embedding_size, output_projection=None):

    def loop_function(prev, i, log_beam_probs, beam_path, beam_symbols):
        if output_projection is not None:
            prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
        # 对输出概率进行归一化和取log，这样序列概率相乘就可以变成概率相加
        probs = tf.log(tf.nn.softmax(prev))
        if i == 1:
            probs = tf.reshape(probs[0, :], [-1, num_symbols])
        if i > 1:
            # 将当前序列的概率与之前序列概率相加得到结果之前有beam_szie个序列，本次产生num_symbols个结果，
            # 所以reshape成这样的tensor
            probs = tf.reshape(probs + log_beam_probs[-1], [-1, beam_size * num_symbols])
        # 选出概率最大的前beam_size个序列,从beam_size * num_symbols个元素中选出beam_size个
        best_probs, indices = tf.nn.top_k(probs, beam_size)
        indices = tf.stop_gradient(tf.squeeze(tf.reshape(indices, [-1, 1])))
        best_probs = tf.stop_gradient(tf.reshape(best_probs, [-1, 1]))

        # beam_size * num_symbols，看对应的是哪个序列和单词
        symbols = indices % num_symbols  # Which word in vocabulary.
        beam_parent = indices // num_symbols  # Which hypothesis it came from.
        beam_symbols.append(symbols)
        beam_path.append(beam_parent)
        log_beam_probs.append(best_probs)

        # 对beam-search选出的beam size个单词进行embedding，得到相应的词向量
        emb_prev = embedding_ops.embedding_lookup(embedding, symbols)
        emb_prev = tf.reshape(emb_prev, [-1, embedding_size])
        return emb_prev

    return loop_function

def beam_attention_decoder(decoder_inputs,
                          initial_state,
                          attention_states,
                          cell,
                           embedding,
                          output_size=None,
                          num_heads=1,
                          loop_function=None,
                          dtype=None,
                          scope=None,
                          initial_state_attention=False, output_projection=None, beam_size=10):
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if not attention_states.get_shape()[1:2].is_fully_defined():
        raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                         % attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with variable_scope.variable_scope(scope or "attention_decoder", dtype=dtype) as scope:
        dtype = scope.dtype
        # batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        if attn_length is None:
            attn_length = array_ops.shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2].value

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(attention_states, [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in xrange(num_heads):
            k = variable_scope.get_variable("AttnW_%d" % a, [1, 1, attn_size, attention_vec_size])
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))

        state = []
        # 将encoder的最后一个隐层状态扩展成beam_size维，因为decoder阶段的batch_size是beam_size。
        # initial_state是一个列表，RNN有多少层就有多少个元素，每个元素都是一个LSTMStateTuple，包含h,c两个隐层状态
        # 所以要将其扩展成beam_size维，其实是把c和h进行扩展，最后再合成LSTMStateTuple就可以了
        for layers in initial_state:
            c = [layers.c] * beam_size
            h = [layers.h] * beam_size
            c = tf.concat(c, 0)
            h = tf.concat(h, 0)
            state.append(rnn_cell_impl.LSTMStateTuple(c, h))
        state = tuple(state)
        # state_size = int(initial_state.get_shape().with_rank(2)[1])
        # states = []
        # for kk in range(beam_size):
        #     states.append(initial_state)
        # state = tf.concat(states, 0)
        # state = initial_state

        def attention(query):
            ds = []  # Results of attention reads will be stored here.
            if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                for q in query_list:  # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(query_list, 1)
            for a in xrange(num_heads):
                with variable_scope.variable_scope("Attention_%d" % a):
                    y = Linear(query, attention_vec_size, True)(query)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
                    a = nn_ops.softmax(s)
                    # Now calculate the attention-weighted vector d.
                    d = math_ops.reduce_sum(array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            return ds

        outputs = []
        prev = None
        # attention也要定义成beam_size为的tensor
        batch_attn_size = array_ops.stack([beam_size, attn_size])
        attns = [array_ops.zeros(batch_attn_size, dtype=dtype) for _ in xrange(num_heads)]
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
        if initial_state_attention:
            attns = attention(initial_state)

        log_beam_probs, beam_path, beam_symbols = [], [], []
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if i == 0:
                #i=0时，输入时一个batch_szie=beam_size的tensor，且里面每个元素的值都是相同的，都是<GO>标志
                inp = tf.nn.embedding_lookup(embedding, tf.constant(1, dtype=tf.int32, shape=[beam_size]))

            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i, log_beam_probs, beam_path, beam_symbols)
            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            inputs = [inp] + attns
            x = Linear(inputs, input_size, True)(inputs)

            # Run the RNN.
            cell_output, state = cell(x, state)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True):
                    attns = attention(state)
            else:
                attns = attention(state)

            with variable_scope.variable_scope("AttnOutputProjection"):
                inputs = [cell_output] + attns
                output = Linear(inputs, output_size, True)(inputs)
            if loop_function is not None:
                prev = output
            outputs.append(tf.argmax(nn_ops.xw_plus_b(output, output_projection[0], output_projection[1]), axis=1))

    return outputs, state, tf.reshape(tf.concat(beam_path, 0), [-1, beam_size]), tf.reshape(tf.concat(beam_symbols, 0),
                                                                                            [-1, beam_size])

def embedding_attention_decoder(decoder_inputs,
                                initial_state,
                                attention_states,
                                cell,
                                num_symbols,
                                embedding_size,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False, beam_search=True, beam_size=10):
    if output_size is None:
        output_size = cell.output_size
    if output_projection is not None:
        proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    with variable_scope.variable_scope(scope or "embedding_attention_decoder", dtype=dtype) as scope:
        embedding = variable_scope.get_variable("embedding", [num_symbols, embedding_size])
        emb_inp = [embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
        loop_function = _extract_beam_search(embedding, beam_size, num_symbols, embedding_size, output_projection)
        return beam_attention_decoder(
            emb_inp, initial_state, attention_states, cell, embedding, output_size=output_size,
            num_heads=num_heads, loop_function=loop_function,
            initial_state_attention=initial_state_attention, output_projection=output_projection,
            beam_size=beam_size)


def embedding_attention_seq2seq(encoder_inputs,
                                decoder_inputs,
                                cell,
                                num_encoder_symbols,
                                num_decoder_symbols,
                                embedding_size,
                                num_heads=1,
                                output_projection=None,
                                feed_previous=False,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False, beam_search=True, beam_size=10):
    with variable_scope.variable_scope(scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
        dtype = scope.dtype
        # Encoder.
        encoder_cell = copy.deepcopy(cell)
        encoder_cell = core_rnn_cell.EmbeddingWrapper(encoder_cell, embedding_classes=num_encoder_symbols, embedding_size=embedding_size)
        encoder_outputs, encoder_state = rnn.static_rnn(encoder_cell, encoder_inputs, dtype=dtype)

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [array_ops.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs]
        attention_states = array_ops.concat(top_states, 1)

        # Decoder.
        output_size = None
        if output_projection is None:
            cell = core_rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
            output_size = num_decoder_symbols

        return embedding_attention_decoder(
            decoder_inputs,
            encoder_state,
            attention_states,
            cell,
            num_decoder_symbols,
            embedding_size,
            num_heads=num_heads,
            output_size=output_size,
            output_projection=output_projection,
            feed_previous=feed_previous,
            initial_state_attention=initial_state_attention, beam_search=beam_search, beam_size=beam_size)

