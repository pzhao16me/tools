

import os

import jieba
#E:/work/chatbot/demo9/data/test
source_path = 'E:/work/chatbot/demo9/data/test/answer.txt'
target_path = 'E:/work/chatbot/demo9/data/test/response.txt.'


# source_text = helper.load_data(source_path)
# target_text = helper.load_data(target_path)
def get_lines(filename):
    with open(filename, encoding="utf-8", errors="ignore") as f:
        lines = []
        sentences = f.readlines()
        lines = [[word for word in "".join(jieba.cut(sentence, cut_all=False))]
                 for _, sentence in enumerate(sentences)]
        return lines


def get_vocab(filename):
    lines = []
    with open(filename, encoding="utf-8", errors="ignore") as f:
        sentences = f.readlines()

        for idx, sentence in enumerate(sentences):
            # print(idx,"-->",sentence)
            seg_list = jieba.cut(sentence, cut_all=False)
            seg_list = "".join(seg_list)
            line = []
            for word in seg_list:
                line.append(word)
            # print(line)
            # print(idx,len(line))
            lines.extend(line)
        vocab = set(lines)
        word_int = {}
        word_int['<PAD>'] = 0
        word_int['<EOS>'] = 1
        word_int['<UNK>'] = 2
        word_int['<GO>'] = 3
        for idx, item in enumerate(vocab):
            word_int[item] = idx + 4
        return word_int


view_sentence_range = (0, 10)
source_text = get_lines(source_path)
target_text = get_lines(target_path)
vocab_size = len(get_vocab(source_path))
source_vocab_to_int = get_vocab(source_path)
target_vocab_to_int = get_vocab(target_path)
source_int_to_vocab=dict((v,k) for k, v in source_vocab_to_int.items())
target_int_to_vocab=dict((v,k) for k, v in target_vocab_to_int.items())
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(vocab_size))

sentences = len(source_text)
word_counts = [len(item) for item in source_text]
print('Number of sentences: {}'.format(sentences))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))

print()
print("the 10 sentences of source_text")


# show_example(source_text)
# show_example(target_text)


def show_example(text):
    print("begin to show some example ", '\n')

    for i in range(10):
        print(text[i])


show_example(source_text)


def get_word2vec(filename):
    word_int = get_vocab(source_path)
    with open(filename, encoding="utf-8", errors="ignore") as f:
        sentences = f.readlines()
        vocab_to_int = [[word_int.get(word, word_int["<UNK>"])
                         for word in "".join(jieba.cut(sentence, cut_all=False))]
                        for _, sentence in enumerate(sentences)]
        return vocab_to_int


def text_to_ids(source_path, target_path):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    # TODO: Implement Function
    source_int_text = get_word2vec(source_path)
    target_int_text = get_word2vec(target_path)
    return source_int_text, target_int_text


source_int_text, target_int_text = text_to_ids(source_path, target_path)
#source_int_text, target_int_text = text_to_ids(source_path, target_path)
source_vocab_to_int=get_vocab(source_path)
target_vocab_to_int=get_vocab(target_path)
# show t 10 sample of the source_int_text
len(source_int_text)
print("show the source word2vec:")
for i in range(10):

    print(source_int_text[i])

# show t 10 sample of the source_int_text
len(source_int_text)
print("show the target word2vec:")
for i in range(10):

    print(source_int_text[i])
# show the version tensorflow
from distutils.version import LooseVersion
import warnings
import tensorflow as tf
from tensorflow.python.layers.core import Dense

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.1'), 'Please use TensorFlow version 1.1 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def model_inputs():
    """
    Create TF Placeholders for input, targets, learning rate, and lengths of source and target sequences.
    :return: Tuple (input, targets, learning rate, keep probability, target sequence length,
    max target sequence length, source sequence length)
    """
    # TODO: Implement Function
    input = tf.placeholder(tf.int32, [None, None], name="input")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    lr = tf.placeholder(tf.float32, name="learning_rate")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    target_sequence_length = tf.placeholder(tf.int32, (None,), name="target_sequence_length")
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name="max_target_length")
    source_sequence_length = tf.placeholder(tf.int32, (None,), name="source_sequence_length")

    return input, targets, lr, keep_prob, target_sequence_length, max_target_sequence_length, source_sequence_length


def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for encoding
    :param target_data: Target Placehoder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """
    # TODO: Implement Function
    encoding = tf.strided_slice(target_data,[0,0],[batch_size,-1],[1,1])
    ppred_data = tf.concat([tf.fill([batch_size,1],target_vocab_to_int["<GO>"]),encoding],1)
    return ppred_data

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob,
                   source_sequence_length, source_vocab_size,
                   encoding_embedding_size):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :param source_sequence_length: a list of the lengths of each sequence in the batch
    :param source_vocab_size: vocabulary size of source data
    :param encoding_embedding_size: embedding size of source data
    :return: tuple (RNN output, RNN state)
    """
    # TODO: Implement Function
    enc_embed_input = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size+1, encoding_embedding_size)
    def make_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        drop = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        return drop
    enc_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])
    enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_embed_input, sequence_length=source_sequence_length, dtype=tf.float32)
    return enc_output, enc_state


def decoding_layer_train(encoder_state, dec_cell, dec_embed_input,
                         target_sequence_length, max_summary_length,
                         output_layer, keep_prob):
    """
    Create a decoding layer for training
    :param encoder_state: Encoder State
    :param dec_cell: Decoder RNN Cell
    :param dec_embed_input: Decoder embedded input
    :param target_sequence_length: The lengths of each sequence in the target batch
    :param max_summary_length: The length of the longest sequence in the batch
    :param output_layer: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: BasicDecoderOutput containing training logits and sample_id
    """
    # TODO: Implement Function
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input, sequence_length=target_sequence_length,
                                                        time_major=False)
    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, training_helper, encoder_state, output_layer)
    training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder, impute_finished=True,
                                                                      maximum_iterations=max_summary_length)

    return training_decoder_output
def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
    """
    Create a decoding layer for inference
    :param encoder_state: Encoder state
    :param dec_cell: Decoder RNN Cell
    :param dec_embeddings: Decoder embeddings
    :param start_of_sequence_id: GO ID
    :param end_of_sequence_id: EOS Id
    :param max_target_sequence_length: Maximum length of target sequences
    :param vocab_size: Size of decoder/target vocabulary
    :param decoding_scope: TenorFlow Variable Scope for decoding
    :param output_layer: Function to apply the output layer
    :param batch_size: Batch size
    :param keep_prob: Dropout keep probability
    :return: BasicDecoderOutput containing inference logits and sample_id
    """
    # TODO: Implement Function
    start_tokens = tf.tile(tf.constant([start_of_sequence_id], dtype=tf.int32), [batch_size], name="start_tokens")
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, start_tokens, end_of_sequence_id)
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, inference_helper, encoder_state, output_layer)
    inference_decoder_output,_ ,_= tf.contrib.seq2seq.dynamic_decode(inference_decoder, impute_finished=True,
                                                                    maximum_iterations=max_target_sequence_length)
    return inference_decoder_output


def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):
    """
    Create decoding layer
    :param dec_input: Decoder input
    :param encoder_state: Encoder state
    :param target_sequence_length: The lengths of each sequence in the target batch
    :param max_target_sequence_length: Maximum length of target sequences
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param target_vocab_size: Size of target vocabulary
    :param batch_size: The size of the batch
    :param keep_prob: Dropout keep probability
    :param decoding_embedding_size: Decoding embedding size
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    # TODO: Implement Function
    dec_embedding = tf.Variable(tf.random_uniform([target_vocab_size+4, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embedding, dec_input)

    # construct decoder cell
    def make_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        drop = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        return drop

    dec_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])
    # output layer

    output_layer = Dense(target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    # decoding layer for train
    with tf.variable_scope("decode"):
        training_decoder_output = decoding_layer_train(encoder_state, dec_cell, dec_embed_input, target_sequence_length,
                                                       max_target_sequence_length, output_layer, keep_prob)
    # decoding for inference
    with tf.variable_scope("decode", reuse=True):
        inference_decoder_output = decoding_layer_infer(encoder_state, dec_cell, dec_embedding,
                                                        target_vocab_to_int["<GO>"],
                                                        target_vocab_to_int["<EOS>"], max_target_sequence_length,
                                                        target_vocab_size,
                                                        output_layer, batch_size, keep_prob)

    return training_decoder_output, inference_decoder_output



def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  source_sequence_length, target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param source_sequence_length: Sequence Lengths of source sequences in the batch
    :param target_sequence_length: Sequence Lengths of target sequences in the batch
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    # TODO: Implement Function
    _, enc_state = encoding_layer(input_data, rnn_size, num_layers, keep_prob, source_sequence_length, source_vocab_size, enc_embedding_size)
    dec_input = process_decoder_input(target_data, target_vocab_to_int, batch_size)
    training_decoder_output, inference_decoder_output = decoding_layer(dec_input, enc_state, target_sequence_length,
                                                                       max_target_sentence_length, rnn_size, num_layers,
                                                                       target_vocab_to_int, target_vocab_size, batch_size,
                                                                       keep_prob, dec_embedding_size)
    return training_decoder_output, inference_decoder_output





# Number of Epochs
epochs = 10
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 512
# Number of Layers
num_layers = 6
# Embedding Size
encoding_embedding_size = 1024
decoding_embedding_size = 1024
# Learning Rate
learning_rate = 0.001
# Dropout Keep Probability
keep_probability = 0.7
display_step = 20


save_path = 'checkpoints/dev'
source_int_text, target_int_text = text_to_ids(source_path, target_path)
source_vocab_to_int=get_vocab(source_path)
target_vocab_to_int=get_vocab(target_path)
#(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
max_target_sentence_length = max([len(sentence) for sentence in source_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, keep_prob, target_sequence_length, max_target_sequence_length, source_sequence_length = model_inputs()

    #sequence_length = tf.placeholder_with_default(max_target_sentence_length, None, name='sequence_length')
    input_shape = tf.shape(input_data)

    train_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                   targets,
                                                   keep_prob,
                                                   batch_size,
                                                   source_sequence_length,
                                                   target_sequence_length,
                                                   max_target_sequence_length,
                                                   len(source_vocab_to_int),
                                                   len(source_vocab_to_int),#  modeified
                                                   encoding_embedding_size,
                                                   decoding_embedding_size,
                                                   rnn_size,
                                                   num_layers,
                                                   target_vocab_to_int)


    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
max_target_sentence_length




def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths


def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))

# Split data to training and validation sets
train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]
valid_source = source_int_text[:batch_size]
valid_target = target_int_text[:batch_size]
(valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths ) = next(get_batches(valid_source,
                                                                                                             valid_target,
                                                                                                             batch_size,
                                                                                                             source_vocab_to_int['<PAD>'],
                                                                                                             target_vocab_to_int['<PAD>']))
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                get_batches(train_source, train_target, batch_size,
                            source_vocab_to_int['<PAD>'],
                            target_vocab_to_int['<PAD>'])):
            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 source_sequence_length: sources_lengths,
                 keep_prob: keep_probability})

            if batch_i % display_step == 0 and batch_i > 0:
                batch_train_logits = sess.run(
                    inference_logits,
                    {input_data: source_batch,
                     source_sequence_length: sources_lengths,
                     target_sequence_length: targets_lengths,
                     keep_prob: 1.0})

                batch_valid_logits = sess.run(
                    inference_logits,
                    {input_data: valid_sources_batch,
                     source_sequence_length: valid_sources_lengths,
                     target_sequence_length: valid_targets_lengths,
                     keep_prob: 1.0})

                train_acc = get_accuracy(target_batch, batch_train_logits)

                valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)

                print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                      .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

                export_path_base = "E:/work/chatbot/demo10/model/"
                export_path = os.path.join(tf.compat.as_bytes(export_path_base),tf.compat.as_bytes(str(1.0))
                print('Exporting trained model to', export_path)
  				builder = tf.saved_model.builder.SavedModelBuilder(export_path)
  				builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING])





