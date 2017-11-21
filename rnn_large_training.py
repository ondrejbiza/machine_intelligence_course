import os, sys, tarfile
import collections
from six.moves.urllib.request import urlretrieve
import numpy as np
import tensorflow as tf

url = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/'
data_root = 'data/rnn'
last_percent_reported = None

# make sure the dataset directory exists
if not os.path.isdir(data_root):
  os.makedirs(data_root)

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
    
def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  dest_filename = os.path.join(data_root, filename)
  if force or not os.path.exists(dest_filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(dest_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', dest_filename)
  else:
    raise Exception(
      'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
  return dest_filename

train_filename = maybe_download('simple-examples.tgz', 34869662)

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall(data_root)
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  
  print(data_folders)
  return data_folders
  
train_folders = maybe_extract(train_filename)

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename, wordsLimit=None):
  data = _read_words(filename)
  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  if (wordsLimit!=None):
        count_pairs = count_pairs[0:wordsLimit]
  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  return word_to_id

def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]

def ptb_raw_data(data_path=None, wordsLimit=None):
  """Load PTB raw data from data directory "data_path".
  Reads PTB text files, converts strings to integer ids, and performs mini-batching of the inputs.
  Args:
    data_path: string path to the directory where simple-examples.tgz has been extracted.
  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path, wordsLimit)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary

# in case of very slow learning we can trim an input dictionary (10000 is a base)
wordsLimit=10000

train_data, valid_data, test_data, vocabulary = ptb_raw_data(os.path.join(data_root, 'simple-examples/data'), wordsLimit)
vocab = _build_vocab(os.path.join(data_root, 'simple-examples','data','ptb.test.txt'), wordsLimit)
firstitems = {k: vocab[k] for k in sorted(vocab.keys())[:30]}

print('train data len:', len(train_data))
print('validation data len:', len(valid_data))
print('test data len:', len(test_data))
print('vocabulary item count:', vocabulary)
print('the first 30 vocabulary items:', firstitems)

def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.
  This chunks up raw_data into batches of examples and returns Tensors that are drawn from these batches.
  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).
  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.
  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len], [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(epoch_size, message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],[batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y

# it says how many floats will represent coordinates of given word in embeddings
# it is a "width" of embeddings
state_size = 650

# default value for weights in RNN_logits
init_scale = 0.05

# one piece of unrolled vanilla RNN
def LSTM_step(previous_hidden_state, input_tensor, init_scale=0.1):
    
    # weights for input
    W = tf.get_variable('W', shape=[4, state_size, state_size], 
                        initializer=tf.random_uniform_initializer(minval=-init_scale, maxval=init_scale))
    # weights for previous hidden state
    U = tf.get_variable('U', shape=[4, state_size, state_size], 
                        initializer=tf.random_uniform_initializer(minval=-init_scale, maxval=init_scale))
    
    bi = tf.get_variable("bi", shape=[state_size], initializer=tf.constant_initializer(0.))
    bf = tf.get_variable("bf", shape=[state_size], initializer=tf.constant_initializer(0.))
    bo = tf.get_variable("bo", shape=[state_size], initializer=tf.constant_initializer(0.))
    bc = tf.get_variable("bc", shape=[state_size], initializer=tf.constant_initializer(0.))
    
    # gather previous internal state and output state
    state, cell = tf.unstack(previous_hidden_state)
    
    # gates
    input_gate = tf.sigmoid(tf.matmul(input_tensor, U[0]) + tf.matmul(state, W[0]) + bi)
    forget_gate = tf.sigmoid(tf.matmul(input_tensor, U[1]) + tf.matmul(state, W[1]) + bf)
    output_gate = tf.sigmoid(tf.matmul(input_tensor, U[2]) + tf.matmul(state, W[2]) + bo)
    gate_weights = tf.tanh(tf.matmul(input_tensor, U[3]) + tf.matmul(state, W[3]) + bc)
    
    # new internal cell state
    cell = cell * forget_gate + gate_weights * input_gate
    
    # output state
    state = tf.tanh(cell) * output_gate
    return tf.stack([state, cell])

# configuration
num_classes = vocabulary
max_gradient_norm = 5
hidden_size = 650
num_steps = 35
batch_size = 20
num_layers = 2
init_scale = 0.05

learning_rate = 1.0
learning_rate_decay = 0.8
epoch_end_decay = 6

keep_prob = 0.5

num_epochs = 39

# LSTM cell
rnn_type = "LSTM"
tf.reset_default_graph()

# take a subset of data
input_tensor, labels_tensor = ptb_producer(train_data, batch_size=batch_size, num_steps=num_steps)

# TODO: kde se naplni to embeddings?
embeddings = tf.get_variable("embeddings", [num_classes, state_size])
# kdyz se tady z nej maji vybrat hodnoty (sloupce nebo radky?) podle idcek v input_tensor
rnn_inputs = tf.nn.embedding_lookup(embeddings, input_tensor)

def build_layer(rnn_inputs, layer_idx):
    
    with tf.variable_scope("layer{}".format(layer_idx)):
    
        # truncated backprop
        hidden_state = tf.placeholder(tf.float32, shape=[2, batch_size, state_size])
        
        # TODO: proc se to transponuje tam a zase zpatky?
        states = tf.scan(LSTM_step, tf.transpose(rnn_inputs, [1,0,2]), initializer=hidden_state) 
        states = tf.transpose(states, [1,2,0,3])
        
        return states, hidden_state
    
sequence = rnn_inputs
is_training = tf.placeholder(tf.bool)

final_states = []
hidden_states = []

for layer_idx in range(num_layers):
    
    sequence = tf.layers.dropout(sequence, rate=keep_prob, training=is_training)
    
    states, hidden_state = build_layer(sequence, layer_idx)
    final_states.append(states[:, :, -1, :])
    hidden_states.append(hidden_state)
    
    sequence = states[0]
    
sequence = tf.layers.dropout(sequence, rate=keep_prob, training=is_training)
    
states_reshaped = tf.reshape(sequence, [-1, state_size])
logits = RNN_logits(states_reshaped, num_classes)
logits = tf.reshape(logits, [batch_size, num_steps, -1])

predictions = tf.nn.softmax(logits)

# calculate a difference between predited and correct labels
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_tensor)
loss = tf.reduce_sum(losses) / batch_size

learning_rate_tensor = tf.Variable(learning_rate, name="learning_rate")
learning_rate_pl = tf.placeholder(tf.float32, name="learning_rate_pl")
assign_learning_rate = tf.assign(learning_rate_tensor, learning_rate_pl)

trainable_vars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, trainable_vars), max_gradient_norm)
optimizer = tf.train.GradientDescentOptimizer(learning_rate_tensor)
train_op = optimizer.apply_gradients(zip(grads, trainable_vars), 
                                     global_step=tf.contrib.framework.get_or_create_global_step())

epoch_size = ((len(train_data) // batch_size) - 1) // num_steps

with tf.Session() as session:
    
  print("RNN type: ", rnn_type)
  print()
    
  saver = tf.train.Saver()

  session.run(tf.global_variables_initializer())
    
  #
  input_coord = tf.train.Coordinator() 
  input_threads = tf.train.start_queue_runners(session, coord=input_coord)
  #
    
  for epoch in range(num_epochs):
      
      learning_rate_decay = learning_rate_decay ** max(epoch + 1 - epoch_end_decay, 0.0)
      session.run(assign_learning_rate, feed_dict={
          learning_rate_pl: learning_rate * learning_rate_decay
      })
        
      total_loss = 0
      total_time_steps = 0
   
      epoch_hidden_states = []
      for state_pl in hidden_states:
            epoch_hidden_states.append(np.zeros((2, batch_size, state_size)))

      for step in range(epoch_size):

        # build feed dict
        feed_dict = {}
        
        feed_dict[is_training] = True
        for state_pl, state_val in zip(hidden_states, epoch_hidden_states):
            feed_dict[state_pl] = state_val
            
        loss_val, _, epoch_hidden_states = session.run([loss, train_op, final_states], feed_dict=feed_dict)

        total_loss += loss_val
        total_time_steps += num_steps
            
      epoch_perplexity = np.exp(total_loss / total_time_steps)
    
      print("epoch {} - perplexity: {:.3f}".format(epoch + 1, epoch_perplexity))
    
  saver.save(session, "language-rnn", global_step=0)
    
  #  
  input_coord.request_stop()
  input_coord.join(input_threads)  
  #

batch_size = 20

rnn_type = "LSTM"
tf.reset_default_graph()

# take a subset of data
input_tensor, labels_tensor = ptb_producer(valid_data, batch_size=batch_size, num_steps=num_steps)

# TODO: kde se naplni to embeddings?
embeddings = tf.get_variable("embeddings", [num_classes, state_size])
# kdyz se tady z nej maji vybrat hodnoty (sloupce nebo radky?) podle idcek v input_tensor
rnn_inputs = tf.nn.embedding_lookup(embeddings, input_tensor)

def build_layer(rnn_inputs, layer_idx):
    
    with tf.variable_scope("layer{}".format(layer_idx)):
    
        # truncated backprop
        hidden_state = tf.placeholder(tf.float32, shape=[2, batch_size, state_size])
        
        # TODO: proc se to transponuje tam a zase zpatky?
        states = tf.scan(LSTM_step, tf.transpose(rnn_inputs, [1,0,2]), initializer=hidden_state) 
        states = tf.transpose(states, [1,2,0,3])
        
        return states, hidden_state
    
sequence = rnn_inputs

final_states = []
hidden_states = []

for layer_idx in range(num_layers):
    states, hidden_state = build_layer(sequence, layer_idx)
    final_states.append(states[:, :, -1, :])
    hidden_states.append(hidden_state)
    
    sequence = states[0]
    
states_reshaped = tf.reshape(sequence, [-1, state_size])
logits = RNN_logits(states_reshaped, num_classes)
logits = tf.reshape(logits, [batch_size, num_steps, -1])

predictions = tf.nn.softmax(logits)

# calculate a difference between predited and correct labels
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_tensor)
loss = tf.reduce_sum(losses) / batch_size

epoch_size = ((len(valid_data) // batch_size) - 1) // num_steps

with tf.Session() as session:
    
  print("RNN type:", rnn_type)
  print()

  session.run(tf.global_variables_initializer())

  saver = tf.train.Saver()
  saver.restore(session, "language-rnn-0")
    
  #
  input_coord = tf.train.Coordinator() 
  input_threads = tf.train.start_queue_runners(session, coord=input_coord)
  #
    
  for epoch in range(1):
        
      total_loss = 0
      total_time_steps = 0
   
      epoch_hidden_states = []
      for state_pl in hidden_states:
            epoch_hidden_states.append(np.zeros((2, batch_size, state_size)))

      for step in range(epoch_size):

        # build feed dict
        feed_dict = {}
        
        for state_pl, state_val in zip(hidden_states, epoch_hidden_states):
            feed_dict[state_pl] = state_val
            
        loss_val, epoch_hidden_states = session.run([loss, final_states], feed_dict=feed_dict)

        total_loss += loss_val
        total_time_steps += num_steps
            
      epoch_perplexity = np.exp(total_loss / total_time_steps)
    
      print("epoch {} - perplexity: {:.3f}".format(epoch + 1, epoch_perplexity))
        
  #  
  input_coord.request_stop()
  input_coord.join(input_threads)  
  #
