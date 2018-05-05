# -*- coding: utf-8 -*-
"""

@author: manjunath.a
"""


import time
import pandas as pd
from tensorflow.contrib.rnn import LSTMCell, GRUCell
import tensorflow as tf  # Version 1.0 or 0.12
import numpy as np
global seq_length, forecasting_length 

start_time = time.time()

seq_length = 48
forecasting_length = 10


##### Using "YAHOO's" opensource anomaly detection time series data
file_path='/home/yahoo_data/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/'
csv_name = 'real_6.csv' 
 
data = pd.read_csv(file_path+csv_name)

if seq_length > 24:
    high_seqlen_flag = 1
else:
    high_seqlen_flag = 0    
    
def generate_train_test_val_data(data):    
    kept_values = data.value.values    
    tmpdata=np.reshape(kept_values, (-1,1))    
    min_max = (np.min(kept_values, axis=0), np.max(kept_values, axis=0))    
    tmpdata = (tmpdata - min_max[0]) / (min_max[1] - min_max[0])    
    kept_values = []
    for g in range(0,len(tmpdata)):
        kept_values.append(tmpdata[g][0])
    
    X, Y = [], []        
    for i in range(len(kept_values) - seq_length - forecasting_length ):
        t=i+seq_length
        X.append(kept_values[i:t])
        Y.append(kept_values[t:t + forecasting_length])        

    X = np.expand_dims(X, axis=2)
    Y = np.expand_dims(Y, axis=2)

    X_train = X[:(int(len(X)) - 50)]
    Y_train = Y[:(int(len(Y)) - 50)]
    X_val = X[(int(len(X)) - 50):(int(len(X)) - 25)]
    Y_val = Y[(int(len(Y)) - 50):(int(len(Y)) - 25)]
    X_test = X[(int(len(X)) - 25):]
    Y_test = Y[(int(len(Y)) - 25):]  

    X_train = X_train.transpose((1, 0, 2))
    Y_train = Y_train.transpose((1, 0, 2))
    X_test = X_test.transpose((1, 0, 2))
    Y_test = Y_test.transpose((1, 0, 2))
    X_val = X_val.transpose((1, 0, 2))
    Y_val = Y_val.transpose((1, 0, 2))            
    return X_train, Y_train, X_test,  Y_test, X_val, Y_val


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)

def mape(y_true, y_pred):
    denominator = (np.abs(y_true) )
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)

# ## Neural network's hyperparameters

sample_x, sample_y, x_, y_, x__, y__ = generate_train_test_val_data(data)

print("Dimensions  ")
print(sample_x.shape)
print(sample_y.shape)

print("(seq_length, batch_size, output_dim)")

# Internal neural network parameters

seq_length = sample_x.shape[0]
out_length = sample_y.shape[0]
batch_size = 1 

output_dim = input_dim = sample_x.shape[-1]

hidden_dim_list_enc = [16, 32, 12]
hidden_dim_list_dec = [16, 32, 12]
dropout_list = [0.5,0.5,0.5]

layers_stacked_count_enc = len(hidden_dim_list_enc)
layers_stacked_count_dec = len(hidden_dim_list_dec)

hidden_dim = hidden_dim_list_enc[0]
out_hidden_dim = hidden_dim_list_dec[-1]
# Optmizer:
learning_rate = 0.005  # Small lr helps not to diverge during training.
learning_rate = 0.007  # Small lr helps not to diverge during training.
nb_iters = 600
lr_decay = 0.92  # default: 0.9 . Simulated annealing.
momentum = 0.5  # default: 0.0 . Momentum technique in weights update
lambda_l2_reg = 0.003  # L2 regularization of weights - avoids overfitting

# Backward compatibility for TensorFlow's version 0.12:
try:
    tf.nn.seq2seq = tf.contrib.legacy_seq2seq
    tf.nn.rnn_cell = tf.contrib.rnn
    tf.nn.rnn_cell.GRUCell = tf.contrib.rnn.GRUCell
    print("TensorFlow's version : 1.0 (or more)")
except:
    print("TensorFlow's version : 0.12")

tf.reset_default_graph()
sess = tf.InteractiveSession()
#dropout = tf.placeholder(tf.float32)

with tf.variable_scope("Encoder") as scope:
    # Encoder: inputs
    enc_inp = [
        tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
        for t in range(seq_length)
    ]
    # Th encoder cell, multi-layered with dropout
    if high_seqlen_flag == 1:
        cells_enc = []
        for i in range(layers_stacked_count_enc):
            cell_enc = LSTMCell(hidden_dim_list_enc[i])
            cell_enc = tf.contrib.rnn.DropoutWrapper(cell_enc, output_keep_prob=1.0-dropout_list[i])    
            cells_enc.append(cell_enc)
        cell_enc = tf.contrib.rnn.MultiRNNCell(cells_enc)
        #cell_enc = tf.contrib.rnn.MultiRNNCell([LSTMCell(hidden_dim_list_enc[i]) for i in range(layers_stacked_count_enc)])
    else:
        cells_enc = []
        for i in range(layers_stacked_count_enc):
            cell_enc = GRUCell(hidden_dim_list_enc[i])
            cell_enc = tf.contrib.rnn.DropoutWrapper(cell_enc, output_keep_prob=1.0-dropout_list[i])    
            cells_enc.append(cell_enc)
        cell_enc = tf.contrib.rnn.MultiRNNCell(cells_enc)
        #cell_enc = tf.contrib.rnn.MultiRNNCell([GRUCell(hidden_dim_list_enc[i]) for i in range(layers_stacked_count_enc)])

    #cell_enc = tf.contrib.rnn.DropoutWrapper(cell_enc, output_keep_prob=self.keep_prob)    

    encoder_outputs, encoder_final_state = tf.contrib.rnn.static_rnn(cell_enc,
                                              inputs=enc_inp, dtype=tf.float32) 
    
                                     
with tf.variable_scope("Decoder") as scope:
    # Decoder: inputs    
    dummy_zero_input = [tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO_dummy_zero_input")]
    
    # Decoder: expected outputs
    expected_sparse_output = [
        tf.placeholder(tf.float32, shape=(None, output_dim), name="expected_sparse_output_".format(t))
        for t in range(out_length)
    ]

    w_out = tf.Variable(tf.random_normal([out_hidden_dim, output_dim]))
    b_out = tf.Variable(tf.random_normal([output_dim]))
    
    # The decoder, also multi-layered
    if high_seqlen_flag == 1:
        cells_dec = []
        for i in range(layers_stacked_count_dec):
            cell_dec = LSTMCell(hidden_dim_list_dec[i])
            cell_dec = tf.contrib.rnn.DropoutWrapper(cell_dec, output_keep_prob=1.0-dropout_list[i])    
            cells_dec.append(cell_dec)
        cell_dec = tf.contrib.rnn.MultiRNNCell(cells_dec)
        #cell_dec = tf.contrib.rnn.MultiRNNCell([LSTMCell(hidden_dim_list_dec[i]) for i in range(layers_stacked_count_dec)])
    else:
        cells_dec = []
        for i in range(layers_stacked_count_dec):
            cell_dec = GRUCell(hidden_dim_list_dec[i])
            cell_dec = tf.contrib.rnn.DropoutWrapper(cell_dec, output_keep_prob=1.0-dropout_list[i])    
            cells_dec.append(cell_dec)
        cell_dec = tf.contrib.rnn.MultiRNNCell(cells_dec)
        #cell_dec = tf.contrib.rnn.MultiRNNCell([GRUCell(hidden_dim_list_dec[i]) for i in range(layers_stacked_count_dec)])
       
    def loop_fn(time, cell_output, cell_state, loop_state):
        emit_output = cell_output  # == None for time == 0
        if cell_output is None:
            next_cell_state = encoder_final_state            
            next_input = dummy_zero_input[0] 
        else:  
            next_cell_state = cell_state
            next_input = tf.add(tf.matmul(cell_output, w_out), b_out)          

        elements_finished = (time >= out_length)
        next_loop_state = None
        return (elements_finished, next_input, next_cell_state,
                emit_output, next_loop_state)        
    
    decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(cell_dec, loop_fn)
    decoder_outputs = decoder_outputs_ta.stack()
    
    decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
    decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
    
    decoder_predictions = tf.add(tf.matmul(decoder_outputs_flat, w_out), b_out)
    reshaped_outputs = tf.reshape(decoder_predictions, (out_length, -1, output_dim))
    

        
# Training loss and optimizer

with tf.variable_scope('Loss'):    
    epsilon = 0.01  # Smoothing factor, helps MAPE to be well-behaved near zero    
    true_o = expected_sparse_output
    pred_o = reshaped_outputs    
    summ = tf.maximum(tf.abs(true_o) , epsilon)     
    mape_ = tf.abs(pred_o - true_o) / summ #* 2.0
    
    loss =tf.losses.compute_weighted_loss(mape_, loss_collection=None)
    
    
with tf.variable_scope('Optimizer'):  # AdamOptimizer
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate, decay=lr_decay, momentum=momentum)
    train_op = optimizer.minimize(loss)


Xtr, Ytr, Xts, Yts, Xv, Yv = generate_train_test_val_data(data)

print("\n\n", Xtr.shape, Ytr.shape, Xts.shape, Yts.shape, Xv.shape, Yv.shape, "\n")

def train_batch(Xtr, Ytr, batch_size):
    
    feed_dict = {enc_inp[t]: Xtr[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: Ytr[
                     t] for t in range(len(expected_sparse_output))})
    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t

def test_batch(Xts, Yts, batch_size):
       
    feed_dict = {enc_inp[t]: Xts[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: Yts[
                     t] for t in range(len(expected_sparse_output))})
    loss_t = sess.run([loss], feed_dict)
    return loss_t[0]


# Training
train_losses = []
test_losses = []

sess.run(tf.global_variables_initializer())
cnt=0
early_stop = 420
Reps, Features = [], []
tmp_loss = []
for t in range(nb_iters + 1):
    train_loss = train_batch(Xtr, Ytr, batch_size)
    train_losses.append(train_loss)    
    
    test_loss = test_batch(Xts, Yts, batch_size)
    test_losses.append(test_loss)
    print("Step {}/{}, train loss: {}, \tTEST loss: {}".format(t,
                                                               nb_iters, train_loss, test_loss))
    if t > 6:
        if test_loss > np.min(test_losses[:(len(test_losses)-1)]):
            tracking_start = 1
            tmp_loss.append(test_loss)
#    if tracking_start == 1 and patience == 0:
#        tmp_loss=[]            
    if len(tmp_loss) == early_stop:    
        print("Early-stopping and breaking")
        break

print("Final. train loss: {}, \tTEST loss: {}".format(train_loss, test_loss))



print("\n TEST DATA ") #, Xts.shape, Yts.shape)
feed_dict = {enc_inp[t]: Xts[t] for t in range(seq_length)}
#feed_dict.update({dropout : 0.5})
outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])
 
print("MAPE : = ", mape(Yts, outputs))    
test_smape = smape(Yts, outputs)   
print("SMAPE : = ", test_smape)   


print("\n VALIDATION DATA ")#, Xv.shape, Yv.shape)
feed_dict = {enc_inp[t]: Xv[t] for t in range(seq_length)}
#feed_dict.update({dropout : 0.5})
outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])
  
print("MAPE : = ", mape(Yv, outputs))    
print("SMAPE : = ", smape(Yv, outputs),"\n")  



print("\n TRAIn DATA ") #, Xtr.shape, Ytr.shape)
feed_dict = {enc_inp[t]: Xtr[t] for t in range(seq_length)}
#feed_dict.update({dropout : 0.5})
outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])
 
print("MAPE : = ", mape(Ytr, outputs))    
print("SMAPE : = ", smape(Ytr, outputs))   



print("\n---Time Taken to run = %s seconds ---" % (time.time() - start_time))

