# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:25:04 2018

@author: manjunath.a
"""

import  time
import pandas as pd
import tensorflow as tf  # Version 1.0 or 0.12
import numpy as np
global seq_length, forecasting_length 
start_time = time.time()

seq_length = 128

forecasting_length = 64

print("\n Building COMPOSITE Model : Forecasting Model + Feature Extraction Model \n")

if seq_length > 60:
    high_seqlen_flag = 1
else:
    high_seqlen_flag = 0    
    
### This pickle file has multiple time series data     
data = pd.read_pickle('/home/timeseriesdata/R_MSC_TREND_Daily.pkl')


seq_length = int(data.shape[0]/2)
window_size = seq_length
forecasting_length = int(data.shape[0]/2)

#seq_length = int(data.shape[0]*0.75)
#window_size = seq_length
#forecasting_length = int(data.shape[0]) - int(data.shape[0]*0.75)

if seq_length > 36:
    high_seqlen_flag = 1
else:
    high_seqlen_flag = 0    
    
def generate_x_y_data(data):
    X, Y = [], []
   
    for z in range(0, data.shape[1]):
        kept_values = data.iloc[:,z].values

        #### NORMALIZING kept_values-
        tmpdata=np.reshape(kept_values, (-1,1))
        #MinMaxScaler(copy=False).fit_transform(tmpdata)
        min_max = (np.min(kept_values, axis=0), np.max(kept_values, axis=0))    
        tmpdata = (tmpdata - min_max[0]) / (min_max[1] - min_max[0])    
        kept_values = []
        for g in range(0,len(tmpdata)):
            kept_values.append(tmpdata[g][0])        
          
        X.append(kept_values[0:seq_length])
        Y.append(kept_values[seq_length:])
    # To be able to concat on inner dimension later on:
    X = np.expand_dims(X, axis=2)
    Y = np.expand_dims(Y, axis=2)

    X = X.transpose((1, 0, 2))
    Y = Y.transpose((1, 0, 2))
    return X, Y
    

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)

def mape(y_true, y_pred):
    denominator = ( np.abs(y_true) )
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)

# ## Neural network's hyperparameters

sample_x, sample_y = generate_x_y_data(data)
print("Dimensions of the dataset for 3 X and 3 Y training examples : ")
print(sample_x.shape)
print(sample_y.shape)
print("(seq_length, batch_size, output_dim)")

# Internal neural network parameters
seq_length = sample_x.shape[0]
out_length_dec1 = sample_x.shape[0]
out_length_dec2 = sample_y.shape[0]


output_dim = input_dim = sample_x.shape[-1]

# Number of stacked recurrent cells, on the neural depth axis.

hidden_dim_list_enc = [64, 128, 32]
hidden_dim_list_dec = [64, 128, 32]


layers_stacked_count_enc = len(hidden_dim_list_enc)
layers_stacked_count_dec = len(hidden_dim_list_dec)

hidden_dim = hidden_dim_list_enc[0]
out_hidden_dim = hidden_dim_list_dec[-1]
# Optmizer:
learning_rate = 0.007  # Small lr helps not to diverge during training.
# How many times we perform a training step (therefore how many times we
# show a batch).
nb_iters = 100
#nb_iters = 600

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
# sess.close()
sess = tf.InteractiveSession()


from tensorflow.contrib.rnn import LSTMCell, GRUCell


with tf.variable_scope("Encoder") as scope:
    # Encoder: inputs
    enc_inp = [
        tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
        for t in range(seq_length)
    ]
    # Th encoder cell, multi-layered with dropout
    if high_seqlen_flag == 1:
        cell_enc = tf.contrib.rnn.MultiRNNCell([LSTMCell(hidden_dim_list_enc[i]) for i in range(layers_stacked_count_enc)])
    else:
        cell_enc = tf.contrib.rnn.MultiRNNCell([GRUCell(hidden_dim_list_enc[i]) for i in range(layers_stacked_count_enc)])

    #cell_enc = tf.contrib.rnn.DropoutWrapper(cell_enc, output_keep_prob=self.keep_prob)
    
    encoder_outputs, encoder_final_state = tf.contrib.rnn.static_rnn(cell_enc,
                                              inputs=enc_inp, dtype=tf.float32) #,
                                              #initial_state=initial_state_enc)   
    
    Feats = encoder_outputs[-1]                                                                                

with tf.variable_scope("Decoder1") as scope:
    print("\n DECODER 1 Feature Extraction: Output = Input \n")      
  
    # Decoder: inputs    
    dummy_zero_input1 = [tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO_dummy_zero_input")]
    
    # Decoder: expected outputs
    expected_sparse_output1 = [
        tf.placeholder(tf.float32, shape=(None, output_dim), name="expected_sparse_output_".format(t))
        for t in range(out_length_dec1)
    ]

    w_out1 = tf.Variable(tf.random_normal([out_hidden_dim, output_dim]))
    b_out1 = tf.Variable(tf.random_normal([output_dim]))
    
    # The decoder, also multi-layered
    if high_seqlen_flag == 1:
        cell_dec1 = tf.contrib.rnn.MultiRNNCell([LSTMCell(hidden_dim_list_dec[i]) for i in range(layers_stacked_count_dec)])
    else:
        cell_dec1 = tf.contrib.rnn.MultiRNNCell([GRUCell(hidden_dim_list_dec[i]) for i in range(layers_stacked_count_dec)])
       
    def loop_fn1(time, cell_output, cell_state, loop_state):
        emit_output = cell_output  # == None for time == 0
        if cell_output is None:
            next_cell_state = encoder_final_state            
            next_input = dummy_zero_input1[0] 
        else:  
            next_cell_state = cell_state
            next_input = tf.add(tf.matmul(cell_output, w_out1), b_out1)          

        elements_finished = (time >= out_length_dec1)
        next_loop_state = None
        return (elements_finished, next_input, next_cell_state,
                emit_output, next_loop_state)        
    
    decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(cell_dec1, loop_fn1)
    decoder_outputs1 = decoder_outputs_ta.stack()
    
    decoder_max_steps, decoder_batch_size, decoder_dim1 = tf.unstack(tf.shape(decoder_outputs1))
    decoder_outputs_flat1 = tf.reshape(decoder_outputs1, (-1, decoder_dim1))
    
    decoder_predictions1 = tf.add(tf.matmul(decoder_outputs_flat1, w_out1), b_out1)
    reshaped_outputs1 = tf.reshape(decoder_predictions1, (out_length_dec1, -1, output_dim))
    
    #reshaped_outputs1 = decoder_outputs_ta.stack()

with tf.variable_scope("Decoder2") as scope:
    print("\n DECODER 2 Feature Extraction: Output = predicting future values \n")   
    
    # Decoder: inputs    
    dummy_zero_input2 = [tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO_dummy_zero_input")]
    
    # Decoder: expected outputs
    expected_sparse_output2 = [
        tf.placeholder(tf.float32, shape=(None, output_dim), name="expected_sparse_output_".format(t))
        for t in range(out_length_dec2)
    ]

    w_out2 = tf.Variable(tf.random_normal([out_hidden_dim, output_dim]))
    b_out2 = tf.Variable(tf.random_normal([output_dim]))
    
    # The decoder, also multi-layered
    if high_seqlen_flag == 1:
        cell_dec2 = tf.contrib.rnn.MultiRNNCell([LSTMCell(hidden_dim_list_dec[i]) for i in range(layers_stacked_count_dec)])
    else:
        cell_dec2 = tf.contrib.rnn.MultiRNNCell([GRUCell(hidden_dim_list_dec[i]) for i in range(layers_stacked_count_dec)])
       
    def loop_fn2(time, cell_output, cell_state, loop_state):
        emit_output = cell_output  # == None for time == 0
        if cell_output is None:
            next_cell_state = encoder_final_state            
            next_input = dummy_zero_input2[0] 
        else:  
            next_cell_state = cell_state
            next_input = tf.add(tf.matmul(cell_output, w_out2), b_out2)          

        elements_finished = (time >= out_length_dec2)
        next_loop_state = None
        return (elements_finished, next_input, next_cell_state,
                emit_output, next_loop_state)        
    
    decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(cell_dec2, loop_fn2)
    decoder_outputs2 = decoder_outputs_ta.stack()
    
    decoder_max_steps, decoder_batch_size, decoder_dim2 = tf.unstack(tf.shape(decoder_outputs2))
    decoder_outputs_flat2 = tf.reshape(decoder_outputs2, (-1, decoder_dim2))
    
    decoder_predictions2 = tf.add(tf.matmul(decoder_outputs_flat2, w_out2), b_out2)
    reshaped_outputs2 = tf.reshape(decoder_predictions2, (out_length_dec2, -1, output_dim))
    
    
# Training loss and optimizer

with tf.variable_scope('Loss'):  
    reg_loss = 0
    for tf_var in tf.trainable_variables():
        if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
            reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
    
    output_loss1 = 0
    output_loss1 += tf.reduce_mean(tf.nn.l2_loss(reshaped_outputs1 - expected_sparse_output1))
        
    output_loss2 = 0
    output_loss2 += tf.reduce_mean(tf.nn.l2_loss(reshaped_outputs2 - expected_sparse_output2))

    # L2 regularization (to avoid overfitting and to have a  better generalization capacity)    
    loss = output_loss1 + output_loss2 + lambda_l2_reg * reg_loss

with tf.variable_scope('Optimizer'): 
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate, decay=lr_decay, momentum=momentum)
    train_op = optimizer.minimize(loss)


# ## Training of the neural net
Xtr, Ytr = generate_x_y_data(data)

Xts, Yts = generate_x_y_data(data)

print("\n\n", Xtr.shape, Ytr.shape, Xts.shape, Yts.shape, "\n")

def train_batch(Xtr, Ytr):    
    feed_dict = {enc_inp[t]: Xtr[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output1[t]: Xtr[
                     t] for t in range(len(expected_sparse_output1))})
    feed_dict.update({expected_sparse_output2[t]: Ytr[
                     t] for t in range(len(expected_sparse_output2))})
    _, loss_t, Featur = sess.run([train_op, loss, Feats], feed_dict)
    return loss_t, Featur


def test_batch(Xts, Yts):       
    feed_dict = {enc_inp[t]: Xts[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output1[t]: Xts[
                     t] for t in range(len(expected_sparse_output1))})
    feed_dict.update({expected_sparse_output2[t]: Yts[
                     t] for t in range(len(expected_sparse_output2))})
    loss_t = sess.run([loss], feed_dict)
    return loss_t[0]


# Training
train_losses = []
test_losses = []

sess.run(tf.global_variables_initializer())
cnt=0
early_stop = 480

tmp_loss = []
Features = []
for t in range(nb_iters + 1):
    train_loss, Featr = train_batch(Xtr, Ytr)
    train_losses.append(train_loss)
    Features.append(Featr)
    
    test_loss = test_batch(Xts, Yts)
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

print("Final train loss: {}, \tTEST loss: {}".format(train_loss, test_loss))



# Test
nb_predictions = 1

X, Y = generate_x_y_data(data)
print(X.shape, Y.shape)

nb_predictions = X.shape[1]
#nb_predictions = 48

feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
#feed_dict.update({dropout : 0.2})
outputs1 = np.array(sess.run([reshaped_outputs1], feed_dict)[0])
outputs2 = np.array(sess.run([reshaped_outputs2], feed_dict)[0])



print("\n Decoder 1: feature Extraction based on Autoencoder approach, Output = Input")
################## decoder 1
print("MAPE : = ", mape(X, outputs1))    
print("SMAPE : = ", smape(X, outputs1)) 


################### decoder 2
print(" \nDecoder 2: feature Extraction based on Forecasting next n values, Output = Next n values")
print("MAPE : = ", mape(Y, outputs2))    
print("SMAPE : = ", smape(Y, outputs2)) 



print("New feature dimensions of entire timeseries data", Features[-1].shape)
print("Every time series of length: ", data.shape[0], "is reduced to dimension: ", Features[-1][1].shape)


print("---Time Taken to run = %s seconds ---" % (time.time() - start_time))





