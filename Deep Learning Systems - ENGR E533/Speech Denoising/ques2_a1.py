
# coding: utf-8

# In[135]:


from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
import numpy as np
import random
import matplotlib.pyplot as mat
import librosa


# In[136]:


epsilon = 1e-3
savePath = "speechDenoising.ckpt/"
test_1 = 'test_x_01.wav'
test_2 = 'test_x_02.wav'
output_1 = 'output_1.wav'
output_2 = 'output_2.wav'


# In[137]:


def add_new_layer(inputs, in_size, out_size, activation_function=None,):
    std_deviation = math.sqrt(2/in_size)
    Weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=std_deviation))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs


# In[138]:


x = tf.placeholder(tf.float32, shape=(None, 513))
y = tf.placeholder(tf.float32, shape=(None, 513))


# In[139]:


input_layer = add_new_layer(x, 513, 1024, activation_function=tf.nn.relu)
hidden_layer_1 = add_new_layer(input_layer, 1024, 1024, activation_function=tf.nn.relu)
hidden_layer_2 = add_new_layer(hidden_layer_1, 1024, 1024, activation_function=tf.nn.relu)
prediction = add_new_layer(hidden_layer_2, 1024, 513, activation_function=tf.nn.relu)

loss=tf.losses.mean_squared_error(labels=y,predictions=prediction)

train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


# In[140]:


sess = tf.Session()
init = tf.global_variables_initializer()
saver=tf.train.Saver()
sess.run(init)


# In[141]:


def trainModel():
    print("TRAINING")
    s, sr = librosa.load('train_clean_male.wav', sr=None)
    S = librosa.stft(s, n_fft=1024, hop_length=512)
    sn, sr = librosa.load('train_dirty_male.wav', sr=None)
    X = librosa.stft(sn, n_fft=1024, hop_length=512)

    for i in range(151):
        batch_x, batch_y = np.transpose(np.abs(X)), np.transpose(np.abs(S))
        sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
        if i % 10 == 0:
            print("Epoch: ",i)
            saver.save(sess, savePath)


# In[142]:


trainModel()


# In[143]:


'''
Testing occurs below 
'''
    
def testingModel(testFile,outputFile):
    print("TESTING")
    print(testFile)
    sn, sr = librosa.load(testFile, sr=None)
    X = librosa.stft(sn, n_fft=1024, hop_length=512)
    T = X
    batch_x = np.transpose(np.abs(X))
    saving_output = sess.run(prediction, feed_dict={x: batch_x})
    saving_output_transpose = np.transpose(saving_output)
    multiplying_factor = np.divide(T, np.abs(T))
    final_answer = np.multiply(multiplying_factor, saving_output_transpose)
    final_answer1 = librosa.istft(final_answer, win_length=1024, hop_length=512)
    librosa.output.write_wav(outputFile, final_answer1, sr)
    print("DONE")
    
    
testingModel(test_1,output_1)
testingModel(test_2,output_2)
sess.close()


# In[133]:





# In[134]:





# In[ ]:




