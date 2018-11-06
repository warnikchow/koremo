import numpy as np
import librosa

mlen=500

def make_data(fname,fnum,mlen):
 data             = np.zeros((fnum,mlen,128))
 data_conv        = np.zeros((fnum,mlen,128,1))
 data_rmse        = np.zeros((fnum,mlen,1))
 data_s_rmse      = np.zeros((fnum,mlen,129))
 data_s_rmse_conv = np.zeros((fnum,mlen,129,1))
 for i in range(1,fnum+1):
  if i%100==0:
    print(i)
  if i <10:
    num = '000'+str(i)
  if 10<=i<100:
    num = '00'+str(i)
  if 100<=i<1000:
    num = '0'+str(i)
  if 1000<=i<10000:
    num = str(i)
  filename = fname+num+'.wav'
  y, sr = librosa.load(filename)
  D = np.abs(librosa.stft(y))**2
  ss, phase = librosa.magphase(librosa.stft(y))
  rmse = librosa.feature.rmse(S=ss)
  rmse = rmse/np.max(rmse)
  rmse = np.transpose(rmse)
  S = librosa.feature.melspectrogram(S=D)
  S = np.transpose(S)
  Srmse = np.multiply(rmse,S)
  if len(S)>=mlen:
    data[i-1][:,:]=S[-mlen:,:]
    data_conv[i-1][:,:,0]=S[-mlen:,:]
    data_rmse[i-1][:,0]=rmse[-mlen:,0]
    data_s_rmse[i-1][:,0]=rmse[-mlen:,0]
    data_s_rmse[i-1][:,1:]=S[-mlen:,:]
    data_s_rmse_conv[i-1][:,0,0]=rmse[-mlen:,0]
    data_s_rmse_conv[i-1][:,1:,0]=S[-mlen:,:]
  else:
    data[i-1][-len(S):,:]=S
    data_conv[i-1][-len(S):,:,0]=S
    data_rmse[i-1][-len(S):,0]=np.transpose(rmse)
    data_s_rmse[i-1][-len(S):,0]=np.transpose(rmse)
    data_s_rmse[i-1][-len(S):,1:]=S
    data_s_rmse_conv[i-1][-len(S):,0,0]=np.transpose(rmse)
    data_s_rmse_conv[i-1][-len(S):,1:,0]=S
 return data,data_conv,data_rmse,data_s_rmse,data_s_rmse_conv

fang_rnn,fang_conv,fang_rmse,fang_s_rmse,fang_s_rmse_conv=make_data('data/female/ANG/wav/0',1000,mlen)
mang_rnn,mang_conv,mang_rmse,mang_s_rmse,mang_s_rmse_conv=make_data('data/male/ANG/wav/0',800,mlen)

ffea_rnn,ffea_conv,ffea_rmse,ffea_s_rmse,ffea_s_rmse_conv=make_data('data/female/FEA/wav/0',500,mlen)
mfea_rnn,mfea_conv,mfea_rmse,mfea_s_rmse,mfea_s_rmse_conv=make_data('data/male/FEA/wav/0',550,mlen)

fjoy_rnn,fjoy_conv,fjoy_rmse,fjoy_s_rmse,fjoy_s_rmse_conv=make_data('data/female/JOY/wav/0',1000,mlen)
mjoy_rnn,mjoy_conv,mjoy_rmse,mjoy_s_rmse,mjoy_s_rmse_conv=make_data('data/male/JOY/wav/0',1000,mlen)

fnor_rnn,fnor_conv,fnor_rmse,fnor_s_rmse,fnor_s_rmse_conv=make_data('data/female/NOR/wav/FY_NOR_skj_0',2700,mlen)
mnor_rnn,mnor_conv,mnor_rmse,mnor_s_rmse,mnor_s_rmse_conv=make_data('data/male/NOR/wav/0',2699,mlen)

fsad_rnn,fsad_conv,fsad_rmse,fsad_s_rmse,fsad_s_rmse_conv=make_data('data/female/SAD/wav/0',500,mlen)
msad_rnn,msad_conv,msad_rmse,msad_s_rmse,msad_s_rmse_conv=make_data('data/male/SAD/wav/0',800,mlen)

label_5_train = np.concatenate([np.zeros(int(0.9*1800)), np.ones(int(0.9*1050)), 2*np.ones(int(0.9*2000)), 3*np.ones(int(np.ceil(0.9*5399))), 4*np.ones(int(0.9*1300))])
label_5_test  = np.concatenate([np.zeros(int(0.1*1800)), np.ones(int(0.1*1050)), 2*np.ones(int(0.1*2000)), 3*np.ones(int(np.floor(0.1*5399))), 4*np.ones(int(0.1*1300))])

total_rnn_train = np.concatenate([fang_rnn[:int(0.9*1000)],mang_rnn[:int(0.9*800)],ffea_rnn[:int(0.9*500)],mfea_rnn[:int(0.9*550)],fjoy_rnn[:int(0.9*1000)],mjoy_rnn[:int(0.9*1000)],fnor_rnn[:int(0.9*2700)],mnor_rnn[:int(np.ceil(0.9*2699))],fsad_rnn[:int(0.9*500)],msad_rnn[:int(0.9*800)]])
total_rnn_test = np.concatenate([fang_rnn[int(0.9*1000):],mang_rnn[int(0.9*800):],ffea_rnn[int(0.9*500):],mfea_rnn[int(0.9*550):],fjoy_rnn[int(0.9*1000):],mjoy_rnn[int(0.9*1000):],fnor_rnn[int(0.9*2700):],mnor_rnn[int(np.ceil(0.9*2699)):],fsad_rnn[int(0.9*500):],msad_rnn[int(0.9*800):]])

total_conv_train = np.concatenate([fang_conv[:int(0.9*1000)],mang_conv[:int(0.9*800)],ffea_conv[:int(0.9*500)],mfea_conv[:int(0.9*550)],fjoy_conv[:int(0.9*1000)],mjoy_conv[:int(0.9*1000)],fnor_conv[:int(0.9*2700)],mnor_conv[:int(np.ceil(0.9*2699))],fsad_conv[:int(0.9*500)],msad_conv[:int(0.9*800)]])
total_conv_test = np.concatenate([fang_conv[int(0.9*1000):],mang_conv[int(0.9*800):],ffea_conv[int(0.9*500):],mfea_conv[int(0.9*550):],fjoy_conv[int(0.9*1000):],mjoy_conv[int(0.9*1000):],fnor_conv[int(0.9*2700):],mnor_conv[int(np.ceil(0.9*2699)):],fsad_conv[int(0.9*500):],msad_conv[int(0.9*800):]])

total_rmse_train = np.concatenate([fang_rmse[:int(0.9*1000)],mang_rmse[:int(0.9*800)],ffea_rmse[:int(0.9*500)],mfea_rmse[:int(0.9*550)],fjoy_rmse[:int(0.9*1000)],mjoy_rmse[:int(0.9*1000)],fnor_rmse[:int(0.9*2700)],mnor_rmse[:int(np.ceil(0.9*2699))],fsad_rmse[:int(0.9*500)],msad_rmse[:int(0.9*800)]])
total_rmse_test = np.concatenate([fang_rmse[int(0.9*1000):],mang_rmse[int(0.9*800):],ffea_rmse[int(0.9*500):],mfea_rmse[int(0.9*550):],fjoy_rmse[int(0.9*1000):],mjoy_rmse[int(0.9*1000):],fnor_rmse[int(0.9*2700):],mnor_rmse[int(np.ceil(0.9*2699)):],fsad_rmse[int(0.9*500):],msad_rmse[int(0.9*800):]])

total_s_rmse_train = np.concatenate([fang_s_rmse[:int(0.9*1000)],mang_s_rmse[:int(0.9*800)],ffea_s_rmse[:int(0.9*500)],mfea_s_rmse[:int(0.9*550)],fjoy_s_rmse[:int(0.9*1000)],mjoy_s_rmse[:int(0.9*1000)],fnor_s_rmse[:int(0.9*2700)],mnor_s_rmse[:int(np.ceil(0.9*2699))],fsad_s_rmse[:int(0.9*500)],msad_s_rmse[:int(0.9*800)]])
total_s_rmse_test = np.concatenate([fang_s_rmse[int(0.9*1000):],mang_s_rmse[int(0.9*800):],ffea_s_rmse[int(0.9*500):],mfea_s_rmse[int(0.9*550):],fjoy_s_rmse[int(0.9*1000):],mjoy_s_rmse[int(0.9*1000):],fnor_s_rmse[int(0.9*2700):],mnor_s_rmse[int(np.ceil(0.9*2699)):],fsad_s_rmse[int(0.9*500):],msad_s_rmse[int(0.9*800):]])

total_s_rmse_conv_train = np.concatenate([fang_s_rmse_conv[:int(0.9*1000)],mang_s_rmse_conv[:int(0.9*800)],ffea_s_rmse_conv[:int(0.9*500)],mfea_s_rmse_conv[:int(0.9*550)],fjoy_s_rmse_conv[:int(0.9*1000)],mjoy_s_rmse_conv[:int(0.9*1000)],fnor_s_rmse_conv[:int(0.9*2700)],mnor_s_rmse_conv[:int(np.ceil(0.9*2699))],fsad_s_rmse_conv[:int(0.9*500)],msad_s_rmse_conv[:int(0.9*800)]])
total_s_rmse_conv_test = np.concatenate([fang_s_rmse_conv[int(0.9*1000):],mang_s_rmse_conv[int(0.9*800):],ffea_s_rmse_conv[int(0.9*500):],mfea_s_rmse_conv[int(0.9*550):],fjoy_s_rmse_conv[int(0.9*1000):],mjoy_s_rmse_conv[int(0.9*1000):],fnor_s_rmse_conv[int(0.9*2700):],mnor_s_rmse_conv[int(np.ceil(0.9*2699)):],fsad_s_rmse_conv[int(0.9*500):],msad_s_rmse_conv[int(0.9*800):]])

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from keras.models import Sequential, Model
from keras.layers import Input, Embedding, LSTM, GRU, SimpleRNN, Dense, Lambda
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import keras.layers as layers

from keras import optimizers
adam_half = optimizers.Adam(lr=0.0005)
adam_half_2 = optimizers.Adam(lr=0.0002)

from keras.preprocessing import sequence
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.embeddings import Embedding

from random import random
from numpy import array
from numpy import cumsum
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization

##### f1 score ftn.
from keras.callbacks import Callback
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from keras.layers import Input, Embedding, LSTM, Dense, Lambda
from keras.models import Model
import keras.backend as K
from keras.layers.normalization import BatchNormalization

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced', np.unique(label_5_train), label_5_train)

class Metricsf1macro_forself2(Callback):
 def on_train_begin(self, logs={}):
  self.val_f1s = []
  self.val_recalls = []
  self.val_precisions = []
  self.val_f1s_w = []
  self.val_recalls_w = []
  self.val_precisions_w = []
 def on_epoch_end(self, epoch, logs={}):
  if len(self.validation_data)>2:
   val_predict = np.asarray(self.model.predict([self.validation_data[0],self.validation_data[1],self.validation_data[2]]))
   val_predict = np.argmax(val_predict,axis=1)
   val_targ = self.validation_data[3]
  else:
   val_predict = np.asarray(self.model.predict(self.validation_data[0]))
   val_predict = np.argmax(val_predict,axis=1)
   val_targ = self.validation_data[1]
  _val_f1 = metrics.f1_score(val_targ, val_predict, average="macro")
  _val_f1_w = metrics.f1_score(val_targ, val_predict, average="weighted")
  _val_recall = metrics.recall_score(val_targ, val_predict, average="macro")
  _val_recall_w = metrics.recall_score(val_targ, val_predict, average="weighted")
  _val_precision = metrics.precision_score(val_targ, val_predict, average="macro")
  _val_precision_w = metrics.precision_score(val_targ, val_predict, average="weighted")
  self.val_f1s.append(_val_f1)
  self.val_recalls.append(_val_recall)
  self.val_precisions.append(_val_precision)
  self.val_f1s_w.append(_val_f1_w)
  self.val_recalls_w.append(_val_recall_w)
  self.val_precisions_w.append(_val_precision_w)
  print("— val_f1: %f — val_precision: %f — val_recall: %f"%(_val_f1, _val_precision, _val_recall))
  print("— val_f1_w: %f — val_precision_w: %f — val_recall_w: %f"%(_val_f1_w, _val_precision_w, _val_recall_w))

metricsf1macro_self2 = Metricsf1macro_forself2()

def validate_cnn_rnnself(conv,rnn,train_y,test_conv,test_rnn,test_y,maxlen,hidden_dim,filename):
    cnn_input = Input(shape=(maxlen,len(rnn[0][0]),1), dtype='float32')
    cnn_layer = layers.Conv2D(32,(5,5),activation='relu')(cnn_input)
    cnn_layer = BatchNormalization()(cnn_layer)
    cnn_layer = layers.MaxPooling2D((2,2))(cnn_layer)
    cnn_layer = Dropout(0.3)(cnn_layer)
    cnn_layer = layers.Conv2D(64,(5,5),activation='relu')(cnn_layer)
    cnn_layer = BatchNormalization()(cnn_layer)
    cnn_layer = layers.MaxPooling2D((2,2))(cnn_layer)
    cnn_layer = Dropout(0.3)(cnn_layer)
    cnn_layer = layers.Conv2D(128,(5,5),activation='relu')(cnn_layer)
    cnn_layer = BatchNormalization()(cnn_layer)
    cnn_layer = layers.MaxPooling2D((2,2))(cnn_layer)
    cnn_layer = Dropout(0.3)(cnn_layer)
    cnn_layer = layers.Conv2D(32,(3,3),activation='relu')(cnn_layer)
    cnn_layer = BatchNormalization()(cnn_layer)
    cnn_layer = layers.MaxPooling2D((2,1))(cnn_layer)
    cnn_layer = layers.Conv2D(32,(3,3),activation='relu')(cnn_layer)
    cnn_layer = BatchNormalization()(cnn_layer)
    cnn_layer = layers.MaxPooling2D((2,1))(cnn_layer)
    cnn_layer = layers.Flatten()(cnn_layer)
    cnn_output= Dense(hidden_dim, activation='relu')(cnn_layer)
    rnn_input = Input(shape=(maxlen,len(rnn[0][0])), dtype='float32')
    rnn_layer = Bidirectional(LSTM(64,return_sequences=True))(rnn_input)
    rnn_att   = Dense(hidden_dim, activation='tanh')(rnn_layer)
    att_source= np.zeros((len(rnn),hidden_dim))
    att_test  = np.zeros((len(test_rnn),hidden_dim))
    att_input = Input(shape=(hidden_dim,),dtype='float32')
    att_vec   = Dense(hidden_dim, activation='relu')(att_input)
    att_vec   = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([att_vec,rnn_att])
    att_vec   = Dense(len(rnn[0]),activation='softmax')(att_vec)
    att_vec   = layers.Reshape((len(rnn[0]),1))(att_vec)
    rnn_output= keras.layers.multiply([att_vec,rnn_layer])
    rnn_output= Lambda(lambda x: K.sum(x, axis=1))(rnn_output)
    output    = layers.concatenate([cnn_output,rnn_output])
    output    = Dense(hidden_dim, activation='relu')(output)
    output    = Dropout(0.3)(output)
    output    = Dense(hidden_dim, activation='relu')(output)
    output    = Dropout(0.3)(output)
    main_output = Dense(5,activation='softmax')(output)
    model = Sequential()
    model = Model(inputs=[cnn_input,rnn_input,att_input], outputs=[main_output])
    model.compile(optimizer=adam_half, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    callbacks_list = [metricsf1macro_self2,checkpoint]
    model.summary()
    model.fit([conv,rnn,att_source],train_y,validation_data=([test_conv,test_rnn,att_test],test_y),epochs=20,batch_size=16,callbacks=callbacks_list)

from keras.layers import Input, Embedding, LSTM, Dense, Lambda
from keras.models import Model
import keras.backend as K
from keras.layers.normalization import BatchNormalization
import keras

validate_cnn_rnnself(total_s_rmse_conv_train,total_s_rmse_train,label_5_train,total_s_rmse_conv_test,total_s_rmse_test,label_5_test,500,64,'model/total_s_rmse_cnn_rnnself')
