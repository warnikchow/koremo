import numpy as np
import librosa

mlen=500

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.01
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

import sys
version = sys.version_info

if version[0]<3:
  print('Version upgrade required!')

from keras.models import load_model
if version[1]<=5:
  mse_crs = load_model('model/total_s_rmse_cnn_rnnself-20-0.9645-f0.9644.hdf5')
elif version[1]==6:
  mse_crs = load_model('model/model_for_6.h5')
  mse_crs.load_weights('model/total_s_rmse_cnn_rnnself-20-0.9645-f0.9644.hdf5')

def make_data(filename):
  data=np.zeros((1,mlen,128))
  data_conv=np.zeros((1,mlen,128,1))
  data_rmse=np.zeros((1,mlen,1))
  data_s_rmse=np.zeros((1,mlen,129))
  data_s_rmse_conv=np.zeros((1,mlen,129,1))
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
    data[0][:,:]=S[-mlen:,:]
    data_conv[0][:,:,0]=S[-mlen:,:]
    data_rmse[0][:,0]=rmse[-mlen:,0]
    data_s_rmse[0][:,0]=rmse[-mlen:,0]
    data_s_rmse[0][:,1:]=S[-mlen:,:]
    data_s_rmse_conv[0][:,0,0]=rmse[-mlen:,0]
    data_s_rmse_conv[0][:,1:,0]=S[-mlen:,:]
  else:
    data[0][-len(S):,:]=S
    data_conv[0][-len(S):,:,0]=S
    data_rmse[0][-len(S):,0]=np.transpose(rmse)
    data_s_rmse[0][-len(S):,0]=np.transpose(rmse)
    data_s_rmse[0][-len(S):,1:]=S
    data_s_rmse_conv[0][-len(S):,0,0]=np.transpose(rmse)
    data_s_rmse_conv[0][-len(S):,1:,0]=S
  return data,data_conv,data_rmse,data_s_rmse,data_s_rmse_conv

def pred_emo(filename):
  data,data_conv,data_rmse,data_s_rmse,data_s_rmse_conv =make_data(filename)
  att_source= np.zeros((1,64))
  z = mse_crs.predict([data_s_rmse_conv,data_s_rmse,att_source])[0]
  y = np.argmax(z)
  if y==0:
    print(">> Angry")
  if y==1:
    print(">> Fear")
  if y==2:
    print(">> Joy")
  if y==3:
    print(">> Normal")
  if y==4:
    print(">> Sad")
  return y


