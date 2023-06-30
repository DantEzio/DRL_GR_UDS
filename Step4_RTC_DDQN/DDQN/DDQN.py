# -*- coding: utf-8 -*-

"""
Created on Wed Aug 17 20:10:48 2022

@author: chong
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from keras import backend as K

def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)

class DDQN:
    #Dueling DQN
    def __init__(self,params):
        #tf.compat.v1.disable_eager_execution()
        self.params=params
        self.action_table=pd.read_csv('./DQN_action_table.csv').values[:,1:]
        
        # Initialize the actor and the critic as keras models
        self.observation_input = keras.Input(shape=(self.params['state_dim'],), dtype=tf.float32, name='sc_input')
        self.encoding = mlp(self.observation_input, self.params['encoding_layer'], tf.tanh, None)
        self.V = mlp(self.encoding, self.params['value_layer']+[1], tf.tanh, None)
        self.A = mlp(self.encoding, self.params['advantage_layer']+[self.params['action_dim']], tf.tanh, None)
        self.Q = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keepdims=True))
        
        self.model = keras.Model(inputs=self.observation_input, outputs=self.Q)
        self.target_model = keras.Model(inputs=self.observation_input, outputs=self.Q)
        self.optimizer = Adam(learning_rate=self.params['learning_rate'])
        
        #self.model.compile(loss='mse', optimizer=Adam(self.params['learning_rate']))
                                                
    def load_model(self,file):
        self.model.load_weights(file+'/ddqn.h5')
        self.target_model.load_weights(file+'/ddqn.h5')

# 
def sample_action(state,model,train_log):
    #input state, output action
    if train_log:
        #epsilon greedy
        pa = np.random.uniform()
        if model.params['epsilon'] < pa:
            action_value = model.model.predict(state,verbose=0)
            action = np.argmax(action_value)
        else:
            action = np.random.randint(model.params['action_dim'])
    else:
        action_value = model.model.predict(state,verbose=0)
        action = np.argmax(action_value)

    return action

def train_value(observation_buffer, action_buffer, reward_buffer, observation_next_buffer, model):
    with tf.GradientTape() as tape:
        y = model.model.predict(observation_buffer,verbose=0)
        q = model.target_model.predict(observation_next_buffer,verbose=0)
        
        for i in range(observation_buffer.shape[0]):
            target = reward_buffer[i] + model.params['gamma'] * np.amax(q[i])
            y[i][action_buffer[i]] = target
        
        loss = tf.reduce_mean((model.model(observation_buffer)-y)**2)
        grads = tape.gradient(loss, model.model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.model.trainable_variables))
        model.target_model.set_weights(model.model.get_weights())
    