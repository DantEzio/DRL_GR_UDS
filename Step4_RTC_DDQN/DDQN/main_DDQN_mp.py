# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:39:01 2022

@author: chong
"""

import numpy as np
import SWMM_ENV as SWMM_ENV
import DDQN as DDQN
import Buffer
import Rainfall_data as RD
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()

env_params={
    'orf':'chaohu_GI_RTC',
    'parm':'./states_yaml/chaohu',
    'GI':True,
    'advance_seconds':300,
    'kf':1e3,
    'kc':1e3,
}
env=SWMM_ENV.SWMM_ENV(env_params)

raindatae = np.load('./training_rainfall/training_raindata.npy').tolist()
raindataw = np.load('./training_rainfall/training_raindata.npy').tolist()
temeast = np.load('./test_rainfall/east.npy').tolist()
temwest = np.load('./test_rainfall/west.npy').tolist()
#raindata2top = np.load('./training_rainfall/2top.npy').tolist()
#raindata3top = np.load('./training_rainfall/3top.npy').tolist()
rr = np.load('./test_rainfall/real.npy').tolist()

agent_params={
    'state_dim':len(env.config['states']),
    'action_dim':2**len(env.config['action_assets']),

    'encoding_layer':[50,50],
    'value_layer':[50],
    'advantage_layer':[50],
    'num_rain':40,

    'train_iterations':2,
    'training_step':100,
    'gamma':0.1,
    'epsilon':0.7,
    'ep_min':1e-100,
    'ep_decay':0.5,
    'learning_rate':0.001
}


Train=True
init_train=False
train_round='2'

model = DDQN.DDQN(agent_params)
if init_train:
    model.model.save_weights('./model/ddqn.h5')    
model.load_model('./model/')
print('model done')

###############################################################################
# Train
###############################################################################
    
def interact(i,ep):   
    env=SWMM_ENV.SWMM_ENV(env_params)
    tem_model = DDQN.DDQN(agent_params)
    tem_model.load_model('./model/')
    tem_model.params['epsilon']=ep
    s,a,r,s_ = [],[],[],[]
    observation, episode_return, episode_length = env.reset(raindatae[i],raindataw[i],i,True), 0, 0
    
    done = False
    while not done:
        # Get the action, and take one step in the environment
        observation = np.array(observation).reshape(1, -1)
        action = DDQN.sample_action(observation,tem_model,True)
        #print(action,'********************8')
        at = tem_model.action_table[int(action)-1].tolist()
        observation_new, reward, flooding,CSO,Res_tn,DRes_tn,main_flow,done,_ = env.step(at)
        episode_return += reward
        episode_length += 1

        # Store obs, act, rew
        # buffer.store(observation, action, reward, value_t, logprobability_t)
        s.append(observation)
        a.append(action)
        r.append(reward)
        s_.append(observation_new)
        
        # Update the observation
        observation = observation_new
    # Finish trajectory if reached to a terminal state
    last_value = 0 if done else tem_model.predict(observation.reshape(1, -1))  
    return s,a,r,s_,last_value,episode_return,episode_length

if Train:
    #tf.config.experimental_run_functions_eagerly(True)

    # main training process   
    history = {'episode': [], 'Batch_reward': [], 'Episode_reward': [], 'Loss': []}
    
    # Iterate over the number of epochs
    for epoch in range(model.params['training_step']):
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0
        
        # Initialize the buffer
        buffer = Buffer.Buffer(model.params['state_dim'], int(len(raindatae[0])*model.params['num_rain']))
        print('Buffer got')
        
        # Iterate over the steps of each epoch
        # Parallel method in joblib
        res = Parallel(n_jobs=10)(delayed(interact)(i,model.params['epsilon']) for i in range(model.params['num_rain'])) 
        print('Sample over')
        
        for i in range(model.params['num_rain']):
            #s, a, r, vt, lo, lastvalue in buffer
            for o,a,r,o_ in zip(res[i][0],res[i][1],res[i][2],res[i][3]):
                buffer.store(o,a,r,o_)
            buffer.finish_trajectory(res[i][4])
            sum_return += res[i][5]
            sum_length += res[i][6]
            num_episodes += 1
        
        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            observation_next_buffer,
            reward_buffer,
            advantage_buffer,
        ) = buffer.get()

        # Update the policy and implement early stopping using KL divergence
        for _ in range(model.params['train_iterations']):
            DDQN.train_value(observation_buffer, action_buffer, reward_buffer, observation_next_buffer, model)
            
        model.model.save_weights('./model/ddqn.h5')
        # log training results
        history['episode'].append(epoch)
        history['Episode_reward'].append(sum_return)
        # reduce the epsilon egreedy and save training log
        if model.params['epsilon'] >= model.params['ep_min'] and epoch % 5 == 0:
            model.params['epsilon'] *= model.params['ep_decay']
        
        # Print mean return and length for each epoch
        print(
            f" Epoch: {epoch + 1}. Return: {sum_return}. Mean Length: {sum_length / num_episodes}"
        )
        
        np.save('./Results//Train'+train_round+'.npy',history)
    
    # plot
    plt.figure()
    plt.plot(history['Episode_reward'])
    plt.savefig('./Results//Train'+train_round+'.tif')

   
###############################################################################
# end Train
###############################################################################


# test PPO agent
def test(model,raine,rainw,i):
    # simulation on given rainfall
    env=SWMM_ENV.SWMM_ENV(env_params)
    test_history = {'time': [], 'state': [], 'action': [], 'reward': [],
                    'F': [], 'C': [], 'RES': [], 'DRES': [], 'mainpip': [], 'pump_flow': []}
    observation = env.reset(raine,rainw,i,False)
    done, t= False, 0
    test_history['time'].append(t)
    test_history['state'].append(observation)
    while not done:
        observation = np.array(observation).reshape(1, -1)
        action = DDQN.sample_action(observation,model,False)
        at=model.action_table[int(action)-1].tolist()
        observation_new,reward,F,C,Res_tn,DRes_tn,main_flow,done,pf = env.step(at)
        observation = observation_new
        t +=1
        
        test_history['time'].append(t)
        test_history['state'].append(observation)
        test_history['action'].append(action)
        test_history['reward'].append(reward)
        test_history['F'].append(F)
        test_history['C'].append(C)
        test_history['RES'].append(Res_tn)
        test_history['DRES'].append(DRes_tn)
        test_history['mainpip'].append(main_flow)
        test_history['pump_flow'].append(pf)
    
    return test_history


# RSH test
rainfalle = np.load('./test_rainfall/east.npy').tolist()
rainfallw = np.load('./test_rainfall/west.npy').tolist()
model.load_model('./model/')
for i in range(len(rainfalle)):
    print(i)
    test_his = test(model,rainfalle[i],rainfallw[i],i)
    np.save('./Results/RSH/'+str(i)+'.npy',test_his)


# Real rainfall
rainfall = np.load('./test_rainfall/real.npy').tolist()
model.load_model('./model/')
for i in range(len(rainfall)):
    print(i)
    test_his = test(model,rainfall[i],rainfall[i],i)
    np.save('./Results/RR/'+str(i)+'.npy',test_his)

'''
# RN test
rainfall = np.load('./test_rainfall/2top.npy').tolist()
model.load_model('./model/')
for i in range(len(rainfall)):
    print(i)
    test_his = test(model,rainfall[i],rainfall[i],i)
    np.save('./Results/RN/2top'+str(i)+'.npy',test_his)

rainfall = np.load('./test_rainfall/3top.npy').tolist()
model.load_model('./model/')
for i in range(len(rainfall)):
    print(i)
    test_his = test(model,rainfall[i],rainfall[i],i)
    np.save('./Results/RN/3top'+str(i)+'.npy',test_his)

'''
print('Done')