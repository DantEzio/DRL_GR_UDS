# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:39:01 2022

@author: chong
"""
import numpy as np
import SWMM_ENV
import Buffer
import PPO as PPO
import Rainfall_data as RD
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from swmm_api import read_out_file

tf.compat.v1.reset_default_graph()
env_params={
    'orf':'chaohu_RTC',
    'parm':'./states_yaml/chaohu',
    'GI':True,
    'advance_seconds':300,
    'kf':0.5,
    'kc':0.5,
    'train':True,
}
env=SWMM_ENV.SWMM_ENV(env_params)

raindatae = np.load('./training_rainfall/training_raindata.npy').tolist()
raindataw = np.load('./training_rainfall/training_raindata.npy').tolist()
temeast = np.load('./test_rainfall/RSH/east.npy').tolist()
temwest = np.load('./test_rainfall/RSH/west.npy').tolist()
#raindata2top = np.load('./training_rainfall/2top.npy').tolist()
#raindata3top = np.load('./training_rainfall/3top.npy').tolist()
rr = np.load('./test_rainfall/RealRain/real.npy').tolist()

raindatae = temeast + rr + raindatae
raindataw = temwest + rr + raindataw 

agent_params={
    'state_dim':len(env.config['states']),
    'action_dim':int(2**len(env.config['action_assets'])),
    'actornet_layer':[50,50,50],
    'criticnet_layer':[50,50,50],
    
    'bound_low':0,
    'bound_high':1,
    
    'clip_ratio':0.01,
    'target_kl':0.01,
    'lam':0.01,
    
    'policy_learning_rate':0.01,
    'value_learning_rate':0.01,
    'train_policy_iterations':2,
    'train_value_iterations':2,
    
    'num_rain':30,
    
    'training_step':100,
    'gamma':0.1,
    'epsilon':1,#1递减，最终1e-100
    'ep_min':1e-100,
    'ep_decay':0.99
}

Train=False
init_train=False
train_round='1'

model = PPO.PPO(agent_params)
if init_train:
    model.critic.save_weights('./model/PPOcritic.h5')
    model.actor.save_weights('./model/PPOactor.h5')
model.load_model('./model/')


###############################################################################
# Train
###############################################################################
    
def interact(i,ep):   
    env=SWMM_ENV.SWMM_ENV(env_params)
    tem_model = PPO.PPO(agent_params)
    tem_model.load_model('./model/')
    tem_model.params['epsilon']=ep
    s,a,r,vt,lo = [],[],[],[],[]
    observation, episode_return, episode_length = env.reset(raindatae[i],raindataw[i],i,True,'train'), 0, 0
    
    done = False
    while not done:
        # Get the logits, action, and take one step in the environment
        observation = np.array(observation).reshape(1, -1)
        logits, action = PPO.sample_action(observation,tem_model,True)
        at = tem_model.action_table[int(action[0].numpy())].tolist()
        observation_new, reward, results, done = env.step(at)
        flooding,CSO=results['flooding'],results['CSO']
        Res_tn,DRes_tn=results['Res_tn'],results['DRes_tn']
        main_flow = results['main_flow']
        capacity = results['capacity']

        episode_return += reward
        episode_length += 1

        # Get the value and log-probability of the action
        value_t = tem_model.critic(observation)
        logprobability_t = PPO.logprobabilities(logits, action, tem_model.params['action_dim'])
        
        # Store obs, act, rew, v_t, logp_pi_t
        # buffer.store(observation, action, reward, value_t, logprobability_t)
        s.append(observation)
        a.append(action)
        r.append(reward)
        vt.append(value_t)
        lo.append(logprobability_t)
        
        # Update the observation
        observation = observation_new
    # Finish trajectory if reached to a terminal state
    last_value = 0 if done else tem_model.critic(observation.reshape(1, -1))
    return s,a,r,vt,lo,last_value,episode_return,episode_length

if Train:
    tf.config.experimental_run_functions_eagerly(True)

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
        
        # Iterate over the steps of each epoch
        # Parallel method in joblib
        res = Parallel(n_jobs=10)(delayed(interact)(i,model.params['epsilon']) for i in range(model.params['num_rain'])) 
        
        for i in range(model.params['num_rain']):
            #s, a, r, vt, lo, lastvalue in buffer
            for o,a,r,vt,lo in zip(res[i][0],res[i][1],res[i][2],res[i][3],res[i][4]):
                buffer.store(o,a,r,vt,lo)
            buffer.finish_trajectory(res[i][5])
            sum_return += res[i][6]
            sum_length += res[i][7]
            num_episodes += 1
        
        
        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = buffer.get()
    
        # Update the policy and implement early stopping using KL divergence
        for _ in range(model.params['train_policy_iterations']):
            kl = PPO.train_policy(observation_buffer, action_buffer, logprobability_buffer, advantage_buffer, model)
            #if kl > 1.5 * target_kl:
                ## Early Stopping
                #break
    
        # Update the value function
        for _ in range(model.params['train_value_iterations']):
            PPO.train_value_function(observation_buffer, return_buffer, model)
        
        #model.critic.save_weights('./model/PPOcritic_'+str(epoch)+'.h5')
        #model.actor.save_weights('./model/PPOactor_'+str(epoch)+'.h5')
        model.critic.save_weights('./model/PPOcritic.h5')
        model.actor.save_weights('./model/PPOactor.h5')
        # log training results
        history['episode'].append(epoch)
        history['Episode_reward'].append(sum_return)
        # reduce the epsilon egreedy and save training log
        if model.params['epsilon'] >= model.params['ep_min'] and epoch % 3 == 0:
            model.params['epsilon'] *= model.params['ep_decay']

        # Print mean return and length for each epoch
        print(
            f" Epoch: {epoch + 1}. Return: {sum_return}. Mean Length: {sum_length / num_episodes}"
        )
        
        np.save('./Results/Train_'+train_round+'.npy',history)
    
    # plot
    plt.figure()
    plt.plot(history['Episode_reward'])
    plt.savefig('./Results/Train_'+train_round+'.tif')

    
###############################################################################
# end Train
###############################################################################
    
env_params={
    'orf':'chaohu_RTC',
    'parm':'./states_yaml/chaohu',
    'GI':True,
    'advance_seconds':30,
    'kf':1,
    'kc':1,
    'train':False,
}
    
# test PPO agent
def test(model,raine,rainw,i,testid):
    # simulation on given rainfall
    env=SWMM_ENV.SWMM_ENV(env_params)
    test_history = {'time': [], 'state': [], 'action': [], 'reward': [],
                    'F': [], 'C': [], 'RES': [], 'DRES': [], 'mainpip': [], 'pump_flow': [],
                    'capacity':[]}
    observation = env.reset(raine,rainw,i,False,testid)
    done, t= False, 0
    test_history['time'].append(t)
    test_history['state'].append(observation)
    while not done:
        observation = np.array(observation).reshape(1, -1)
        logits, action = PPO.sample_action(observation,model,False)
        at=model.action_table[int(action[0].numpy())].tolist()
        for _ in range(int(300/env.params['advance_seconds'])):
            observation_new,reward,results,done = env.step(at)
            if done:
                break
            F,C=results['flooding'],results['CSO']
            Res_tn,DRes_tn=results['Res_tn'],results['DRes_tn']
            pf = results['pump_flow']
            main_flow = results['main_flow']
            capacity = results['capacity']
            t += 1
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
            test_history['capacity'].append(capacity)
            
        observation = observation_new
        
    return test_history

# RSH test
rainfalle = np.load('./test_rainfall/RSH/east.npy').tolist()
rainfallw = np.load('./test_rainfall/RSH/west.npy').tolist()
model.load_model('./model/')
for i in range(len(rainfalle)):
    print(i)
    test_his = test(model,rainfalle[i],rainfallw[i],i,'RSH')
    np.save('./Results/RSH/'+str(i)+'.npy',test_his)


# Real rainfall
rainfalle = np.load('./test_rainfall/RealRain/realw.npy').tolist()
rainfallw = np.load('./test_rainfall/RealRain/realw.npy').tolist()
model.load_model('./model/')
for i in range(len(rainfalle)):
    print(i)
    test_his = test(model,rainfalle[i],rainfallw[i],i,'RR')
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

