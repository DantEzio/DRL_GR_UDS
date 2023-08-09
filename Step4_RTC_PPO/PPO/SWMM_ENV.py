# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 22:41:39 2022

@author: chong

SWMM environment
can be used for any inp file
established based pyswmm
"""
import os
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE']="1"
import numpy as np
import pandas as pd
#import pyswmm.toolkitapi as tkai

from swmm_api.input_file import read_inp_file
from pyswmm import Simulation,Links,Nodes,RainGages,SystemStats
from swmm_api.input_file.sections.others import TimeseriesData
from swmm_api.input_file.section_labels import TIMESERIES
import GI_setting as gs

import matplotlib.pyplot as plt
import datetime
import yaml
import shutil 

class SWMM_ENV:
    #can be used for every SWMM inp
    def __init__(self,params):
        '''
        params: a dictionary with input
        orf: original file of swmm inp
        control_asset: list of contorl objective, pumps' name
        advance_seconds: simulation time interval
        flood_nodes: selected node for flooding checking
        '''
        self.params = params
        self.config = yaml.load(open(self.params['parm']+".yaml"), yaml.FullLoader)
        #self.t=[]
    
    def reset(self,raine,rainw,i,trainlog,testid):
        #分东西的情况raine与rainw为两场降雨，不分的情况两者为同一个
        if trainlog:
            root='_teminp'
        else:
            root='_temtestinp'
        shutil.copyfile('./SWMM/'+self.params['orf']+'.inp', './'+root+'/'+testid+'/'+self.params['orf']+str(i)+'.inp')
        inp = read_inp_file('./'+root+'/'+testid+'/'+self.params['orf']+str(i)+'.inp')
        # gs.add_GI(inp)
        inp[TIMESERIES]['rainfalle']=TimeseriesData('rainfalle',raine)
        inp[TIMESERIES]['rainfallw']=TimeseriesData('rainfallw',rainw)
        inp.write_file('./'+root+'/'+testid+'/'+self.params['orf']+str(i)+'_GI_rain.inp')
        self.sim=Simulation('./'+root+'/'+testid+'/'+self.params['orf']+str(i)+'_GI_rain.inp')
        self.sim.start()
        
        #模拟一步
        if self.params['advance_seconds'] is None:
            self.sim._model.swmm_step()
        else:
            self.sim._model.swmm_stride(self.params['advance_seconds'])
        
        #记录总体cso和flooding
        self.CSO,self.flooding=0,0
        
        #obtain states and reward term by yaml (config)
        nodes = Nodes(self.sim)
        links = Links(self.sim)
        rgs = RainGages(self.sim)
        states = []
        for _temp in self.config["states"]:
            if _temp[1] == 'depthN':
                states.append(nodes[_temp[0]].depth)
            elif _temp[1] == 'flow':
                states.append(links[_temp[0]].flow)
            elif _temp[1] == 'inflow':
                states.append(nodes[_temp[0]].total_inflow)
            else:
                states.append(rgs[_temp[0]].rainfall)

        #计算flooding duration用于计算间隔时间
        self.tn={}
        for n in nodes:
            self.tn[n.nodeid]=n.statistics['flooding_duration']
        #计算干管节点的流量，用于计算DRes指标
        self.Qn={}
        for _temp in self.config['reward_targets']:
            if _temp[0] == 'DRes': 
                self.Qn[_temp[1]] = nodes[_temp[1]].total_inflow

        self.allnode,self.alllink=[],[]
        for l in links:
            self.alllink.append(l.linkid)

        for n in nodes:
            self.allnode.append(n.nodeid)

        return states
        
    def step(self,action):
        #获取模拟结果
        nodes = Nodes(self.sim)
        links = Links(self.sim)
        rgs = RainGages(self.sim)
        sys = SystemStats(self.sim)
        #obtain states and reward term by yaml (config)
        states = []
        for _temp in self.config["states"]:
            if _temp[1] == 'depthN':
                states.append(nodes[_temp[0]].depth)
            elif _temp[1] == 'flow':
                states.append(links[_temp[0]].flow)
            elif _temp[1] == 'inflow':
                states.append(nodes[_temp[0]].total_inflow)
            else:
                states.append(rgs[_temp[0]].rainfall)
            
        
        #设置控制
        if self.params['train']:
            for item,a in zip(self.config['action_assets'],action):
                links[item].target_setting = a
        else:
            for item,a in zip(self.config['action_assets'],action):
                links[item].target_setting = a
        
        
        #模拟一步
        if self.params['advance_seconds'] is None:
            time = self.sim._model.swmm_step()
        else:
            time = self.sim._model.swmm_stride(self.params['advance_seconds'])
        #self.t.append(self.sim._model.getCurrentSimulationTime())
        Interval_t=self.params['advance_seconds']/60 #minutes
        done = False if time > 0 else True
        
        #获取reward
        #reward分为2个部分，Res和DRes，其中Res通过计算SevC和SevF获取，DRes通过计算干管节点流量获取
        #均需要通过流量乘时间间隔计算
        nodes = Nodes(self.sim)
        links = Links(self.sim)
        rgs = RainGages(self.sim)
        sys = SystemStats(self.sim)
        flooding,CSO,CSOtem=0,0,0
        Qtw=0
        tem_sevC,tem_sevF=0,0
        tem_dres=0
        for _temp in self.config['reward_targets']:
            if _temp[0] == 'sevF':
                for n in nodes:
                    deltt=(n.statistics['flooding_duration']-self.tn[n.nodeid])*60 #minutes
                    tem_sevF+=n.flooding*deltt/Interval_t
                    #更新tn
                    self.tn[n.nodeid]=n.statistics['flooding_duration']
                
            elif _temp[0] == 'sevC':
                tem_sevC += nodes[_temp[1]].total_inflow
            
            elif _temp[0] == 'DRes': 
                tem_dres += np.square(nodes[_temp[1]].total_inflow/5-self.Qn[_temp[1]]/5) #minutes
                #更新Qn
                self.Qn[_temp[1]]=nodes[_temp[1]].total_inflow
            
            elif _temp[0] == 'Flooding':
                flooding += sys.routing_stats[_temp[1]]-self.flooding
                self.flooding = sys.routing_stats[_temp[1]]
            
            elif  _temp[0] == 'CSO':
                CSOtem += nodes[_temp[1]].cumulative_inflow
                
        Qtw = (sys.routing_stats['dry_weather_inflow']
                  +sys.routing_stats['wet_weather_inflow']
                  +sys.routing_stats['groundwater_inflow']
                  +sys.routing_stats['II_inflow'])
        
        pumps_flow=[]
        for it in ['CC-R1','CC-R2','CC-S1','CC-S2','JK-R1','JK-R2','JK-S']:
            pumps_flow.append(links[it].flow)

        Res_tn = 1/(1+self.params['kc']*((tem_sevC)/(Qtw))+self.params['kf']*((tem_sevF)/(Qtw)))
        DRes_tn = 1/(1+(tem_dres)/(Qtw))
        CSO = CSOtem - self.CSO
        self.CSO = CSOtem
        #FC = -(flooding+CSO)/Qtw
        rewards = Res_tn
        #rewards = -(flooding+CSO)/inflow
        #rewards = np.exp(-(flooding/inflow)**2/0.01) + np.exp(-(CSO/inflow)**2/0.01)
        
        #获取干管流量结果
        mainpip=['WS02006251WS02006249','WS02006235WS02006234','WS02006229WS02006228']
        main_flow=[]
        for pip in mainpip:
            main_flow.append(links[pip].flow)


        #获取所有管道的capacity/depth，用于证明利用率的问题
        captem=0
        for pipe in self.alllink:
            captem+=links[pipe].depth#capacity
        
        results={'flooding':self.flooding,'CSO':self.CSO,'Res_tn':Res_tn,
                 'DRes_tn':DRes_tn,
                 'main_flow':main_flow,
                 'pump_flow':pumps_flow,
                 'capacity':captem,
                 }

        #降雨结束检测
        if done:
            self.sim._model.swmm_end()
            self.sim._model.swmm_close()

        return states,rewards,results,done
        
            
if __name__=='__main__':
    params={
            'orf':'chaohu_GI_init',
            'parm':'./states_yaml/chaohu',
            'GI':True,
            'advance_seconds':300,
            'kf':0.5,
            'kc':0.5,
           }
    env=SWMM_ENV(params)
    
    #prepare rainfall
    raindata = np.load('./training_rainfall/training_raindata.npy').tolist()
            
    env.reset(raindata[0],0,True)
    
    done = False
    states,actions,rewards=[],[],[]
    F,C,R,DR=[],[],[],[]
    t=0
    while not done:
        action = [0.5 for _ in range(len(env.config['action_assets']))]
        s,r,flooding,cso,Res_tn,DRes_tn,done = env.step(action)
        states.append(s)
        actions.append(action)
        rewards.append(r)
        F.append(flooding)
        C.append(cso)
        R.append(Res_tn)
        DR.append(DRes_tn)
        t+=1
        
    plt.plot(states)
    plt.plot(rewards)
    plt.plot(F)
    plt.plot(C)
    plt.plot(R)
    plt.plot(DR)
