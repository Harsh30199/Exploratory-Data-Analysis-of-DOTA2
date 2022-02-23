#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd 
import numpy 


# In[18]:

hero_name= pd.read_csv('../data/hero-names/hero_name.csv')
hero = list(hero_name['heroid'].unique())
tournamensts = ['2021','2019','2018','2017','2016']


# In[19]:


for p in tournamensts: 
    hero_name= pd.read_csv('../data/hero-names/hero_name.csv')
    data = pd.read_csv('../data/match-details/' +p + '.csv')
    darft = pd.read_csv('../data/drafting/International_'+ p +'_drafting.csv')

    hero_count = {}
    hero_ban = {}
    hero_contested = {}
    hero_Winrate = {}
    hero_kill = {}
    hero_xpm_avg = {}
    hero_lasthit_avg = {}


    for i in hero:
        hero_count[i] = 0
        hero_ban[i] = 0
        hero_contested[i] = 0
        hero_Winrate[i] = 0
        hero_xpm_avg[i] = 0
        hero_kill[i] = 0
        hero_lasthit_avg[i] = 0

    a = darft.columns
    new_col = []
    for i in a:
        if 'Ban' in i:
            new_col.append(i)
    
    for j in hero:
        for i in new_col:
            hero_ban[j] = hero_ban[j] + len(darft[darft[i]==j])


    for i in hero_count:
        hero_count[i] = len(darft[darft['Pick_T1_1']==i]) + len(darft[darft['Pick_T1_2']==i]) + len(darft[darft['Pick_T1_3']==i])  + len(darft[darft['Pick_T1_4']==i]) + len(darft[darft['Pick_T1_5']==i]) + len(darft[darft['Pick_T2_1']==i]) + len(darft[darft['Pick_T2_2']==i]) + len(darft[darft['Pick_T2_3']==i]) + len(darft[darft['Pick_T2_4']==i]) + len(darft[darft['Pick_T2_5']==i])
        hero_contested[i] = hero_count[i] + hero_ban[i]


    radiant = darft[darft['result']=='Radiant']
    dire = darft[darft['result']=='Dire']
    for i in hero_count:
        hero_Winrate[i] = len(radiant[radiant['Pick_T1_1']==i]) + len(radiant[radiant['Pick_T1_2']==i]) + len(radiant[radiant['Pick_T1_3']==i]) + len(radiant[radiant['Pick_T1_4']==i]) + len(radiant[radiant['Pick_T1_5']==i]) + len(dire[dire['Pick_T2_1']==i]) +len(dire[dire['Pick_T2_2']==i]) + len(dire[dire['Pick_T2_3']==i]) + len(dire[dire['Pick_T2_4']==i]) + len(dire[dire['Pick_T2_5']==i])     
        if hero_count[i]!=0:
            hero_Winrate[i] = round(hero_Winrate[i] / hero_count[i],4)
        else:
            hero_Winrate[i] = 0

    for j in hero:
        for i in range(1,11):
            heroid = 'Player_' + str(i) + '_hero_id'
            kills = 'Player_' + str(i) + '_kills'
            temp = data[[heroid,kills]]
            hero_kill[j] = temp[temp[heroid]==j][kills].sum() + hero_kill[j]
        if hero_count[j]!=0:
            hero_kill[j] = round(hero_kill[j]/hero_count[j],4)
        else:
            hero_kill[j] = 0

    for j in hero:
        for i in range(1,11):
            heroid = 'Player_' + str(i) + '_hero_id'
            kills = 'Player_' + str(i) + '_xp_per_min'
            temp = data[[heroid,kills]]
            hero_xpm_avg[j] = temp[temp[heroid]==j][kills].sum() + hero_xpm_avg[j]
        if hero_count[j]!=0:
            hero_xpm_avg[j] = round(hero_xpm_avg[j] / hero_count[j],4)
        else:
            hero_xpm_avg[j] = 0

    for j in hero:
        for i in range(1,11):
            heroid = 'Player_' + str(i) + '_hero_id'
            last = 'Player_' + str(i) + '_last_hits'
            temp = data[[heroid,last]]
            hero_lasthit_avg[j] = temp[temp[heroid]==j][last].sum() + hero_lasthit_avg[j]
        if hero_count[j]!=0:
            hero_lasthit_avg[j] = round(hero_lasthit_avg[j]/hero_count[j],4)
        else:
            hero_lasthit_avg[j] = 0

    final  = []
    for i in hero:
        final.append([i,hero_count[i],hero_ban[i],hero_contested[i],hero_Winrate[i],hero_kill[i],hero_xpm_avg[i],hero_lasthit_avg[i]])
    final = pd.DataFrame(final,columns=['heroid','hero_pick','hero_ban','hero_contested','hero_Winrate','hero_kill_avg','hero_xpm_avg','hero_lasthit_avg'])
    name = '../output/hero-analysis-' + p + '.csv'

    hero_name=hero_name.merge(final,on='heroid')
    hero_name.drop('heroid',inplace=True,axis='columns')
    hero_name.to_csv(name,index=False)

