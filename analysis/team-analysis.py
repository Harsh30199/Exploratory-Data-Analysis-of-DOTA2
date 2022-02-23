#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd 
import json
import numpy as np


# In[6]:


tournamensts = ['2021','2019','2018','2017','2016']

for p in tournamensts:
    data = pd.read_csv('../data/match-details/' +p + '.csv')

    teams= list(data['Radiant_team_name'].unique())
    teams = [x for x in teams if str(x) != 'nan']
    matches_played ={}
    for i in teams:
        matches_played[i] = 0

    for i in matches_played:
        matches_played[i] = len(data[data['Radiant_team_name']==i]) + len(data[data['Dire_team_name']==i])


    total_death = {}
    for i in teams:
        total_death[i] = 0

    total_death_avg={}
    for i in matches_played:
        total_death[i] = data[data['Radiant_team_name']==i]['Dire_score'].sum() + data[data['Dire_team_name']==i]['Radiant_score'].sum()
        total_death_avg[i] = round(total_death[i]/matches_played[i],4)

    total_kill = {}
    for i in teams:
        total_kill[i] = 0

    total_KDA = {}
    total_kill_avg={}
    for i in matches_played:
        total_kill[i] = data[data['Radiant_team_name']==i]['Radiant_score'].sum() + data[data['Dire_team_name']==i]['Dire_score'].sum()
        total_kill_avg[i] = round(total_kill[i]/matches_played[i],4)
        total_KDA[i]= round(total_kill[i]/total_death[i],4)


    unique = {}
    for i in matches_played:
        radiant = data[data['Radiant_team_name']==i]
        dire = data[data['Dire_team_name']==i]

        p1 = radiant['Player_1_hero_id'].unique()
        p2 = radiant['Player_2_hero_id'].unique()
        p3 = radiant['Player_3_hero_id'].unique()
        p4 = radiant['Player_4_hero_id'].unique()
        p5 = radiant['Player_5_hero_id'].unique()
        p6 = dire['Player_6_hero_id'].unique()
        p7 = dire['Player_7_hero_id'].unique()
        p8 = dire['Player_8_hero_id'].unique()
        p9 = dire['Player_9_hero_id'].unique()
        p10 = dire['Player_10_hero_id'].unique()

        a = np.concatenate([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10])
        unique[i] = len(np.unique(a))

    time_avg = {}

    for i in matches_played:
        radiant = data[data['Radiant_team_name']==i]
        dire = data[data['Dire_team_name']==i]
        time_avg[i]= radiant['Duration'].sum() + dire['Duration'].sum()
        time_avg[i] = time_avg[i]/matches_played[i]

    total_wins = {}
    total_winrate = {}
    for i in teams:
        total_wins[i] = len(data[(data['Radiant_team_name']==i) & (data['Result']=='Radiant')]) + len(data[(data['Dire_team_name']==i) & (data['Result']=='Dire')])
        total_winrate[i] = round(total_wins[i]/matches_played[i],4)

    final = []
    for i in teams:
        final.append([i,matches_played[i],total_kill[i],total_kill_avg[i],total_death[i],total_death_avg[i],total_KDA[i],total_wins[i],total_winrate[i],time_avg[i],unique[i]])

    final = pd.DataFrame(final,columns=['Team_name','Total_matches','Total_kills','Kill_avg','Total_deaths','Total_deaths_avg','KDA','Wins','Winrate','Avg_time','Unique_heros_played'])
    final.to_csv('../output/Team-analysis'+ p + '.csv')


# In[ ]:




