#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


yearList = [2016, 2017, 2018, 2019, 2021]


# In[3]:


def highLow(data__, button, descend, avg):
    avg__ = []
    tempavg__ = []
    retList = []
    for pid in pidList:
        for i_d in idList:
            buttonval, buttoncount = 0, 0
            filt1 = data__[pid] == i_d
            count = data__[filt1].count()[0]
            if count == 0:
                continue
            else:
                if pid[8] == '_':
                    buttonval += data__[filt1][str(pid[:9])+button].sum()
                    buttoncount += data__[filt1][str(pid[:9])+button].count()
                    tempavg__.append([i_d, buttonval, buttoncount])
                elif pid[8] == '0':
                    buttonval += data__[filt1][str(pid[:10])+button].sum()
                    buttoncount += data__[filt1][str(pid[:10])+button].count()
                    tempavg__.append([i_d, buttonval, buttoncount])
    for i_d in idList:
        tempone = 0
        temptwo = 0
        for item in tempavg__:
            if i_d == item[0]:
                tempone += item[1]
                temptwo += item[2]
        avg__.append([i_d, tempone, temptwo])
    for item in avg__:
        if avg:
            retList.append([item[0], item[1]/item[2]])
        else:
            retList.append([item[0], item[1]])
    retList.sort(key = lambda x : x[1], reverse = descend)
    del tempavg__
    del avg__
    return retList


# In[4]:


attri_list = ['highest_kill_avg_pid', 'highest_kill_avg', 'most_kills_pid', 'most_kills', 'lowest_death_avg_pid',
              'lowest_death_avg', 'highest_assists_avg_pid', 'highest_assists_avg', 'most_assists_pid',
              'most_assists', 'most_last_hits_pid', 'most_last_hits', 'most_last_hit_avg_pid', 'most_last_hit_avg',
              'most_gpm_pid', 'most_gpm',  'highest_gpm_avg_pid', 'highest_gpm_avg']


# In[5]:


attri_list = ['Player_id', 'highest_kill_avg', 'most_kills','lowest_death_avg', 'highest_assists_avg',
              'most_assists', 'most_last_hits', 'most_gpm', 'highest_gpm_avg']


# In[6]:


for year in yearList:
    data__ = pd.read_csv('../data/match-details/' +str(year)+'.csv')
    columnlist = data__.columns.tolist()
    
    pidList = []
    for itemm in columnlist:
        if itemm.endswith('_playerid'):
            pidList.append(itemm)
            
    idList = []
    for pid in pidList:
        tempList = data__[pid].unique().tolist()
        for item in tempList:
            if item not in idList:
                idList.append(item)
    
    highest_kill_avg = highLow(data__, 'kills', True, True)
    most_kills = highLow(data__, 'kills', True, False)
    lowest_death_avg = highLow(data__, 'deaths', False, True)
    highest_assists_avg = highLow(data__, 'assists', True, True)
    most_assists = highLow(data__, 'assists', True, False)
    most_last_hits = highLow(data__, 'last_hits', True, False)
    most_last_hit_avg = highLow(data__, 'last_hits', True, True)
    most_gpm = highLow(data__, 'gold_per_min', True, False)
    highest_gpm_avg = highLow(data__, 'gold_per_min', True, True)
    
    df = pd.DataFrame(highest_kill_avg, columns = ['Player_id', 'avg_kills'])
    df = df.merge(pd.DataFrame(most_kills, columns = ['Player_id', 'total_kills']))
    df = df.merge(pd.DataFrame(lowest_death_avg, columns = ['Player_id', 'lowest_death_avg']))
    df = df.merge(pd.DataFrame(highest_assists_avg, columns = ['Player_id', 'highest_assists_avg']))
    df = df.merge(pd.DataFrame(most_assists, columns = ['Player_id', 'total_assists']))
    df = df.merge(pd.DataFrame(most_last_hits, columns = ['Player_id', 'total_last_hits']))
    df = df.merge(pd.DataFrame(most_last_hit_avg, columns = ['Player_id', 'most_last_hit_avg']))
    df = df.merge(pd.DataFrame(most_gpm, columns = ['Player_id', 'total_gpm']))
    df = df.merge(pd.DataFrame(highest_gpm_avg, columns = ['Player_id', 'highest_gpm_avg']))
    
    
#     csv_data = pd.DataFrame(columns = attri_list)
   
#     for i in range(len(highest_kill_avg)):
#         csv_data.loc[len(csv_data.index)] = [highest_kill_avg[i][0], highest_kill_avg[i][1],
#                                             most_kills[i][0], most_kills[i][1],
#                                             lowest_death_avg[i][0], lowest_death_avg[i][1],
#                                             highest_assists_avg[i][0], highest_assists_avg[i][1],
#                                             most_assists[i][0], most_assists[i][1],
#                                             most_last_hits[i][0], most_last_hits[i][1],
#                                             most_last_hit_avg[i][0], most_last_hit_avg[i][1],
#                                             most_gpm[i][0], most_gpm[i][1], highest_gpm_avg[i][0],
#                                             highest_gpm_avg[i][1]]
    
    df.to_csv('../output/player-analysis'+ str(year) +'.csv', index = False)


# In[ ]:




