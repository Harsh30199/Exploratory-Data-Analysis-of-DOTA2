#!/usr/bin/env python
# coding: utf-8

# In[60]:


import pandas as pd
import numpy as np


# In[61]:


df = pd.read_csv('../data/match-details/2016.csv')
df.head()


# In[62]:


df.shape


# In[63]:


df1 = pd.read_csv('../data/drafting/International_2016_drafting.csv')
final = []
df1.head()


# In[64]:


df2 = pd.read_csv('../data/hero-names/hero_name.csv')
hero_name={}
for i in range(len(df2)):
    hero_name[df2.loc[i,'heroid']] = df2.loc[i,'hero']
hero_name


# In[65]:


hero_list = np.array([])
for i in range(1,11):
    hero_list = np.concatenate([hero_list,(df['Player_'+str(i)+'_hero_id']).to_numpy()])


# In[66]:


unique_hero,count = np.unique(hero_list,return_counts=True)
st = 'No of Unique Heroes picked : ' + str(len(unique_hero)) + '\n'
print(st)
final.append(st)

temp = {int(unique_hero[i]) : int(count[i]) for i in range (len(unique_hero))}
hero_count = {hero_name[k]: v for k, v in sorted(temp.items(), key=lambda item: item[1], reverse = True)}

print(hero_count)


# In[67]:


ban_hero_list = np.array([])
for i in range(1,6):
    ban_hero_list = np.concatenate([ban_hero_list,(df1['Ban_T1_'+str(i)]).to_numpy()])
    
    ban_hero_list = np.concatenate([ban_hero_list,(df1['Ban_T2_'+str(i)]).to_numpy()])


# In[68]:


unique_ban_hero,ban_count = np.unique(ban_hero_list,return_counts=True)
st = 'No of Unique Heroes Banned : '+ str(len(unique_ban_hero)) + '\n'
print(st)
final.append(st)



temp_ban = {int(unique_ban_hero[i]) : int(ban_count[i]) for i in range (len(unique_ban_hero)) }
ban_hero_count = {hero_name[k]: v for k, v in sorted(temp_ban.items(), key=lambda item: item[1], reverse = True)}

(ban_hero_count)


# In[69]:


df2 = df.copy(deep=True)
df2['Total Kills'] = df2['Radiant_score'] + df2['Dire_score']
df2 = df2[['Match_id','Radiant_score','Dire_score','Total Kills']]
most_kills = df2.iloc[df2['Total Kills'].argmax()]
st = 'Most Kills in a match are ' + str(most_kills[3]) + ' in Match id '  + str(most_kills[0]) + '\n' 
print(st)
final.append(st)


# In[70]:


df3 = df[['Match_id','Duration']]
longest_duration = df3.iloc[df3['Duration'].argmax()]
hours = int(longest_duration[1]/3600)
minutes = int((longest_duration[1] - hours*3600)/60)
seconds = int(longest_duration[1] - (hours*3600 + minutes*60))

st = 'Longest match in Tournament is ' + str(hours) + ' H ' + str(minutes) + ' M '  + str(seconds) + ' S for Match id ' + str(longest_duration[0]) + '\n'
final.append(st)
print(st)


# In[71]:


df3 = df[['Match_id','Duration']]
longest_duration = df3.iloc[df3['Duration'].argmin()]
hours = int(longest_duration[1]/3600)
minutes = int((longest_duration[1] - hours*3600)/60)
seconds = int(longest_duration[1] - (hours*3600 + minutes*60))

st = 'Shortest match in Tournament is ' + str(hours) + ' H ' + str(minutes) + ' M ' + str(seconds) + ' S for Match id ' + str(longest_duration[0]) + '\n'
final.append(st)
print(st)


# In[72]:


hero_kill_temp=[]
for i in range(1,11):
    hero_kill_temp.append(list(df.iloc[df['Player_'+str(i)+'_kills'].argmax(),[0,22*(i-1)+2,22*(i-1)+11]]))

hero_kill_sorted = [[a,hero_name[b],c] for a,b,c in sorted(hero_kill_temp, key=lambda item: item[2], reverse = True)]


st = 'Match id : '+str(hero_kill_sorted[0][0]) +' Hero name: ' +str(hero_kill_sorted[0][1]) + ' Total kills '+ str(hero_kill_sorted[0][2]) + '\n'
final.append(st)
print(st) 


# In[73]:


hero_death_temp=[]
for i in range(1,11):
    hero_death_temp.append(list(df.iloc[df['Player_'+str(i)+'_deaths'].argmax(),[0,22*(i-1)+2,22*(i-1)+12]]))

hero_death_sorted = [[a,hero_name[b],c] for a,b,c in sorted(hero_death_temp, key=lambda item: item[2], reverse = True)]


st = 'Match id : '+str(hero_death_sorted[0][0]) +' Hero name: ' +str(hero_death_sorted[0][1]) + ' Total deaths '+ str(hero_death_sorted[0][2]) + '\n'
final.append(st)
print(st)


# In[74]:


hero_gpm_temp=[]
for i in range(1,11):
    hero_gpm_temp.append(list(df.iloc[df['Player_'+str(i)+'_gold_per_min'].argmax(),[0,22*(i-1)+2,22*(i-1)+16]]))

hero_gpm_sorted = [[a,hero_name[b],c] for a,b,c in sorted(hero_gpm_temp, key=lambda item: item[2], reverse = True)]

st = 'Match id : '+str(hero_gpm_sorted[0][0]) +' Hero name: ' +str(hero_gpm_sorted[0][1]) + ' Total GPM '+ str(hero_gpm_sorted[0][2]) + '\n'
final.append(st)
print(st)


# In[77]:


file1 = open('../output/Tournament-Analysis-2016.txt','w')
file1.writelines(final)
file1.close()

