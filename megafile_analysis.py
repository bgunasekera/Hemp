import pandas as pd
import numpy as np
import pingouin as pg
import statsmodels.formula.api as smf


#%%
df = pd.read_spss("C:/Users/bjgun/OneDrive - King's College London/HEMP FEEDBACK/megafile.sav")
df.info() 

#%%
df.pivot_table(values='SubjectID', columns='drug', aggfunc=lambda x: len(x.unique()))

df["SubjectID"] = df["SubjectID"].apply(str)
df["SubjectID"]= df["SubjectID"].str.split(".", n = 1, expand = True)
df['id_upd'] = df[['SubjectID', 'drug']].agg('-'.join, axis=1)

df['IsHit'] = df['IsHit'].fillna('No')
df["IsHit"] = df["IsHit"].str.replace('yes','0')
df["IsHit"] = df["IsHit"].str.replace('No','1').astype(int)

df["Salient"] = df["Salient"].str.replace('Salient','0')
df["Salient"] = df["Salient"].str.replace('control','1').astype(int)

df["False"] = df["False"].str.replace('False','1')
df["False"] = df["False"].str.replace('True','0')
#%%
reward = df.copy()
reward = reward.groupby(['id_upd']).Reward.agg(['sum'])
reward = reward.reset_index()

zero = [20]*48
zero = pd.DataFrame({'col':zero})

reward = pd.merge(reward, zero, how='outer', left_index=True, right_index=True)

reward["end_reward"]= reward["sum"] + reward["col"]


reward[['SubjectID', 'drug']] = reward['id_upd'].str.split('-', 1, expand=True)

#%%
cbd = reward.query('drug == "cbd"')['end_reward'].copy()
plb = reward.query('drug == "placebo"')['end_reward'].copy()
hc = reward.query('drug == "HC"')['end_reward'].copy()

plb_paired= reward[reward['drug'] == 'placebo'].copy()
plb_paired = plb_paired[plb_paired.SubjectID != '17']
plb_p = plb_paired.query('drug == "placebo"')['end_reward'].copy()


#%%
df_hc_plb = df.copy()
df_hc_plb = df_hc_plb[df_hc_plb['drug'] != 'cbd']

df_plb_cbd = df.copy()
df_plb_cbd = df_plb_cbd[df_plb_cbd['drug'] != 'HC']


df_hc_plb["drug"] = df_hc_plb["drug"].str.replace('placebo','1')
df_hc_plb["drug"] = df_hc_plb["drug"].str.replace('HC','0').astype(int)

df_plb_cbd["drug"] = df_plb_cbd["drug"].str.replace('cbd','1')
df_plb_cbd["drug"] = df_plb_cbd["drug"].str.replace('placebo','0').astype(int)

#%%
df_hc_plb_false= df_hc_plb.copy()
df_hc_plb_false = df_hc_plb_false[df_hc_plb_false['False'].notna()]
df_hc_plb_false.isna().sum() 
df_hc_plb_false = df_hc_plb_false.astype({"False": int})
df_hc_plb_false.rename(columns={'False': 'false_start'}, inplace=True)

df_plb_cbd_false= df_plb_cbd.copy()
df_plb_cbd_false = df_plb_cbd_false[df_plb_cbd_false['False'].notna()]
df_plb_cbd_false.isna().sum() 
df_plb_cbd_false = df_plb_cbd_false.astype({"False": int})
df_plb_cbd_false.rename(columns={'False': 'false_start'}, inplace=True)



#%%
#Total reward £
reward.groupby(['drug']).end_reward.agg(['mean', 'std'])


hc_plb = pg.ttest(plb, hc, paired=False)
cbd_plb = pg.ttest(plb_p, cbd, paired=True)



#%%
#Accuracy (successful hits on target %)
accuracy= df.groupby(['drug', 'IsHit']).size().copy()


accuracy_percentage = accuracy.groupby(level=0).apply(lambda x:
                                                 100 * x / float(x.sum()))
    


m = 'IsHit ~ drug*Salient'
blogistic = smf.logit(formula = str(m), data = df_hc_plb).fit()
print(blogistic.summary()) 

params = blogistic.params
conf = blogistic.conf_int()
conf['Odds Ratio'] = params
conf.columns = ['5%', '95%', 'Odds Ratio']
print(np.exp(conf))

m = 'IsHit ~ drug*Salient'
blogistic = smf.logit(formula = str(m), data = df_plb_cbd).fit()
print(blogistic.summary()) 

params = blogistic.params
conf = blogistic.conf_int()
conf['Odds Ratio'] = params
conf.columns = ['5%', '95%', 'Odds Ratio']
print(np.exp(conf))

    
#%%
#False starts (%)
false= df.groupby(['drug', 'False']).size().copy()


false_percentage = false.groupby(level=0).apply(lambda x:
                                                 100 * x / float(x.sum()))
    
neutral = df[df['CueName'] == 'control'].copy()
salient = df[df['Salient'] == 'Salient'].copy()

neutral= df.groupby(['drug', 'False']).size().copy()


neutral_percentage = neutral.groupby(level=0).apply(lambda x:
                                                 100 * x / float(x.sum()))
    
salient= df.groupby(['drug', 'False']).size().copy()


salient_percentage = false.groupby(level=0).apply(lambda x:
                                                 100 * x / float(x.sum()))


m = 'false_start ~ drug*Salient'
blogistic = smf.logit(formula = str(m), data = df_hc_plb_false).fit()
print(blogistic.summary()) 

params = blogistic.params
conf = blogistic.conf_int()
conf['Odds Ratio'] = params
conf.columns = ['5%', '95%', 'Odds Ratio']
print(np.exp(conf))

m = 'false_start ~ drug*Salient'
blogistic = smf.logit(formula = str(m), data = df_plb_cbd_false).fit()
print(blogistic.summary()) 

params = blogistic.params
conf = blogistic.conf_int()
conf['Odds Ratio'] = params
conf.columns = ['5%', '95%', 'Odds Ratio']
print(np.exp(conf))


#%%
#RT
df_rt = df.copy()
df_rt = df_rt[df_rt.RT >= 100]
df_rt = df_rt[df_rt['RT'].notna()]
df_rt.info()
df_rt['drug'] = df_rt['drug'].astype(object)
df_rt['Salient'] = df_rt['Salient'].astype(object)


df_rt.groupby(['drug']).RT.agg(['mean', 'std'])
neutral = df_rt[df_rt['CueName'] == 'control'] 
neutral.groupby(['drug']).RT.agg(['mean', 'std'])
salient = df_rt[df_rt['Salient'] == 0] 
salient.groupby(['drug']).RT.agg(['mean', 'std'])


hc_plb_RT = df_rt.copy()
hc_plb_RT = hc_plb_RT[hc_plb_RT['drug'] != 'cbd']

plb_cbd_RT = df_rt.copy()
plb_cbd_RT = plb_cbd_RT[plb_cbd_RT['drug'] != 'HC']
plb_cbd_RT = plb_cbd_RT[plb_cbd_RT.SubjectID != '17']

#%%
aov = pg.anova(dv='RT', between=['drug', 'Salient'], ss_type=3, data=hc_plb_RT)
pg.print_table(aov)


#%%
aov1 = pg.rm_anova(dv='RT', within=['drug', 'Salient'], subject='SubjectID', data=plb_cbd_RT)
pg.print_table(aov1)











