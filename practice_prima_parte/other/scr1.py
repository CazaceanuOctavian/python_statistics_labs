import pandas as pd
import numpy as np

df_agricultura = pd.read_csv('input2/Agricultura.csv')
df_populatie = pd.read_csv('input2/PopulatieLocalitati.csv')
df_ronuts = pd.read_csv('input2/RO_NUTS.csv')

#CERINTA 1 -- SUMA PE SUBACTIVITATI PER LOCALITATE
df_agricultura_sum = pd.DataFrame()
df_agricultura_sum['Siruta'] = df_agricultura['Siruta']
df_agricultura_sum['Localitate'] = df_agricultura['Localitate']
df_agricultura_sum['Sum_subactivitati'] = df_agricultura.iloc[:, 2:].sum(axis=1)
df_agricultura_sum.sort_values('Sum_subactivitati', inplace=True, ascending=False)
df_agricultura_sum.set_index('Siruta', inplace=True)
df_agricultura_sum.to_csv('output2/Cerinta1.csv')

#Cerinta 2 -- CIFRA DE AFACERI PER LOCUITOR PER LOCALITATE
df_agricultura_pop_merge = df_agricultura.merge(df_populatie[['Siruta', 'Populatie']], left_on='Siruta', right_on='Siruta', how='inner')
df_cif_afaceri_per_loc = pd.DataFrame()
df_cif_afaceri_per_loc['Siruta'] = df_agricultura_pop_merge['Siruta']
df_cif_afaceri_per_loc['Localitate'] = df_agricultura_pop_merge['Localitate']
df_cif_afaceri_per_loc['Cif_afaceri_per_locuitor'] = df_agricultura_pop_merge.iloc[:, 2:-1].sum(axis=1)/df_agricultura_pop_merge['Populatie']
df_cif_afaceri_per_loc.to_csv('output2/Cerinta2.csv')

#Cerinta 3 -- CIFRA DE AFACERI LA NIVEL DE JUDET
df_agricultura_jud_merge = df_agricultura.merge(df_populatie[['Judet', 'Siruta']], left_on='Siruta', right_on='Siruta', how='inner')
df_agricultura_sum_jud = df_agricultura_jud_merge.groupby('Judet')[df_agricultura_jud_merge.iloc[:, 2:-1].columns].sum()
df_agricultura_sum_jud.to_csv('output2/Cerinta3.csv')

#Cerinta 4 -- MEDIA ARITMETICA PONDERATA LA NIVEL DE REGIUNE (PONDEREA ESTE POPULATIA PER JUDET)
def calc_medie_ponderata(group:pd.DataFrame):
    return pd.Series(np.average(group.iloc[:, :-1], weights=group['Populatie'], axis=0))

df_agricultura_reg =  df_agricultura.merge(df_populatie[['Siruta', 'Populatie', 'Judet']], left_on='Siruta', right_on='Siruta', how='inner')
df_agricultura_reg = df_agricultura_reg.merge(df_ronuts[['IndicativJudet', 'Regiune']], left_on='Judet', right_on='IndicativJudet', how='inner')
df_agricultura_reg.drop('IndicativJudet', axis=1, inplace=True)
df_agricultura_reg.drop('Judet', inplace=True, axis=1)

df_agric_reg_grouped = df_agricultura_reg.groupby('Regiune')[df_agricultura_reg.iloc[:, 2:].columns].apply(calc_medie_ponderata, include_groups=False)
df_agric_reg_grouped.to_csv('output2/Cerinta4.csv')

#CERINTA 5 -- REGIUNILE SI SUMA VEITURILOR  LA NIVEL DE REGIUNE PENTRU REGIUNILE CARE AU SUMA VENITURILOR MAI MARE DECAT MEDIA PER REGIUNE LA NIVEL DE TARA
def calc_sum_reg(group:pd.DataFrame):
    sum = 0
    for column in group.columns:
        sum += group[column].sum()

    return sum

df_agricultura_reg_sum =  df_agricultura.merge(df_populatie[['Siruta', 'Populatie', 'Judet']], left_on='Siruta', right_on='Siruta', how='inner')
df_agricultura_reg_sum = df_agricultura_reg_sum.merge(df_ronuts[['IndicativJudet', 'Regiune']], left_on='Judet', right_on='IndicativJudet', how='inner')
df_agricultura_reg_sum.drop('IndicativJudet', axis=1, inplace=True)
df_agricultura_reg_sum.drop('Judet', inplace=True, axis=1)

df_sum_tot = df_agricultura_reg_sum.groupby('Regiune')[df_agricultura_reg_sum.iloc[:, 2:-1].columns].apply(calc_sum_reg, include_groups=False)
df_todf_sum_tot = pd.DataFrame(df_sum_tot)
avg_national = df_todf_sum_tot[0].sum()/df_todf_sum_tot[0].count()

mask = df_todf_sum_tot[0] > avg_national

df_todf_sum_tot[mask].to_csv('output2/Cerinta5.csv')

#SKLEARN functia --> pca pentru functia def acp

