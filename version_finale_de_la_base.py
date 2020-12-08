#!/usr/bin/env python
# coding: utf-8

# # **Plan du projet**
# 
# ### **1.** Importation des données et constitution de la DataFrame
# 1.   Fonctions permettant l'importation des données
# 2.   Base comparateur des territoires en 2017 - INSEE
# 3.   Base salaire net horaire moyen en 2017 - INSEE
# 4.   Base scolarisation en 2017 - INSEE
# 5.   Consolidation des DataFrames
# 6.   Base établissements d'enseignement du premier et second degrés
# 
# 
# 

# In[439]:




import pandas as pd
import numpy


import bs4
import re

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

from functools import reduce


# # **Importation des données et constitution de la DataFrame**
# 

# ## Fonctions permettant l'importation des données
# 

# La première fonction (**csv_zip_to_df**) permet de convertir un **fichier CSV en format ZIP** grâce à son URL et au nom du fichier en un **dataframe**.
# 
# La deuxième fonction (**taux**) permet de **créer taux** correspondant au rapport de 2 autres variables initialement dans la base de données. La variable créée est **arrondie au centième**.

# In[440]:


def csv_zip_to_df (url_file,file_name):
  url = urlopen(url_file)
  zipfile = ZipFile(BytesIO(url.read()))
  dataframe = pd.read_csv(zipfile.open(file_name), header = 0, sep = ';')
  return dataframe

def taux(new_var,nom_var,denom_var,bdd):
  bdd[new_var] = bdd[nom_var] / bdd[denom_var]
  bdd[new_var] = bdd[new_var].round(decimals=3)
  return bdd[new_var]


# ## Base comparateur des territoires en 2017 - INSEE
# Cette base, issue de l'INSEE (https://www.insee.fr/fr/statistiques/2521169) renvoie des **indicateurs économiques** à l'échelle communale. 

# Pour notre étude, nous n'avons pas besoin de l'intégralité des variables présentes sur cette base, nous allons en sélectionner quelques unes : 
# 
# - Population en 2017 (**P17_POP**);
# - Médiane du niveau vie en 2017 (**MED17**);
# - Taux de pauvreté en 2O17 (**TP6017**);
# 
# 
# 
# Et en créer certaines :
# - Densité en 2017 (**DEN17**);;
# - Taux de chômage en 2017 (**T17_CHOM1564**)
# Le taux de chômage et le taux d'activité sont redondants, on ne décide de retenir que le taux de chômage de la liste des covariables que l'on souhaite utiliser dans notre modèle. 
# 

# In[437]:


bdd_ind_eco = csv_zip_to_df("https://www.insee.fr/fr/statistiques/fichier/2521169/base-comparateur-2017_CSV.zip","base_cc_comparateur.CSV")


# In[441]:


for i in bdd_ind_eco.columns:
  print(i)

#Création du taux de chômage et d'activité, et de la densité
taux('T17_ACT1564','P17_ACT1564','P17_POP1564',bdd_ind_eco)
taux('T17_CHOM1564','P17_CHOM1564','P17_POP1564',bdd_ind_eco)
taux('DEN17','P17_POP','SUPERF',bdd_ind_eco)

for i in bdd_ind_eco.columns:
  print(i)

#Sélection de variables et arrondis
bdd_ind_eco = bdd_ind_eco[['CODGEO','P17_POP','DEN17','MED17','P17_EMPLT','T17_CHOM1564', 'TP6017']]

bdd_ind_eco[['T17_CHOM1564']] = bdd_ind_eco[['T17_CHOM1564' ]].round(decimals=3)
bdd_ind_eco[['DEN17']] = bdd_ind_eco[['DEN17']].round(decimals=0)

bdd_ind_eco


# 

# ## Base scolarisation en 2017 - INSEE
# Cette base, issue de l'INSEE (https://www.insee.fr/fr/statistiques/4516086?sommaire=4516089) renvoie des indicateurs sur la **scolarisation par catégories d'âge et la part des diplômés dans la population active** *(personnes non scolarisées de 15 ans ou plus)* **selon le diplôme**, à l'échelle communale. 
# 

# Nous allons sélectionner certaines variables afin de créer les suivantes :
# - Taux de scolarisation entre 2 et 5 ans (**T17_0205**);
# - Taux de scolarisation entre 6 et 10 ans (**T17_0610**);
# - Taux de scolarisation entre 11 et 14 ans (**T17_1114**);
# - Taux de scolarisation entre 15 et 17 ans (**T17_1517**);
# - Taux de scolarisation entre 18 et 24 ans (**T17_1824**);
# - Taux de scolarisation entre 25 et 29 ans (**T17_2529**);
# 
# 
# - Part de la population active n'ayant pas ou peu de diplôme (**T17_NDIP**);
# - Part de la population active dont le diplôme le plus élevé est le BEPC ou le brevet (**T17_BEPC**);
# - Part de la population active dont le diplôme le plus élevé est un CAP ou BEP (**T17_CAPBEP**);
# - Part de la population active dont le diplôme le plus élevé est le BAC (**T17_BAC**);
# - Part des diplômés d'un BAC+2 dans la population active (**T17_SUP2**);
# - Part des diplômés d'un BAC+3 ou BAC+4 dans la population active (**T17_SUP34**);
# - Part des diplômés d'un BAC+5 ou plus dans la population active (**T17_SUP5**).
# 

# In[366]:


bdd_scolar = csv_zip_to_df("https://www.insee.fr/fr/statistiques/fichier/4516086/base-ccc-diplomes-formation-2017.zip","base-cc-diplomes-formation-2017.CSV")


# In[442]:


#Création des taux de scolarisation
taux('T17_0205','P17_SCOL0205','P17_POP0205',bdd_scolar)
taux('T17_0610','P17_SCOL0610','P17_POP0610',bdd_scolar)
taux('T17_1114','P17_SCOL1114','P17_POP1114',bdd_scolar)
taux('T17_1517','P17_SCOL1517','P17_POP1517',bdd_scolar)
taux('T17_1824','P17_SCOL1824','P17_POP1824',bdd_scolar)
taux('T17_2529','P17_SCOL2529','P17_POP2529',bdd_scolar)

#Part de diplômés
taux('T17_NDIP','P17_NSCOL15P_DIPLMIN','P17_NSCOL15P',bdd_scolar)
taux('T17_BEPC','P17_NSCOL15P_BEPC','P17_NSCOL15P',bdd_scolar)
taux('T17_CAPBEP','P17_NSCOL15P_CAPBEP','P17_NSCOL15P',bdd_scolar)
taux('T17_BAC','P17_NSCOL15P_BAC','P17_NSCOL15P',bdd_scolar)
taux('T17_SUP2','P17_NSCOL15P_SUP2','P17_NSCOL15P',bdd_scolar)
taux('T17_SUP34','P17_NSCOL15P_SUP34','P17_NSCOL15P',bdd_scolar)
taux('T17_SUP5','P17_NSCOL15P_SUP5','P17_NSCOL15P',bdd_scolar)


# In[443]:


bdd_scolar['CODGEO']= bdd_scolar['CODGEO'].astype(str).str.zfill(5)
bdd_scolar['DEP']= bdd_scolar['CODGEO'].str[:2]

#Sélection de variables
bdd_scolar["Taux_sup"] = bdd_scolar["T17_BAC"] + bdd_scolar["T17_SUP2"] + bdd_scolar["T17_SUP34"] + bdd_scolar["T17_SUP5"]


bdd_scolar = bdd_scolar[["Taux_sup", 'CODGEO']]


# ## Base établissements d'enseignement du premier et second degrés

# In[8]:


import csv
import io
import urllib.request

#url = "https://data.education.gouv.fr/explore/dataset/fr-en-adresse-et-geolocalisation-etablissements-premier-et-second-degre/download/?format=csv&timezone=Europe/Berlin&lang=fr&use_labels_for_header=true&csv_separator=%3B/fr-en-adresse-et-geolocalisation-etablissements-premier-et-second-degre.csv"
url = "https://raw.github.com/datasets/gdp/master/data/gdp.csv"
webpage = urllib.request.urlopen(url)
datareader = csv.reader(io.TextIOWrapper(webpage))
data = list(datareader)

#for row in datareader:
   # print(row)


# ## Détermination des bases des variables à étudier 
# - bdd_results
# - bdd_ind_eco
# - bdd_academ
# - bdd_cinema
# - bdd_scolar
# 

# In[319]:


url = "https://data.education.gouv.fr/explore/dataset/fr-en-indicateurs-de-resultat-des-lycees-denseignement-general-et-technologique/download/?format=csv&timezone=Europe/Berlin&lang=fr&use_labels_for_header=true&csv_separator=%3B"
bdd_results = pd.read_csv(url, sep =";")


# In[321]:


for i in bdd_results.columns :
  print(i)

print(bdd_results.shape)
bdd_results = bdd_results.rename(columns = {'Code commune' : 'CODGEO'})
#Taux d'élèves présents dans les fillières générales
bdd_results["Taux_gen"] = (bdd_results["Effectif Présents série S"] + bdd_results["Effectif Présents série L"] + bdd_results["Effectif Présents série ES"])/ bdd_results['Effectif Présents Total séries']


# On utilisera la variable 'CODGEO' pour merger les différentes bases, en se basant sur le code qui indique la ville dans laquelle se trouve le lycée.
# 
# 

# 

# In[377]:


col = ['Code Etablissement','CODGEO', 'Secteur Public/Prive', 'Académie', 'Département', 'Taux_Mention_brut_serie_L', 'Taux_Mention_brut_serie_S', 'Taux_Mention_brut_serie_ES', 'Taux Brut de Réussite série L', 'Taux Brut de Réussite série ES', 'Taux Brut de Réussite série S', 'Taux_gen']

bdd_results = bdd_results[col]

bdd_results['DEP']= bdd_results['CODGEO'].str[:2]
print(pd.unique(bdd_results["DEP"]))
print(bdd_results[bdd_results["CODGEO"] == '0'])
#On se rend compte qu'une des observations a 0 pour CODGEO, il doit s'agir d'une erreur de saisie qu'on supprime
bdd_results = bdd_results.drop(bdd_results[ bdd_results['CODGEO'] == '0' ].index)
# Notre variable d'intérêt est le taux de réussite au bac général, on supprime donc les lignes où ces données sont manquantes
bdd_results = bdd_results.dropna(subset=['Taux Brut de Réussite série L'])
bdd_results  = bdd_results.dropna(subset=['Taux Brut de Réussite série S'])
bdd_results  = bdd_results.dropna(subset=['Taux Brut de Réussite série ES'])

#Du même biais, on se débarasse des NAN dans la colonne taux_gen

print(bdd_results["Taux_Mention_brut_serie_ES"].isna().sum()/ len(bdd_results))
#On pensait que le taux de mention brut serait un bon indicateur, mais trop de NA. On le garde pour peut-être l'utiliser plus tard


# In[482]:


bdd_academ= pd.read_excel('https://www.insee.fr/fr/statistiques/fichier/1281332/ip1512.xls', sheet_name= "Figure 2a")
bdd_academ = bdd_academ.iloc[3:32]
bdd_academ.columns = ['Académie','part_fav', 'taux_retard_sixième']
print(bdd_academ["Académie"])


# In[13]:


bdd_cinema = pd.read_excel('https://www.data.gouv.fr/fr/datasets/r/cdb918e7-7f1a-44fc-bf6f-c59d1614ed6d', sheet_name ='2017')
print(bdd_cinema.columns)
bdd_cinema = bdd_cinema.iloc[4:]
bdd_cinema = bdd_cinema.rename(columns = {'Unnamed: 1' : "nom", 'Unnamed: 4' : 'CODGEO'})
bdd_cinema


# In[14]:


liste_cinema = bdd_cinema.groupby(['CODGEO'])['nom'].count()


# In[465]:


#On transforme cette liste en data frame
bdd_cinema = pd.DataFrame(liste_cinema)
bdd_cinema = bdd_cinema.rename(columns = {'nom' : 'nb_cinema_com'})


# In[459]:


bdd_agriculture = pd.read_excel('https://www.insee.fr/fr/statistiques/fichier/2012795/TCRD_073.xlsx')
bdd_agriculture = bdd_agriculture.iloc[3:]
bdd_agriculture["DEP"] = bdd_agriculture["Cheptel présent dans les exploitations agricoles en fin d'année 2019 : comparaisons départementales"]
bdd_agriculture["cheptel"] = bdd_agriculture["Unnamed: 2"] + bdd_agriculture["Unnamed: 3"] + bdd_agriculture["Unnamed: 4"] + bdd_agriculture["Unnamed: 5"]
bdd_agriculture = bdd_agriculture[["DEP", 'cheptel']]
print(bdd_agriculture.head())

bdd_agriculture.drop(bdd_agriculture.loc[bdd_agriculture['DEP']== 'P'].index, inplace=True)
bdd_agriculture.drop(bdd_agriculture.loc[bdd_agriculture['DEP']== 'M'].index, inplace=True)
bdd_agriculture.drop(bdd_agriculture.loc[bdd_agriculture['DEP']== 'F'].index, inplace=True)


# In[84]:


url = urlopen('https://www.insee.fr/fr/statistiques/fichier/2021703/base-cc-tourisme-2020.zip')
zipfile = ZipFile(BytesIO(url.read()))
bdd_tourism = pd.read_excel(zipfile.open('base-cc-tourisme-2020.xlsx'), header = 0, sep = ';').iloc[5:]
bdd_tourism


# In[85]:


bdd_tourism = bdd_tourism[['Chiffres détaillés    -     Tourisme', 'Unnamed: 4']]


# In[86]:


bdd_tourism = bdd_tourism.rename(columns = {'Chiffres détaillés    -     Tourisme' : "CODGEO"})
bdd_tourism = bdd_tourism.rename(columns = {'Unnamed: 4': 'nb_hotel_com'})


# In[64]:


bdd_foncier = pd.read_csv('https://www.data.gouv.fr/fr/datasets/r/58b6b75e-4f15-4efb-adb5-3f7b939fb2d1', sep = ",")
bdd_foncier = bdd_foncier[["INSEE_COM", 'Prixm2']]


# In[65]:


print(bdd_foncier["Prixm2"].isna().sum()/ len(bdd_foncier))
#La variable Prixm2 contient 19% de NA
bdd_foncier = bdd_foncier.rename(columns = {'INSEE_COM': 'CODGEO'})
bdd_foncier


# In[466]:


print(bdd_ind_eco.shape) #CODGEO : ok
print(bdd_cinema.shape) #CODGEO  :ok 
print(bdd_results.shape) #CODGEO  :ok
print(bdd_scolar.shape) #CODGEO   :ok
print(bdd_academ.shape) #Académie 
print(bdd_tourism.shape) #CODGEO  :ok
print(bdd_foncier.shape) #CODGEO  :ok
print(bdd_agriculture.shape) #Département :ok 


# ### Consolidation des data frames retenus 
# 

# In[469]:


liste_df = [bdd_ind_eco, bdd_results, bdd_scolar, bdd_tourism, bdd_foncier]

df = reduce(lambda left,right: pd.merge(left, right, on='CODGEO',left_index=True, right_index=True, how='inner'), liste_df)

df  = df.dropna(subset=['Code Etablissement'])
print(df.shape)
print(df.columns)
#On a perdu 4 observations dans l'opération
#On utilise how = "inner" car on souhaite retenir uniquement les valeurs pour lesquelles on a bdd_results, donc on écrase les occurences des villes qui ne comptent pas de lycée


# In[470]:


df1 = pd.merge(df, bdd_agriculture, on ="DEP", how ="outer")
df1  = df1.dropna(subset=['Code Etablissement'])
print(df1.shape)
print(df1.columns)

print(df1["cheptel"])


# In[472]:


df2 = pd.merge(df1, bdd_cinema, on ="CODGEO", how ="outer")
df2  = df2.dropna(subset=['Code Etablissement'])
print(df2.shape)
print(df2.columns)


# In[535]:


#Pour fusionner la base, les valeurs dans les colonnes Académie doivent être identiques
bdd_academ = bdd_academ.sort_values(by ='Académie', ascending = True)
def up(x):
    return(x.upper())
bdd_academ["Académie"] = bdd_academ["Académie"].apply(up)
bdd_academ['Académie'] = bdd_academ['Académie'].str.replace('Ç', "C")
bdd_academ['Académie'] = bdd_academ['Académie'].str.replace('É', "E")
bdd_academ['Académie'] = bdd_academ['Académie'].replace({'CLERMONT' : 'CLERMONT-FERRAND'})

print(pd.unique(bdd_academ["Académie"]))
print(pd.unique(df2["Académie"]))
print(pd.unique(bdd_academ['Académie'].isin(df2['Académie'])))
#On va donc pouvoir utiliser les données comprises dans la colonne Académie pour fusionner nos bases


# In[538]:


df3 = pd.merge(df2, bdd_academ, on ="Académie", how ="outer")
df2  = df2.dropna(subset=['Code Etablissement'])
print(df3.shape)
print(df3.columns)

print(df3["part_fav"])


#Version finale de la base le 8/12 : df3


# ### PREMIERES OBSERVATIONS DE LA BASE

# In[543]:


#On veut aussi faire une étude détaillée au sein de certains départements, et de certaines académies. Pour ce faire, on cherche à
#déterminer les départements médians et extrêmes. 

import numpy as np
reussite_dep = pd.DataFrame(bdd_results.groupby(['DEP'])['Taux Brut de Réussite série S'].mean())

def quant_result(alpha, echelle):
    a = pd.DataFrame(bdd_results.groupby([echelle])['Taux Brut de Réussite série S'].mean())
    return((a[a["Taux Brut de Réussite série S"] == np.quantile(a, 1 - alpha)]))
#Fonction qui nous donne la médiane et les quantiles de la réussite au bac selon l'échelle voulue
#echelle = "DEP", "Académie"

print(reussite_dep)
reussite_dep.sort_values(by = 'Taux Brut de Réussite série S', ascending= False).head(15)
reussite_dep["DEP"] = pd.to_numeric(reussite_dep.index, errors ="coerce")
#On transforme les valeurs de DEP en numérique pour obtenir un graphe
#On exclut donc les départements de la Corse
print(reussite_dep.columns)

reussite_dep.plot.scatter(x='DEP' , y='Taux Brut de Réussite série S', title= "taux de réussite au bac série S par département")
plt.ylim(80, 100)
plt.axhline(y= quant_result(1,'DEP').values, color='gray',linestyle='--' )
plt.axhline(y= quant_result(0, 'DEP').values, color='blue',linestyle='--' )
plt.axhline(y= quant_result(0.5, 'DEP').values, color='red',linestyle='--' )
print(quant_result(1,'DEP'), quant_result(0, 'DEP'), quant_result(0.5, 'DEP'))

#Hors Corse
#On obtient pour valeur médiane le Tarn-et-Garonne (82), pour valeur minimale la SSD (93), et pout maximale la Mayenne(53)
#On poussera notre étude statistique dans ces départements précis, pour déterminer les causes de l


# In[564]:


retard_academ = pd.DataFrame(df3.groupby(['Académie'])['taux_retard_sixième'].mean())
part_fav_academ = pd.DataFrame(df3.groupby(['Académie'])['part_fav'].mean())

fig = plt.figure()

ax = fig.add_subplot(111) 
ax2 = ax.twinx() 

width = 0.4

retard_academ.plot(kind='bar', color='red', ax=ax, width=width, position=1)
reussite_academ.plot(kind='bar', color='blue', ax=ax2, width=width, position=0)

ax.set_ylabel('Retard à l entrée en 6_ème')
ax2.set_ylabel('réussite au bac')

plt.show()


fig = plt.figure()

ax = fig.add_subplot(111) 
ax2 = ax.twinx() 

width = 0.4

retard_academ.plot(kind='bar', color='red', ax=ax, width=width, position=1)
part_fav_academ.plot(kind='bar', color='blue', ax=ax2, width=width, position=0)

ax.set_ylabel('Part favorisé par académie')
ax2.set_ylabel('réussite au bac')

plt.show()



print(quant_result(1,'Académie'), quant_result(0, 'Académie'), quant_result(0.5, 'Académie'))
#Académie médiane : ORLEANS-TOURS
#Académie min : MAYOTTE 
#Académie max : CORSE


# In[160]:


df3["nb_hotel_com"] = df3["nb_hotel_com"].fillna(0, inplace = False)


# In[567]:


correlation_liste = df3[["Taux Brut de Réussite série L","Taux Brut de Réussite série ES", "Taux Brut de Réussite série S", "Taux_gen", "Taux_sup", 'Prixm2','part_fav', 'taux_retard_sixième', "MED17", "P17_POP", "DEN17", "nb_hotel_com", "T17_CHOM1564"]].corr()


# In[568]:


correlation_liste.style.background_gradient(cmap="RdBu_r", low = 0 , high = 0)

