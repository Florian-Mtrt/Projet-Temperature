import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

@st.cache
def load_data_owid():
    return pd.read_csv('owidco2data.csv', header=0)

@st.cache
def load_data_nasa():
    return pd.read_csv('GLB.Ts+dSST.csv', header=1, index_col=1)

@st.cache
def load_data_zonann():
    return pd.read_csv('ZonAnn.Ts+dSST.csv', header=0)

# Chargement des données
df_github = load_data_owid()
df_GLB_NASA = load_data_nasa()
df_ZonAnn_Ts_dSST = load_data_zonann()

#df_github = pd.read_csv('owidco2data.csv', header=0)
#df_GLB_NASA = pd.read_csv('GLB.Ts+dSST.csv', header=1, index_col=1)
#df_ZonAnn_Ts_dSST = pd.read_csv('ZonAnn.Ts+dSST.csv', header=0)

st.title("Température Terrestre")

st.sidebar.title("Sommaire")
pages=["Introduction au projet", "Compréhension et manipulation des données", "DataVisualisation", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

##########################################################
#INTRODUCTION AU PROJET
##########################################################

if page == pages[0] :
  st.write("### I. INTRODUCTION AU PROJET")
  texte_introduction_au_projet = """
  Ce projet s’inscrit pleinement dans notre apprentissage du métier de Data Analyst grâce à la manipulation de données “from scratch”, menant à l’analyse et l’interprétation de ces données brutes.
  De plus, ce sujet s’inscrit pleinement dans le contexte professionnel d’une partie de notre groupe, confrontés aux objectifs carbones et environnementaux qui sont essentiels face aux évolutions rapides des réglementations.
  L’objectif de ce projet est de constater le réchauffement climatique et le dérèglement climatique global à l’échelle de la planète sur les derniers siècles et dernières décennies. Ce phénomène sera analysé au niveau mondial et par zone géographique. Nous comparerons avec des phases d’évolution de température antérieure à notre époque.
  Le projet est piloté par Florian Delattre, Philippe Grenesche, Yves Liais et Florian Matrat,
  et est supervisé par Alain Ferlac.
  Aucun des membres du projet n’a d’expérience dans l’analyse du réchauffement climatique.
  """
  st.write(texte_introduction_au_projet)

##########################################################
#COMPRÉHENSION ET MANIPULATION DES DONNÉES
##########################################################

if page == pages[1] :
  st.write("### II. COMPRÉHENSION ET MANIPULATION DES DONNÉES")
  st.write("#### Cadre")
  texte_cadre = """
  Les données utilisées sont celles de la NASA et de Our World in Data. Elles sont accessibles librement sur le site de la NASA et via GitHub.
  Concernant les données de la NASA nous avons accès à 4 fichiers. GLB, NH et SH sont structurés en 145 lignes de 19 colonnes.
  ZonAnn est organisé en 144 lignes sur 15 colonnes. Nous avons une ligne de moins car nous n’avons pas de données pour l’année 2024.

  Concernant les données GitHub, nous y trouvons 47416 lignes pour 79 colonnes.

  Rappel des ressources à consulter :

  NASA : https://data.giss.nasa.gov/gistemp/
  GitHub : https://github.com/owid/co2-data
  """
  st.write(texte_cadre)

  st.write("#### Pertinence")
  texte_pertinence = """
  Quelles variables vous semblent les plus pertinentes au regard de vos objectifs ?
  Les variables les plus pertinentes que nous avons sélectionnées pour établir nos datavisualisations sont :

  NASA :

  - la période (année/mois/saisons) pour suivre les évolutions dans le temps
  - les écarts de température au fil du temps
  - le zonage géographique (hémisphères)

  GitHub :

  - Nom des pays
  - ISO CODE des pays
  - Années
  - Densité de population par pays
  - Émissions total annuelles de CO₂ (en millions de tonnes)
  - Émissions annuelles de CO₂ (par habitant)
  - Émissions cumulées totales de CO₂ (en millions de tonnes)
  - Consommation d'énergie primaire par habitant (en kWh par habitant)
  - Émissions annuelles de CO₂ liées au changement d’affectation des terres (en millions de tonnes)
  - Émissions annuelles de CO₂ liées au changement d’affectation des terres (en millions de tonnes par habitant)
  - Consommation d'énergie primaire (en térawattheures)
  - Changement de la température moyenne mondiale provoqué par les émissions de méthane (en °C)
  - Modification de la température moyenne mondiale provoqué par les émissions de CO₂ (en °C)
  - Modification de la température moyenne mondiale provoqué par les Gaz à effet de serre (en °C)
  - Modification de la température moyenne mondiale provoqué par les émissions d'oxyde d'azote (en °C)
  - Émissions totales de gaz à effet de serre (en millions de tonnes)


  Quelle est la variable cible ?

  Notre principale variable cible est l’écart de température par rapport à la moyenne comprise pour la période 1951-1980 (Dataframe de la NASA).

  Quelles particularités de votre jeu de données pouvez-vous mettre en avant ?

  Les données provenant de GitHub ne couvrent pas les mêmes périodes que les données fournies par la NASA (Les données de la NASA couvrent la période de 1880 jusqu'à aujourd'hui, tandis que les données de GitHub incluent des informations antérieures à 1880.)
  Les données provenant de GitHub ont énormément de valeurs manquantes.
  Cela peut entraîner un manque d’information (NaN) qui peut poser problème pour la partie visualisation. Nous avons choisi de ne pas garder ses “NaN” pour ne pas influencer les graphiques.
  Il y a également beaucoup d'occurrences répétées pour le même pays ou “code iso” dans le fichier GitHub.

  Etes-vous limités par certaines de vos données ?

  Certaines variables contiennent très peu d'informations, ce qui limite leur utilisation à des plus petites périodes d’observation.
  """
  st.write(texte_pertinence)

  st.write("### Pre-processing et feature engineering")
  texte_Pre_processing = """
  Avez-vous eu à nettoyer et à traiter les données ? Si oui, décrivez votre processus de traitement.

  Oui, sur le dataset de la NASA GLB_Ts_dSST il a fallu opérer quelques modifications  :
  remplacer les *** par NaN pour pouvoir convertir les colonnes de valeurs en type ‘float’
  dé-pivoter les colonnes de ‘mois’ en lignes avec pd.melt
  encoder en valeur numérique les variables mois en valeur alphabétique
  concaténer 'année’ et ‘mois’ pour utiliser le date_time pandas
  encoder les périodes de 3 mois pour faire les 4 saisons

  Concernant GitHub, pour faire le merge (la fusion) des deux dataframes NASA  GLB_Ts_dSST et GitHub il a également fallu également opérer des transformations :
  faire un sous dataframe de Github pour filtrer sur la variable country = ‘world’ pour corréler à la zone géographique du dataset de la NASA  GLB_Ts_dSST
  filtrer sur la variable Year supérieur ou égal à  1880 pour corréler avec la période d’observation du dataset de la NASA  GLB_Ts_dSST
  ne retenir que les variables utiles et pertinentes du dataset GitHub (78 colonnes dans Github, trop d’information)

  Concernant le graphique représentant la hausse de la température moyenne mondiale par zones (globe terrestre).

  Créer des sous Dataframes de “wid-co2-data” et “ZonAnn_Ts_dSST” pour conserver uniquement les variables utiles au graphique.
  Renommer la colonne “year” pour pouvoir effectuer un merge des Dataframes par la suite.
  Créer 2 dictionnaires, le premier est un mapping entre les codes ISO et les hémisphères (Sud - Équateur - Nord).
  Le second un mapping entre les codes ISO et les zones par hémisphères (3 zones pour le Sud et le Nord, 2 pour l’équateur).
  L’ajout des colonnes “hemisphere” et “zones” pour chaque dataframes.
  L’ajout d’une colonne “temperature” qui récupère le bon écart de température en fonction de la colonne “hemisphere” ou “zones”.
  Puis la création de la carte choroplèthe sous Plotly

  Avez-vous dû procéder à des transformations de vos données de type normalisation/standardisation ? Si oui, pourquoi ?
  Envisagez-vous des techniques de réduction de dimension dans la partie de modélisation ? Si oui, pourquoi ?

  Nous inclurons dans notre prochain rapport une section dédiée au machine learning, qui nous permettra de mieux comprendre et anticiper le réchauffement climatique.
  """
  st.write(texte_Pre_processing)

##########################################################
#VISUALISATION
##########################################################

if page == pages[2] :
  st.write("### III. DataVisualisation")

  st.write("### 1. Boite à moustache des écarts de température à la période de référence par saison et par période")

  df_GLB_NASA = df_GLB_NASA.replace('***', float('NaN'))
  df_GLB_NASA[df_GLB_NASA.columns[3:]] = df_GLB_NASA[df_GLB_NASA.columns[3:]].astype('float')
  df_GLB_NASA['Year']=df_GLB_NASA.index

  df_season = pd.melt(df_GLB_NASA, id_vars=['Year'], value_vars=['J-D','DJF','MAM','JJA','SON'])
  df_season = df_season.replace(['J-D','DJF','MAM','JJA','SON'],['Year','Winter','Spring','Summer','Autumn'])
  df_season = df_season.rename(columns={'variable': 'Season', 'value': 'Value'})
  df_season['sub_Period'] = df_season['Year'].apply(lambda x: '1880 à 1940' if x < 1940 else ('1980 à 2000' if 1980 <= x < 2000 else ('2000 à 2024' if 2000 <= x <= 2024 else '1940 à 1980')))

  fig1 = px.box(df_season, x="Season", y="Value", color="Season", facet_col = "sub_Period",
            color_discrete_sequence=px.colors.qualitative.Dark24,
             title = "boxplot par saison par période des écarts de températures",
             labels={
                     "Year": "Année",
                     "Value": "Ecart de température",
                     "Season": "Season",
                     "sub_Period": "Période"
                 },
              width=500)

  st.plotly_chart(fig1)

  st.write("### 2. Swarmplot des écarts de température à la période de référence par saison et par période")


  sns.color_palette(palette = "OrRd", as_cmap=True)
  fig2 = sns.catplot(x = "Season", y = "Value", kind = "swarm", hue = 'Year', data = df_season, aspect=2, palette = "OrRd")
  plt.xlabel('Saisons')
  plt.ylabel('Ecart de températures')
  st.pyplot(fig2)

  st.write("### 3. Catplot des écarts de température à la période de référence par période et par saison")

  sns.color_palette(palette = "OrRd", as_cmap=True)
  fig3 = sns.catplot(x = "sub_Period", y = "Value", hue = 'Season', data = df_season.loc[df_season['Season'] != "Year"], aspect=2,palette = "OrRd")
  plt.xlabel('Périodes')
  plt.ylabel('Ecart de températures')
  st.pyplot(fig3)

  st.write("### 4. Scatterplot des écarts de température à la période de référence par saison, regression linéaire")

  fig4 = px.scatter(df_season, x="Year", y="Value", color="Season",
                    trendline="ols", # ligne de lissage de nuage de points des moindres carrés
                    facet_col='Season',
                    labels={
                     "Year": "Année",
                     "Value": "Ecart de température",
                     "Season": "Season",
                     },
                    title="Nuage de points avec régression des moindres carrés",
                    width=1000, height=400)
  st.plotly_chart(fig4)

  st.write("### 5. Scatterplot des écarts de température à la période de référence par saison, regression localement pondérée")

  fig5 = px.scatter(df_season, x="Year", y="Value", color="Season",
                 trendline='lowess', # ligne de lissage de nuage de points localement pondérée
                 facet_col='Season',
                 facet_col_wrap=5,
                 labels={
                     "Year": "Année",
                     "Value": "Ecart de température",
                     "Season": "Season",
                 },
                 title="Evolution des écarts de températures avec lissage de nuage de points localement pondérée",
                 width=1000, height=400)
  st.plotly_chart(fig5)

##########################################################
#MODELISATION
##########################################################

if page == pages[3] :
  st.write("### IV. Modélisation")

  st.write("#### Partie Yves")

  st.write("#### Prédiction des futures données de température")
  texte_modelisation_fm_1 = """
  Pour le choix du modèle, nous avons testé plusieurs algorithmes, parmi lesquels le modèle ARIMA a été retenu pour prédire les températures jusqu'en 2050.
  Ce modèle est particulièrement adapté à la modélisation des données climatiques, car il permet de gérer à la fois la tendance et la saisonnalité des données.
  Ceci est un test
  """
  st.write(texte_modelisation_fm_1)

