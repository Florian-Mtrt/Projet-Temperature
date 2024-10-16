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

  df_github_fm = df_github[["country", "year", "iso_code"]]
  df_ZonAnn_Ts_dSST_fm = df_ZonAnn_Ts_dSST.rename(columns={"Year": "year"})
  df_ZonAnn_Ts_dSST_hem = df_ZonAnn_Ts_dSST_fm[["year", "24N-90N", "24S-24N", "90S-24S"]]
  df_ZonAnn_Ts_dSST_zone = df_ZonAnn_Ts_dSST_fm[["year", "64N-90N", "44N-64N", "24N-44N", "EQU-24N", "24S-EQU", "44S-24S", "64S-44S", "90S-64S"]]
  iso_code_unique = df_github["iso_code"].unique()

  iso_code_hemisphere = {
    'AFG': "24N-90N",
    'ALB': "24N-90N",
    'DZA': "24N-90N",
    'AND': "24N-90N",
    'AGO': "24S-24N",
    'AIA': "24S-24N",
    'ATA': "90S-24S",
    'ATG': "24S-24N",
    'ARG': "90S-24S",
    'ARM': "24N-90N",
    'ABW': "24S-24N",
    'AUS': "90S-24S",
    'AUT': "24N-90N",
    'AZE': "24N-90N",
    'BHS': "24S-24N",
    'BHR': "24N-90N",
    'BGD': "24N-90N",
    'BRB': "24S-24N",
    'BLR': "24N-90N",
    'BEL': "24N-90N",
    'BLZ': "24S-24N",
    'BEN': "24S-24N",
    'BMU': "24N-90N",
    'BTN': "24N-90N",
    'BOL': "24S-24N",
    'BES': "24N-90N",
    'BIH': "24N-90N",
    'BWA': "24S-24N",
    'BRA': "24S-24N",
    'VGB': "24S-24N",
    'BRN': "24S-24N",
    'BGR': "24N-90N",
    'BFA': "24S-24N",
    'BDI': "24S-24N",
    'KHM': "24S-24N",
    'CMR': "24S-24N",
    'CAN': "24N-90N",
    'CPV': "24S-24N",
    'CAF': "24S-24N",
    'TCD': "24S-24N",
    'CHL': "90S-24S",
    'CHN': "24N-90N",
    'CXR': "24S-24N",
    'COL': "24S-24N",
    'COM': "24S-24N",
    'COG': "24S-24N",
    'COK': "24S-24N",
    'CRI': "24S-24N",
    'CIV': "24S-24N",
    'HRV': "24N-90N",
    'CUB': "24S-24N",
    'CUW': "24S-24N",
    'CYP': "24N-90N",
    'CZE': "24N-90N",
    'COD': "24S-24N",
    'DNK': "24N-90N",
    'DJI': "24S-24N",
    'DMA': "24S-24N",
    'DOM': "24S-24N",
    'TLS': "24S-24N",
    'ECU': "24S-24N",
    'EGY': "24N-90N",
    'SLV': "24S-24N",
    'GNQ': "24S-24N",
    'ERI': "24S-24N",
    'EST': "24N-90N",
    'SWZ': "90S-24S",
    'ETH': "24S-24N",
    'FRO': "24N-90N",
    'FJI': "24S-24N",
    'FIN': "24N-90N",
    'FRA': "24N-90N",
    'PYF': "24S-24N",
    'GAB': "24S-24N",
    'GMB': "24S-24N",
    'GEO': "24N-90N",
    'DEU': "24N-90N",
    'GHA': "24S-24N",
    'GRC': "24N-90N",
    'GRL': "24N-90N",
    'GRD': "24S-24N",
    'GTM': "24S-24N",
    'GIN': "24S-24N",
    'GNB': "24S-24N",
    'GUY': "24S-24N",
    'HTI': "24S-24N",
    'HND': "24S-24N",
    'HKG': "24N-90N",
    'HUN': "24N-90N",
    'ISL': "24N-90N",
    'IND': "24N-90N",
    'IDN': "24S-24N",
    'IRN': "24N-90N",
    'IRQ': "24N-90N",
    'IRL': "24N-90N",
    'ISR': "24N-90N",
    'ITA': "24N-90N",
    'JAM': "24S-24N",
    'JPN': "24N-90N",
    'JOR': "24N-90N",
    'KAZ': "24N-90N",
    'KEN': "24S-24N",
    'KIR': "24S-24N",
    'KWT': "24N-90N",
    'KGZ': "24N-90N",
    'LAO': "24N-90N",
    'LVA': "24N-90N",
    'LBN': "24N-90N",
    'LSO': "90S-24S",
    'LBR': "24S-24N",
    'LBY': "24N-90N",
    'LIE': "24N-90N",
    'LTU': "24N-90N",
    'LUX': "24N-90N",
    'MAC': "24N-90N",
    'MDG': "24S-24N",
    'MWI': "24S-24N",
    'MYS': "24S-24N",
    'MDV': "24S-24N",
    'MLI': "24S-24N",
    'MLT': "24N-90N",
    'MHL': "24S-24N",
    'MRT': "24S-24N",
    'MUS': "24S-24N",
    'MEX': "24S-24N",
    'FSM': "24S-24N",
    'MDA': "24N-90N",
    'MCO': "24N-90N",
    'MNG': "24N-90N",
    'MNE': "24N-90N",
    'MSR': "24S-24N",
    'MAR': "24N-90N",
    'MOZ': "24S-24N",
    'MMR': "24S-24N",
    'NAM': "24S-24N",
    'NRU': "24S-24N",
    'NPL': "24N-90N",
    'NLD': "24N-90N",
    'NCL': "24S-24N",
    'NZL': "90S-24S",
    'NIC': "24S-24N",
    'NER': "24S-24N",
    'NGA': "24S-24N",
    'NIU': "24S-24N",
    'PRK': "24N-90N",
    'MKD': "24N-90N",
    'NOR': "24N-90N",
    'OMN': "24N-90N",
    'PAK': "24N-90N",
    'PLW': "24S-24N",
    'PSE': "24N-90N",
    'PAN': "24S-24N",
    'PNG': "24S-24N",
    'PRY': "24S-24N",
    'PER': "24S-24N",
    'PHL': "24S-24N",
    'POL': "24N-90N",
    'PRT': "24N-90N",
    'PRI': "24S-24N",
    'QAT': "24N-90N",
    'ROU': "24N-90N",
    'RUS': "24N-90N",
    'RWA': "24S-24N",
    'SHN': "24S-24N",
    'KNA': "24S-24N",
    'LCA': "24S-24N",
    'SPM': "24N-90N",
    'VCT': "24S-24N",
    'WSM': "24S-24N",
    'SMR': "24N-90N",
    'STP': "24S-24N",
    'SAU': "24N-90N",
    'SEN': "24S-24N",
    'SRB': "24N-90N",
    'SYC': "24S-24N",
    'SLE': "24S-24N",
    'SGP': "24S-24N",
    'SXM': "24S-24N",
    'SVK': "24N-90N",
    'SVN': "24N-90N",
    'SLB': "24S-24N",
    'SOM': "24S-24N",
    'ZAF': "90S-24S",
    'KOR': "24N-90N",
    'SSD': "24S-24N",
    'ESP': "24N-90N",
    'LKA': "24N-90N",
    'SDN': "24S-24N",
    'SUR': "24S-24N",
    'SWE': "24N-90N",
    'CHE': "24N-90N",
    'SYR': "24N-90N",
    'TWN': "24N-90N",
    'TJK': "24N-90N",
    'TZA': "24S-24N",
    'THA': "24S-24N",
    'TGO': "24S-24N",
    'TON': "24S-24N",
    'TTO': "24S-24N",
    'TUN': "24N-90N",
    'TUR': "24N-90N",
    'TKM': "24N-90N",
    'TCA': "24S-24N",
    'TUV': "24S-24N",
    'UGA': "24S-24N",
    'UKR': "24N-90N",
    'ARE': "24N-90N",
    'GBR': "24N-90N",
    'USA': "24N-90N",
    'URY': "90S-24S",
    'UZB': "24N-90N",
    'VUT': "24S-24N",
    'VAT': "24N-90N",
    'VEN': "24S-24N",
    'VNM': "24S-24N",
    'WLF': "24S-24N",
    'YEM': "24N-90N",
    'ZMB': "24S-24N",
    'ZWE': "24S-24N",
  }
    
  iso_code_zones = {
      'AFG': "24N-44N",
      'ALB': "24N-44N",
      'DZA': "24N-44N",
      'AND': "24N-44N",
      'AGO': "24S-EQU",
      'AIA': "EQU-24N",
      'ATA': "90S-64S",
      'ATG': "EQU-24N",
      'ARG': "44S-24S",
      'ARM': "24N-44N",
      'ABW': "EQU-24N",
      'AUS': "44S-24S",
      'AUT': "44N-64N",
      'AZE': "24N-44N",
      'BHS': "EQU-24N",
      'BHR': "24N-44N",
      'BGD': "24N-44N",
      'BRB': "EQU-24N",
      'BLR': "44N-64N",
      'BEL': "44N-64N",
      'BLZ': "EQU-24N",
      'BEN': "EQU-24N",
      'BMU': "24N-44N",
      'BTN': "24N-44N",
      'BOL': "24S-EQU",
      'BES': "44N-64N",
      'BIH': "44N-64N",
      'BWA': "24S-EQU",
      'BRA': "24S-EQU",
      'VGB': "EQU-24N",
      'BRN': "EQU-24N",
      'BGR': "24N-44N",
      'BFA': "EQU-24N",
      'BDI': "24S-EQU",
      'KHM': "EQU-24N",
      'CMR': "EQU-24N",
      'CAN': "44N-64N",
      'CPV': "EQU-24N",
      'CAF': "EQU-24N",
      'TCD': "EQU-24N",
      'CHL': "44S-24S",
      'CHN': "24N-44N",
      'CXR': "24S-EQU",
      'COL': "EQU-24N",
      'COM': "24S-EQU",
      'COG': "24S-EQU",
      'COK': "24S-EQU",
      'CRI': "EQU-24N",
      'CIV': "EQU-24N",
      'HRV': "44N-64N",
      'CUB': "EQU-24N",
      'CUW': "EQU-24N",
      'CYP': "24N-44N",
      'CZE': "44N-64N",
      'COD': "24S-EQU",
      'DNK': "44N-64N",
      'DJI': "EQU-24N",
      'DMA': "EQU-24N",
      'DOM': "EQU-24N",
      'TLS': "24S-EQU",
      'ECU': "24S-EQU",
      'EGY': "24N-44N",
      'SLV': "EQU-24N",
      'GNQ': "EQU-24N",
      'ERI': "EQU-24N",
      'EST': "44N-64N",
      'SWZ': "44S-24S",
      'ETH': "EQU-24N",
      'FRO': "44N-64N",
      'FJI': "24S-EQU",
      'FIN': "44N-64N",
      'FRA': "44N-64N",
      'PYF': "24S-EQU",
      'GAB': "24S-EQU",
      'GMB': "EQU-24N",
      'GEO': "24N-44N",
      'DEU': "44N-64N",
      'GHA': "EQU-24N",
      'GRC': "24N-44N",
      'GRL': "64N-90N",
      'GRD': "EQU-24N",
      'GTM': "EQU-24N",
      'GIN': "EQU-24N",
      'GNB': "EQU-24N",
      'GUY': "EQU-24N",
      'HTI': "EQU-24N",
      'HND': "EQU-24N",
      'HKG': "24N-44N",
      'HUN': "44N-64N",
      'ISL': "64N-90N",
      'IND': "24N-44N",
      'IDN': "24S-EQU",
      'IRN': "24N-44N",
      'IRQ': "24N-44N",
      'IRL': "44N-64N",
      'ISR': "24N-44N",
      'ITA': "24N-44N",
      'JAM': "EQU-24N",
      'JPN': "24N-44N",
      'JOR': "24N-44N",
      'KAZ': "44N-64N",
      'KEN': "EQU-24N",
      'KIR': "24S-EQU",
      'KWT': "24N-44N",
      'KGZ': "24N-44N",
      'LAO': "24N-44N",
      'LVA': "44N-64N",
      'LBN': "24N-44N",
      'LSO': "24S-EQU",
      'LBR': "EQU-24N",
      'LBY': "24N-44N",
      'LIE': "44N-64N",
      'LTU': "44N-64N",
      'LUX': "44N-64N",
      'MAC': "24N-44N",
      'MDG': "24S-EQU",
      'MWI': "24S-EQU",
      'MYS': "24S-EQU",
      'MDV': "EQU-24N",
      'MLI': "EQU-24N",
      'MLT': "24N-44N",
      'MHL': "24S-EQU",
      'MRT': "EQU-24N",
      'MUS': "24S-EQU",
      'MEX': "EQU-24N",
      'FSM': "24S-EQU",
      'MDA': "44N-64N",
      'MCO': "24N-44N",
      'MNG': "44N-64N",
      'MNE': "44N-64N",
      'MSR': "EQU-24N",
      'MAR': "24N-44N",
      'MOZ': "24S-EQU",
      'MMR': "24N-44N",
      'NAM': "24S-EQU",
      'NRU': "24S-EQU",
      'NPL': "24N-44N",
      'NLD': "44N-64N",
      'NCL': "24S-EQU",
      'NZL': "44S-24S",
      'NIC': "EQU-24N",
      'NER': "EQU-24N",
      'NGA': "EQU-24N",
      'NIU': "24S-EQU",
      'PRK': "24N-44N",
      'MKD': "24N-44N",
      'NOR': "44N-64N",
      'OMN': "24N-44N",
      'PAK': "24N-44N",
      'PLW': "24S-EQU",
      'PSE': "24N-44N",
      'PAN': "EQU-24N",
      'PNG': "24S-EQU",
      'PRY': "24S-EQU",
      'PER': "24S-EQU",
      'PHL': "24S-EQU",
      'POL': "44N-64N",
      'PRT': "24N-44N",
      'PRI': "EQU-24N",
      'QAT': "24N-44N",
      'ROU': "44N-64N",
      'RUS': "44N-64N",
      'RWA': "24S-EQU",
      'SHN': "24S-EQU",
      'KNA': "EQU-24N",
      'LCA': "EQU-24N",
      'SPM': "44N-64N",
      'VCT': "EQU-24N",
      'WSM': "24S-EQU",
      'SMR': "24N-44N",
      'STP': "EQU-24N",
      'SAU': "24N-44N",
      'SEN': "EQU-24N",
      'SRB': "44N-64N",
      'SYC': "24S-EQU",
      'SLE': "EQU-24N",
      'SGP': "EQU-24N",
      'SXM': "EQU-24N",
      'SVK': "44N-64N",
      'SVN': "44N-64N",
      'SLB': "24S-EQU",
      'SOM': "EQU-24N",
      'ZAF': "44S-24S",
      'KOR': "24N-44N",
      'SSD': "EQU-24N",
      'ESP': "24N-44N",
      'LKA': "24N-44N",
      'SDN': "EQU-24N",
      'SUR': "EQU-24N",
      'SWE': "44N-64N",
      'CHE': "44N-64N",
      'SYR': "24N-44N",
      'TWN': "24N-44N",
      'TJK': "24N-44N",
      'TZA': "EQU-24N",
      'THA': "24N-44N",
      'TGO': "EQU-24N",
      'TON': "24S-EQU",
      'TTO': "EQU-24N",
      'TUN': "24N-44N",
      'TUR': "24N-44N",
      'TKM': "24N-44N",
      'TCA': "EQU-24N",
      'TUV': "24S-EQU",
      'UGA': "EQU-24N",
      'UKR': "44N-64N",
      'ARE': "24N-44N",
      'GBR': "44N-64N",
      'USA': "24N-44N",
      'URY': "44S-24S",
      'UZB': "24N-44N",
      'VUT': "24S-EQU",
      'VAT': "24N-44N",
      'VEN': "EQU-24N",
      'VNM': "24N-44N",
      'WLF': "24S-EQU",
      'YEM': "24N-44N",
      'ZMB': "24S-EQU",
      'ZWE': "24S-EQU",
  }

  df_ZonAnn_Ts_dSST_fm2 = df_github_fm.copy()
  df_ZonAnn_Ts_dSST_fm2["zones"] = df_ZonAnn_Ts_dSST_fm2["iso_code"].map(iso_code_zones)
  df_ZonAnn_Ts_dSST_fm2 = df_ZonAnn_Ts_dSST_fm2[df_ZonAnn_Ts_dSST_fm2["year"] >= 1900]
  df_github_fm["hemisphere"] = df_github_fm["iso_code"].map(iso_code_hemisphere)
  df_github_fm = df_github_fm[df_github_fm["year"] >= 1900]

  df_merge_fm = pd.merge(df_github_fm, df_ZonAnn_Ts_dSST_hem, on="year", how="left")
  df_merge_fm.fillna(value={'24N-90N': 0, '24S-24N': 0, '90S-24S': 0}, inplace=True)

  def get_temperature(row):
      return row[row['hemisphere']] if row['hemisphere'] in row else None

  df_merge_fm['temperature'] = df_merge_fm.apply(get_temperature, axis=1)

  df_merge_fm2 = pd.merge(df_ZonAnn_Ts_dSST_fm2, df_ZonAnn_Ts_dSST_zone, on="year", how="left")
  df_merge_fm2.fillna(value={'64N-90N': 0, '44N-64N': 0, '24N-44N': 0, 'EQU-24N': 0, '24S-EQU': 0, '44S-24S': 0, '64S-44S': 0, '90S-64S': 0}, inplace=True)
    
  df_filtered_fm2 = df_merge_fm2[df_merge_fm2["year"] % 10 == 0]
  st.write("Colonnes dans df_filtered_fm2 :", df_filtered_fm2.columns)
    
  min_temp_change = df_filtered_fm2["temperature"].min()
  max_temp_change = df_filtered_fm2["temperature"].max()

  fig = px.choropleth(df_filtered_fm2,
                      locations="iso_code",
                      color="temperature",
                      color_continuous_scale=px.colors.sequential.Plasma,
                      labels={'temperature': 'Température'},
                      title='Changement de température par pays')

  st.title("Visualisation des Changements de Température")
  st.plotly_chart(fig)

  if st.checkbox('Afficher les données filtrées'):
      st.write(df_filtered_fm2)


##########################################################
#MODELISATION
##########################################################

if page == pages[3] :
  st.write("### IV. Modélisation")

  st.write("#### Prédiction des futures données de température")
  texte_modelisation_fm_1 = """
  Pour le choix du modèle, nous avons testé plusieurs algorithmes, parmi lesquels le modèle ARIMA a été retenu pour prédire les températures jusqu'en 2050.
  Ce modèle est particulièrement adapté à la modélisation des données climatiques, car il permet de gérer à la fois la tendance et la saisonnalité des données.
  Ceci est un test
  """
  st.write(texte_modelisation_fm_1)

