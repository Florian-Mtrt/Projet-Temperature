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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

@st.cache
def load_data_owid():
    return pd.read_csv('owidco2data.csv', header=0)

@st.cache
def load_data_nasa():
    return pd.read_csv('GLB.Ts+dSST.csv', header=1, index_col=0)

@st.cache
def load_data_nasa_bis():
    return pd.read_csv('GLB.Ts+dSST.csv', header=1, index_col=0)

@st.cache
def load_data_zonann():
    return pd.read_csv('ZonAnn.Ts+dSST.csv', header=0)

@st.cache
def load_data_zonann_bis():
    return pd.read_csv('ZonAnn.Ts+dSST.csv', header=0, index_col=0)

# Chargement des données
df_github = load_data_owid()
df_GLB_NASA = load_data_nasa()
df_GLB_NASA_bis = load_data_nasa_bis()
df_ZonAnn_Ts_dSST = load_data_zonann()
df_ZonAnn_Ts_dSST_bis = load_data_zonann_bis()

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
  
  L’objectif de ce projet est de constater le réchauffement climatique et le dérèglement climatique global à l’échelle de la planète sur les derniers siècles et dernières décennies.
  Ce phénomène sera analysé au niveau mondial et par zone géographique. 
  
  Le projet a été piloté par Philippe Grenesche, Yves Liais, Florian Matrat et Florian Delattre
  et a été supervisé par Alain Ferlac.

  """
  st.write(texte_introduction_au_projet)

##########################################################
#COMPRÉHENSION ET MANIPULATION DES DONNÉES
##########################################################

if page == pages[1] :
  st.write("### II. COMPRÉHENSION ET MANIPULATION DES DONNÉES")
  st.write("#### Cadre")
  texte_cadre1_1 = """
  Les données utilisées sont celles de la **NASA** et de *Our World in Data* via **GitHub**. 
  
  Concernant les données de la NASA nous avons accès à 4 fichiers.
  GLB, NH et SH sont structurés en 145 lignes de 19 colonnes.
  ZonAnn est organisé en 144 lignes sur 15 colonnes. Nous avons une ligne de moins car nous n’avons pas de données pour l’année 2024.
  """
  texte_cadre1_2 = """
  Concernant les données GitHub, nous y trouvons 47416 lignes pour 79 colonnes.
  """
  st.write(texte_cadre1_1)
  st.write(df_GLB_NASA_bis.head())
  st.write(df_ZonAnn_Ts_dSST_bis.head())
  st.write(texte_cadre1_2)
  st.write(df_github[df_github['country']=='World'].head())
    
  texte_cadre2 = """
  Rappel des ressources à consulter :

  NASA : https://data.giss.nasa.gov/gistemp/
  GitHub : https://github.com/owid/co2-data
  """
  st.write(texte_cadre2)

  st.write("#### Pertinence")
  texte_pertinence = """
 
  Les variables les plus pertinentes que nous avons sélectionnées pour établir nos datavisualisations sont :

  *NASA* :
  - la **période** (année/mois/saisons) pour suivre les évolutions dans le temps
  - les **écarts de température** au fil du temps
  - le **zonage géographique** (hémisphères)

  *GitHub* :
  - **Nom des pays**
  - **ISO CODE des pays**
  - **Années**
  - **Densité de population par pays**
  - **Émissions total annuelles de CO₂** (en millions de tonnes)
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


  *Particularités du jeu de données* :
  
  Les données provenant de **GitHub ne couvrent pas les mêmes périodes que les données fournies par la NASA**.
  Les données de la NASA couvrent la période de 1880 jusqu'à aujourd'hui, tandis que les données de GitHub incluent des informations antérieures à 1880.
  Les données provenant de GitHub ont **énormément de valeurs manquantes**.
  
  """
  st.write(texte_pertinence)

  st.write("### Pre-processing et feature engineering")
  texte_Pre_processing = """

  Sur le dataset de la NASA GLB_Ts_dSST il a fallu opérer quelques modifications :
  - remplacer les *** par NaN pour pouvoir convertir les colonnes de valeurs en type ‘float’
  - dé-pivoter les colonnes de ‘mois’ en lignes avec pd.melt
  - encoder en valeur numérique les variables mois en valeur alphabétique
  - concaténer 'année’ et ‘mois’ pour utiliser le date_time pandas
  - encoder les périodes de 3 mois pour faire les 4 saisons

  Concernant GitHub, pour faire le merge (la fusion) des deux dataframes NASA  GLB_Ts_dSST et GitHub il a également fallu également opérer des transformations :
  - faire un sous dataframe de Github pour filtrer sur la variable country = ‘world’ pour corréler à la zone géographique du dataset de la NASA  GLB_Ts_dSST
  - filtrer sur la variable Year supérieur ou égal à  1880 pour corréler avec la période d’observation du dataset de la NASA  GLB_Ts_dSST
  - ne retenir que les variables utiles et pertinentes du dataset GitHub (78 colonnes dans Github, trop d’information)

  Concernant le graphique représentant la hausse de la température moyenne mondiale par zones (globe terrestre).
  - Créer des sous Dataframes de “wid-co2-data” et “ZonAnn_Ts_dSST” pour conserver uniquement les variables utiles au graphique.
  - Renommer la colonne “year” pour pouvoir effectuer un merge des Dataframes par la suite.
  - Créer 2 dictionnaires, le premier est un mapping entre les codes ISO et les hémisphères (Sud - Équateur - Nord).
  - Le second un mapping entre les codes ISO et les zones par hémisphères (3 zones pour le Sud et le Nord, 2 pour l’équateur).
  - L’ajout des colonnes “hemisphere” et “zones” pour chaque dataframes.
  - L’ajout d’une colonne “temperature” qui récupère le bon écart de température en fonction de la colonne “hemisphere” ou “zones”.
  - Puis la création de la carte choroplèthe sous Plotly
  """
  st.write(texte_Pre_processing)

##########################################################
#VISUALISATION
##########################################################

if page == pages[2] :
  st.write("### III. DataVisualisation")
    
  st.write("### 0. Nuage de points des écarts de températures à la période de référence mois par mois")
    
  """
  Nous affichons ci-contre les écarts de températures mondiales, mois par mois. 
  Les écarts de températures sont mesurés par rapport à la période de référence de 1940-1980. 
  Ainsi nous observons qu'avant cette période de référence les écarts sont globalement négatifs (il faisait "globalement plus froid").
  Et qu'après cette période de référence les écarts de températures sont gloabalement de plus en plus positif (il fait de plus en plus chaud !).
  """
    
  df_GLB_NASA = df_GLB_NASA.replace('***', float('NaN'))
  df_GLB_NASA[df_GLB_NASA.columns[3:]] = df_GLB_NASA[df_GLB_NASA.columns[3:]].astype('float')
  df_GLB_NASA['Year']=df_GLB_NASA.index
  #st.dataframe(df_GLB_NASA.tail())
    
  df_month = pd.melt(df_GLB_NASA, id_vars=['Year'], value_vars=['Jan', 'Feb','Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
  df_month = df_month.replace(['Jan', 'Feb','Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                              ['01', '02','03', '04', '05', '06', '07', '08', '09', '10', '11', '12'])
  df_month['Date'] = df_month['Year'].astype(str) + '-' + df_month['variable'].astype(str) + '-01'
  df_month['Date'] = pd.to_datetime(df_month['Date'], yearfirst = True)
  df_month = df_month.rename(columns={'variable': 'Month', 'value': 'Value'})
  df_month['Absolute'] = np.absolute(df_month['Value'])
  df_month['Month'] = df_month['Month'].astype(int)
  #st.dataframe(df_month.head())
    
  df_month_dropna = df_month.dropna()
  df_month_dropna['Season'] = df_month_dropna['Month'].replace(np.arange(1,13),['Winter','Winter','Spring','Spring','Spring','Summer','Summer','Summer','Autumn','Autumn','Autumn','Winter'])
  #st.dataframe(df_month_dropna.head())
    
  fig0 = px.scatter(df_month_dropna, x="Year", y="Value",size = "Absolute", color = "Season",
                    hover_name="Date", color_discrete_sequence=px.colors.qualitative.Dark24,
                    #title='nuage de point des écarts de température',
                    labels={
                     "Year": "Année",
                     "Value": "Ecart de température (°C)",
                     "Season": "Season",
                 },
                 width=800, height=400)
  st.plotly_chart(fig0)
    
  st.write("### 1. Evolution des émissions de Gaz à effet de serres dans le monde")

  """
  Nous commençons par examiner les tendances mondiales, en montrant l'augmentation des concentrations des différents types de GES (Gaz à Effet de Serres) dans l'atmosphère
  et principalement du dioxyde de carbone (CO₂), qui est l'un des principaux moteurs du réchauffement climatique.
  """
  fig_fd_1, ax1 = plt.subplots()
  sns.lineplot(x='year', y='temperature_change_from_ghg', data=df_github[df_github['country']=='World'], ax=ax1, label='Gaz à effet de serre (GHG)')
  sns.lineplot(x='year', y='temperature_change_from_ch4', data=df_github[df_github['country']=='World'], ax=ax1, label='Méthane (CH4)')
  sns.lineplot(x='year', y='temperature_change_from_co2', data=df_github[df_github['country']=='World'], ax=ax1, label='Dioxyde de carbone (CO2)')
  sns.lineplot(x='year', y='temperature_change_from_n2o', data=df_github[df_github['country']=='World'], ax=ax1, label='Protoxyde d\'azote (N2O)')
  plt.title('Changement de la température en fonction de différentes causes')
  plt.xlabel('Années')
  plt.ylabel('Changement de la température moyenne globale (en °C)')

  st.pyplot(fig_fd_1.get_figure())
  st.write()
  """
  L'impact le plus important est causé par le Dioxyde de carbone, avant le Méthane et le Protoxyde d'azote.
  """
  fig_fd_2, ax2 = plt.subplots()
  sns.lineplot(x='year', y='co2', data=df_github[df_github['country']=='World'], ax=ax2)
  plt.title('Emissisons de CO2 dans le monde par année')
  plt.xlabel('Années')
  plt.ylabel('Emissions de CO2 (en millions de tonnes)')

  st.pyplot(fig_fd_2.get_figure())
  st.write()
  """
  Grâce aux données GitHub concernant les émissions de CO2 par année, nous pouvons constater une augmentation de ces émissions dans le monde au fil du temps.
  Nous pouvons même remarquer une forte augmentation après 1950.
  """
  df_america = df_github[df_github['country'].isin(['North America (GCP)', 'South America (GCP)', 'Central America (GCP)'])]
  df_america_data = df_america.groupby('year')['co2'].sum().reset_index()
  df_america_data['country'] = 'America (GCP)'
  df_github_bis = pd.concat([df_github, df_america_data], ignore_index=True)
  df_continents = df_github_bis[df_github_bis['country'].isin(['Africa (GCP)','America (GCP)','Asia (GCP)','Europe (GCP)','Oceania (GCP)'])]

  fig_fd_3, ax3 = plt.subplots()
  sns.lineplot(x='year', y='co2', hue='country', data=df_continents, ax=ax3)
  plt.title('Emissions de CO2 par année par continents')
  plt.xlabel('Années')
  plt.ylabel('Emissions de CO2 (en millions de tonnes)')
  plt.legend()

  st.pyplot(fig_fd_3.get_figure())
  st.write()
  """
  Nous affichons ici les émissions de CO2 par année et par continent.
  Nous pouvons remarquer qu'à partir des années 1990 il y a un croisement entre l'Europe, l'Amérique et l'Asie.
  L'Europe devient le moins émetteur des 3 suite à une forte baisse.
  L'Amérique entame également un ralentissement après les années 2000.
  Au contraire, l'Asie a fortement augmenté ses émissions après les années 1950. A partir de 2000 c'est le continent le plus émetteur de CO2.
  """
  
  st.write("### 2. Boite à moustache des écarts de température à la période de référence par saison et par période")
    
  """
  Le boxplot ci-dessous permet de segmenter les écarts dans le temps et par saison.
  Boxplot globalement inférieur à 0°C entre 1880 et 1940 : période globalement plus froide de -0.5°C à 0°C versus la période de référence ;
  La médiane du Boxplot globalement positionné à 0°C sur la période de 1940 et 1980, c’est la période de référence pour les mesures d’écarts de températures ;
  Boxplot globalement supérieure à 0°C entre 1980 et aujourd’hui : entre 0°C et +0.5°C sur la période 1989 à 2000 et supérieur à +0.5°C jusqu’à des valeurs extrêmes supérieurs à +1°C de 2000 à maintenant.
  """
    
  df_season = pd.melt(df_GLB_NASA, id_vars=['Year'], value_vars=['J-D','DJF','MAM','JJA','SON'])
  df_season = df_season.replace(['J-D','DJF','MAM','JJA','SON'],['Year','Winter','Spring','Summer','Autumn'])
  df_season = df_season.rename(columns={'variable': 'Season', 'value': 'Value'})
  df_season['sub_Period'] = df_season['Year'].apply(lambda x: '1880 à 1940' if x < 1940 else ('1980 à 2000' if 1980 <= x < 2000 else ('2000 à 2024' if 2000 <= x <= 2024 else '1940 à 1980')))
  #st.dataframe(df_season.tail())

  fig1 = px.box(df_season, x="Season", y="Value", color="Season", facet_col = "sub_Period",
            color_discrete_sequence=px.colors.qualitative.Dark24,
             #title = "boxplot par saison par période des écarts de températures",
             labels={
                     "Year": "Année",
                     "Value": "Ecart de température(°C)",
                     "Season": "Season",
                     "sub_Period": "Période"
                 })

  st.plotly_chart(fig1)

  st.write("### 3. Swarmplot des écarts de température à la période de référence par saison et par année")


  sns.color_palette(palette = "OrRd", as_cmap=True)
  fig2 = sns.catplot(x = "Season", y = "Value", kind = "swarm", hue = 'Year', data = df_season, aspect=2, palette = "OrRd")
  plt.xlabel('Saisons')
  plt.ylabel('Ecart de températures(°C)')
  st.pyplot(fig2)

  st.write("### 4. Catplot des écarts de température à la période de référence par période et par saison")

  sns.color_palette(palette = "OrRd", as_cmap=True)
  fig3 = sns.catplot(x = "sub_Period", y = "Value", hue = 'Season', data = df_season.loc[df_season['Season'] != "Year"], aspect=2,palette = "OrRd")
  plt.xlabel('Périodes')
  plt.ylabel('Ecart de températures(°C)')
  st.pyplot(fig3)

  st.write("### 5. Scatterplot des écarts de température à la période de référence par saison, regression linéaire")

  fig5 = px.scatter(df_season, x="Year", y="Value", color="Season",
                    trendline="ols", # ligne de lissage de nuage de points des moindres carrés
                    trendline_color_override="grey",
                    facet_col='Season',
                    facet_col_wrap=3,
                    labels={
                     "Year": "Année",
                     "Value": "Ecart de température(°C)",
                     "Season": "Season",
                     },
                    #title="Nuage de points avec régression des moindres carrés",
                    width=800, height=400)
  fig5.update_traces(marker_size=4)
  st.plotly_chart(fig5)

  st.write("### 6. Scatterplot des écarts de température à la période de référence par saison, regression localement pondérée")

  fig6 = px.scatter(df_season, x="Year", y="Value", color="Season",
                 trendline='lowess', # ligne de lissage de nuage de points localement pondérée
                 trendline_color_override="grey",
                 facet_col='Season',
                 facet_col_wrap=3,
                 labels={
                     "Year": "Année",
                     "Value": "Ecart de température(°C)",
                     "Season": "Season",
                 },
                 #title="Evolution des écarts de températures avec lissage de nuage de points localement pondérée",
                 width=800, height=400)
  fig6.update_traces(marker_size=4)
  st.plotly_chart(fig6)
    
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

  missing_zones = df_merge_fm2['zones'].isna().sum()
  if missing_zones > 0:
      print(f"Il y a {missing_zones} valeurs manquantes dans la colonne 'zones'.")
  def get_temperature(row):
      try:
          return row[row['zones']]
      except KeyError:
          return None
  df_merge_fm2['temperature'] = df_merge_fm2.apply(get_temperature, axis=1)

  df_filtered_fm2 = df_merge_fm2[df_merge_fm2["year"] % 10 == 0]
  min_temp_change = df_filtered_fm2["temperature"].min()
  max_temp_change = df_filtered_fm2["temperature"].max()
  range_color = [min_temp_change * 0, max_temp_change * 0.8]

  fig = px.choropleth(df_filtered_fm2,
                      locations="iso_code",
                      color="temperature",
                      hover_name="country",
                      animation_frame="year",
                      color_continuous_scale=px.colors.sequential.Reds,
                      range_color=range_color,
                      projection="natural earth")

  st.write("### 7. Visualisation des écarts des températures au fil des années")
  st.plotly_chart(fig)

  st.write("### 8. Matrice de corrélation entre les variables : écarts de températures, années, populations, émissions CO2, stock de CO2, et GHG (GreenHouse Gas ou Gaz à Effet de Serre)")
    
  df_year = pd.melt(df_GLB_NASA, id_vars=['Year'], value_vars=['J-D'])
  df_year = df_year.drop('variable', axis=1)
    
  df_github_world = df_github[df_github['country'] == 'World']
  df_github_world_1880 = df_github_world[df_github_world['year']>=1880]
  df_github_world_1880_co2 = df_github_world_1880.filter(items = ['country','year','population','co2', 'cumulative_co2', 'total_ghg'])
  df_merge = pd.merge(df_year, df_github_world_1880_co2, left_on = ['Year'], right_on = ['year'])
  df_merge = df_merge.drop('country',axis=1)
  df_merge = df_merge.drop('year',axis=1)
  fig8 = px.imshow(df_merge.corr(),
                text_auto=True,
                width=800, height=800,
                color_continuous_scale='RdBu_r',   
                range_color=[0,1],
                labels={
                     "Year": "Année",
                     "value": "Ecart de température",
                     "population": "Population",
                     "co2": "Emission de CO2",
                     "cumulative_co2": "Stock cumulatif de CO2",
                     "total_ghg": "Stock cumulatif de CO2"
                 })
  st.plotly_chart(fig8)

  st.write("### 8bis. Scatter Matrix entre les variables : écarts de températures, années, populations, émissions CO2, stock de CO2, et GHG")
    
  fig9 = px.scatter_matrix(
    df_merge,
    dimensions=['Year','value', 'population','co2','cumulative_co2','total_ghg'],
    #title="Matrice des nuages de points entres les variables pré-séléctionnées",
    width=800, height=800,
    color_continuous_scale='RdBu_r',
    labels={
                     "Year": "Année",
                     "value": "Température",
                     "population": "Population",
                     "co2": "CO2",
                     "cumulative_co2": "Stock CO2",
                     "total_ghg": "GHG"
                 },
    color = "value"
    )
  fig9.update_traces(diagonal_visible=False)
  st.plotly_chart(fig9)
##########################################################
#MODELISATION
##########################################################

if page == pages[3] :
  st.write("### IV. Modélisation")
    # récupération des df déjà traité
  df_y = pd.read_csv('df_y.csv')
  df_y_pred = pd.read_csv('df_y_pred.csv')
  df_hem_N = pd.read_csv('df_hem_N.csv')
  df_hem_S = pd.read_csv('df_hem_S.csv')

  texte_modelisation_y_1 = """
  Le but de modélisation est de prédire l'évolution des températures, par rapport à la référence, dans les années futures.
  Nous avons des travaux précédents des données au niveau monte, à la maille de chaque hémisphère et découpées en zone de latitude.
  Les hypothèses suivantes ont été prises:
  - Un modèle entraîné sur les doonées du monde devrait être applicables sur les différents découpages
  - Il est possible de prédire l'évolution enfonction des années et/ou du CO2

  Première approche: essai "naïf" de différents modèles de régression:
  """
  st.write(texte_modelisation_y_1)

  # séparation des variables explicatives de la variable cible
  X = df_y.drop(["temperature_change"], axis = "columns")
  y = df_y["temperature_change"]

  # instanciation des modèles
  reg = LinearRegression()
  dtr = DecisionTreeRegressor()
  rfr = RandomForestRegressor()
  hgb = HistGradientBoostingRegressor()
  abr = AdaBoostRegressor()
  ext = ExtraTreeRegressor()
  sgd = ARDRegression()
  gbr = GradientBoostingRegressor()

  modeles = [("regression lineaire", reg), 
           ("decision tree regressor", dtr), 
           ("random forest regressor", rfr),
           ("HistGradientBoosting regressor", hgb), 
           ("Ada boosting regressor", abr), 
           ("Extra Tree regressor", ext), 
           ("ARD Regression", sgd), 
           ("Gradient boosting regressor", gbr)]

  # création du dataframe de stockage des résultats
  resultats = pd.DataFrame(columns = ["Nom", "score_train", "score_test", "MAE","MSE","RMSE"])
  # séparation du jeu de données
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)
  i = resultats.shape[0]
  for model in modeles:
  # entrainement
      model[1].fit(X_train, y_train)
      # prediction
      y_pred = model[1].predict(X_test)
      # calcul des résultats
      resultats.loc[i, "Nom"] = model[0]
      resultats.loc[i, "score_train"] = model[1].score(X_train, y_train)
      resultats.loc[i, "score_test"] = model[1].score(X_test, y_test)
      resultats.loc[i, "MAE"] = mean_absolute_error(y_test, y_pred)
      resultats.loc[i, "MSE"] = mean_squared_error(y_test, y_pred)
      resultats.loc[i, "RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred))
      i += 1

  # affihage du tableau de résultats
  st.dataframe(resultats.sort_values(by = "score_test", ascending = False).head(8))

  texte_modelisation_y_2 = """
    Le modèle offrant les meilleurs résultats (avec ses hyper-paramètres par défaut) est le Random Forest Regressor.
  C'est lui qui sera considéré pour la suite.
  """
  st.write(texte_modelisation_y_2)
    
  fig_y_1 = go.Figure(data = (go.Scatter(x= df_y_pred["year"], y=df_y_pred["temperature_change"], mode="lines", name = "T°C réelle, monde", marker_color = "green"),
          go.Scatter(x= df_y_pred["year"], y=df_y_pred["prediction"], mode="markers", name = "predictions", marker_color = "green")))
  fig_y_1.update_layout(height=800, legend=dict( yanchor="top", y=0.99, xanchor="left", x=0.01 ))         
  st.plotly_chart(fig_y_1, use_container_width=True)

  texte_modelisation_y_3 = """
  Le modèle entrainé sur le jeu de données monde (ou globale) a ensuite été appliqué sur les jeu de données des différentes hémisphères (sud et nor)
  """
  st.write(texte_modelisation_y_3)
    
  fig_y_2 = go.Figure(data = (go.Scatter(x= df_hem_N["year"], y=df_hem_N["temperature_change"], mode="lines", name = "T°C réelle, hémisphère nord", marker_color = "blue"),
          go.Scatter(x= df_hem_N["year"], y=df_hem_N["prediction"], mode="markers", name = "predictions", marker_color = "blue"),
          go.Scatter(x= df_hem_S["year"], y=df_hem_S["temperature_change"], mode="lines", name = "T°C réelle, hémisphère sud", marker_color = "red"),
          go.Scatter(x= df_hem_S["year"], y=df_hem_S["prediction"], mode="markers", name = "predictions", marker_color = "red")))
  fig_y_2.update_layout(height=800, legend=dict( yanchor="top", y=0.99, xanchor="left", x=0.01 ))         
  st.plotly_chart(fig_y_2, use_container_width=True)
          
  st.title("Prédiction des futures données de température")
  texte_modelisation_fm_1 = """
  Pour prédire les futures valeurs de données de température nous avons également testé la Régression polynomiale et le modèle ARIMA qui sont tous deux intéressants dans un contexte de modélisation de séries temporelles, comme les changements de température terrestre.
  Pour le choix du modèle, et suite aux tests de ces plusieurs algorithmes, le modèle ARIMA a été retenu pour prédire les températures jusqu'en 2050 car il obtient des résultats plus réels.
  Ce modèle est particulièrement adapté à la modélisation des données climatiques, car il permet de gérer à la fois la tendance et la saisonnalité des données.
  """
  st.write(texte_modelisation_fm_1)

  st.write("### 1. Modélisation de la Régression Polynomiale")
    
  X = df_ZonAnn_Ts_dSST[['Year']]
  y = df_ZonAnn_Ts_dSST['Glob']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  poly_degree = 3
  poly_model = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())
  poly_model.fit(X_train, y_train)
  y_poly_pred = poly_model.predict(X_test)
  mse_poly = mean_squared_error(y_test, y_poly_pred)
  rmse_poly = np.sqrt(mse_poly)
  r2_poly = r2_score(y_test, y_poly_pred)
  st.write(f'Score MSE (Régression polynomiale): {mse_poly}')
  st.write(f'Score RMSE (Régression polynomiale): {rmse_poly}')
  st.write(f'Score R² (Régression polynomiale): {r2_poly}')

  st.write("Un RMSE de 0.118 indique que, en moyenne, les prédictions du modèle s'écartent des valeurs réelles d'environ 0.12 °C.")
  
  texte_modelisation_fm_5 = """
  Malgré de meilleurs scores pour le modèle de régression polynomiale, les données prédites avec ce modèle ne sont pas réelles.
  """
  st.write(texte_modelisation_fm_5)

  X_range = np.linspace(X['Year'].min(), X['Year'].max(), 100).reshape(-1, 1)
  y_range_pred = poly_model.predict(X_range)

  fig_poly = plt.figure(figsize=(10, 6))
  plt.scatter(X_test, y_test, color='blue', label='Données Réelles')
  plt.plot(X_range, y_range_pred, color='red', label='Prédictions Polynomiales')
  plt.errorbar(X_test, y_poly_pred, yerr=rmse_poly, fmt='o', color='orange', label='Intervalle d\'Erreur (RMSE)')
  plt.title('Température Globale : Données Réelles vs Prédictions (Régression Polynomiale)')
  plt.xlabel('Année')
  plt.ylabel('Température Globale (°C)')
  plt.legend()
  plt.grid()
  st.plotly_chart(fig_poly)

  Pred_ZonAnn_Ts_dSST = pd.read_csv('Pred_ZonAnn_Ts_dSST.csv')
  Hist_ZonAnn_Ts_dSST = pd.read_csv('Hist_ZonAnn_Ts_dSST.csv')
  Resultats_ZonAnn_Ts_dSST = pd.read_csv('Resultats_ZonAnn_Ts_dSST.csv')
    
  
  st.write("### 2. Modélisation deu modèle ARIMA")

  texte_modelisation_fm_2 = """
  Nous avons mis en œuvre des techniques d'optimisation, notamment Grid Search, qui nous ont permis d'explorer différentes combinaisons de paramètres (p, d, q) pour notre modèle ARIMA. Cette approche a non seulement facilité
  l'évaluation de la performance des modèles, mais a également conduit à l'identification des meilleures prévisions pour les températures futures.
  """
  st.write(texte_modelisation_fm_2)
    
  df_ZonAnn_Ts_dSST = load_data_zonann()

  # Test de stationnarité
  result = adfuller(df_ZonAnn_Ts_dSST['Glob'])
  adf_stat = result[0]
  p_value = result[1]
  st.write(f'Statistique du test ADF : {adf_stat}')
  st.write(f'p-value: {p_value}')

  # ---- VISUALISATION Données Historiques et Prédictions Global---- #
  fig_pred = plt.figure(figsize=(12, 8))
  plt.plot(Hist_ZonAnn_Ts_dSST['Year'], Hist_ZonAnn_Ts_dSST['Glob'], label='Données Historiques', color='blue')
  plt.plot(Pred_ZonAnn_Ts_dSST['Year'], Pred_ZonAnn_Ts_dSST['Glob'], color='green', linestyle='--', label='Prédictions ARIMA')
  plt.title('Données Historiques et Prédictions ARIMA (1880 à 2050)')
  plt.xlabel('Année')
  plt.ylabel('Température Globale (°C)')
  plt.legend()
  st.pyplot(fig_pred)

  texte_modelisation_fm_3 = """
  Graphique représentant les prédictions des températures pour l’hémisphère nord, sud et le global
  """
  st.write(texte_modelisation_fm_3)

  # ---- VISUALISATION Données Historiques et Prédictions Nord, Sud et Global---- #
  df_latitude_zones = ["Glob", "NHem", "SHem"]
  fig = go.Figure()

  for column in df_latitude_zones:
    fig.add_trace(go.Scatter(
        x=Hist_ZonAnn_Ts_dSST["Year"], 
        y=Hist_ZonAnn_Ts_dSST[column], 
        mode="lines+markers", 
        name=f"Historique {column}"
        ))

  for column in df_latitude_zones:
    fig.add_trace(go.Scatter(
        x=Pred_ZonAnn_Ts_dSST["Year"], 
        y=Pred_ZonAnn_Ts_dSST[column], 
        mode="lines+markers", 
        name=f"Prédictions {column}", 
        line=dict(dash="dash")
        ))

  fig.update_layout(
    title="Données Historiques et Prédictions (1880-2050)",
    xaxis_title="Année",
    yaxis_title="Température Globale (°C)",
    hovermode="closest"
    )
  st.plotly_chart(fig)

  texte_modelisation_fm_4 = """
  Heatmap des données historiques avec prédictions des températures détaillées par hémisphères
  """
  st.write(texte_modelisation_fm_4)

  # ---- VISUALISATION DE LA HEATMAP (Nord, Équateur, Sud) ---- #
  fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

  heatmap_data = Resultats_ZonAnn_Ts_dSST[['Year', '24N-90N', '24S-24N', '90S-24S']]
  heatmap_data_melted = heatmap_data.melt(id_vars='Year', var_name='Zone', value_name='Anomalie')

  pivot_table = heatmap_data_melted.pivot_table(index="Zone", columns="Year", values="Anomalie", aggfunc='mean')
  sns.heatmap(pivot_table, cmap='RdYlBu_r', vmin=-2.60, vmax=4, annot=False, fmt=".2f", linewidths=.5, ax=axes[0])
  axes[0].set_title('Heatmap des Températures Historiques et des Prédictions Futures (Nord, Équateur, Sud)')
  axes[0].set_xlabel('Année')
  axes[0].set_ylabel('Hémisphère')

  # ---- VISUALISATION DE LA HEATMAP des Températures Historiques et des Prédictions Futures pour les Hémisphères ---- #
  heatmap_data_2 = Resultats_ZonAnn_Ts_dSST[['Year', '64N-90N', '44N-64N', '24N-44N', 'EQU-24N', '24S-EQU', '44S-24S', '64S-44S', '90S-64S']]
  heatmap_data_melted_2 = heatmap_data_2.melt(id_vars='Year', var_name='Zone', value_name='Anomalie')

  pivot_table_2 = heatmap_data_melted_2.pivot_table(index="Zone", columns="Year", values="Anomalie", aggfunc='mean')
  pivot_table_2 = pivot_table_2.reindex(['64N-90N', '44N-64N', '24N-44N',
                                       'EQU-24N', '24S-EQU',
                                       '44S-24S', '64S-44S', '90S-64S'])
  sns.heatmap(pivot_table_2, cmap='RdYlBu_r', vmin=-2.60, vmax=4, annot=False, fmt=".2f", linewidths=.5, ax=axes[1])
  axes[1].set_title('Heatmap des Températures Historiques et des Prédictions Futures pour les Hémisphères')
  axes[1].set_xlabel('Année')
  axes[1].set_ylabel('Hémisphère')

  plt.tight_layout()
  st.pyplot(fig)

  
