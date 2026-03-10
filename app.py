import streamlit as st
import yfinance as yf
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Analyse de Sentiment Financier", 
    page_icon="📈", 
    layout="wide"
)

st.title("📈 Analyse de Sentiment Financier avec FinBERT")

st.markdown("""
### 🎯 Décodez les émotions du marché boursier en temps réel.
Cette application est une démonstration d'**Analyse de Sentiment Financier**, un outil puissant utilisé par les investisseurs quantitatifs pour automatiser la lecture des actualités et dégager une tendance de marché (haussière ou baissière).

**Le processus est simple :**
1. L'application se connecte à **Yahoo Finance** pour récupérer le flux des dernières actualités d'une action ciblée.
2. Chaque titre d'article est ensuite analysé par le modèle d'Intelligence Artificielle de pointe **FinBERT** (développé par Prosus AI), qui a été spécifiquement entraîné sur un corpus énorme de documents financiers.
3. Le modèle classe instantanément le sentiment de l'article (**Positif**, **Négatif** ou **Neutre**) et nous fournit son indice de confiance.
""")

# L'utilisation du cache permet de ne charger le modèle qu'une seule fois
@st.cache_resource
def load_model():
    # FinBERT est un modèle spécialisé dans le texte financier
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Chargement du modèle avec un indicateur visuel
with st.spinner("Chargement du modèle d'IA FinBERT... (Cela peut prendre quelques secondes la première fois pour télécharger le modèle)"):
    sentiment_pipeline = load_model()

# Entrée utilisateur pour le ticker
col1, col2 = st.columns([1, 2])
with col1:
    ticker = st.text_input("Entrez le symbole boursier (ex: AAPL, TSLA, MSFT) :", "AAPL").upper()
    analyze_btn = st.button("Analyser les actualités", type="primary")

if analyze_btn:
    if ticker:
        with st.spinner(f"Récupération des actualités pour {ticker} via yfinance..."):
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                st.warning(f"Aucune actualité récente n'a été trouvée pour le symbole {ticker}.")
            else:
                results = []
                
                # Traitement de chaque actualité
                for item in news:
                    # Extraction robuste des données avec gestion des différents formats de yfinance
                    title = ""
                    date = ""
                    link = ""
                    
                    if 'content' in item and isinstance(item['content'], dict):
                        content = item['content']
                        title = content.get('title', '')
                        date = content.get('pubDate', '')
                        click_url = content.get('clickThroughUrl', {})
                        if isinstance(click_url, dict):
                            link = click_url.get('url', '')
                    else:
                        title = item.get('title', '')
                        date = item.get('providerPublishTime', '')
                        link = item.get('link', '')
                        
                    if not title:
                        continue
                        
                    # Formatage de la date pour un affichage propre
                    formatted_date = date
                    if isinstance(date, (int, float)):
                        try:
                            # Unix timestamp
                            formatted_date = pd.to_datetime(date, unit='s').strftime('%Y-%m-%d %H:%M')
                        except:
                            formatted_date = str(date)
                    elif isinstance(date, str) and 'T' in date:
                        try:
                            # ISO format
                            formatted_date = pd.to_datetime(date).strftime('%Y-%m-%d %H:%M')
                        except:
                            formatted_date = str(date)

                    # Analyse du sentiment via le pipeline FinBERT
                    analysis = sentiment_pipeline(title)[0]
                    
                    sentiment = analysis['label']
                    confidence = analysis['score']
                    
                    # Adaptation des labels FinBERT (qui sont en anglais) pour un affichage en français
                    sentiment_fr = {"positive": "Positif", "negative": "Négatif", "neutral": "Neutre"}.get(sentiment.lower(), sentiment)
                    
                    results.append({
                        "Date": formatted_date,
                        "Titre de l'actualité": title,
                        "Lien": link,
                        "Sentiment": sentiment_fr,
                        "Confiance": f"{confidence * 100:.1f}%"
                    })
                
                if results:
                    st.success(f"Analyse de {len(results)} actualités terminée avec succès !")
                    
                    # Convertir les résultats en un DataFrame Pandas propre
                    df = pd.DataFrame(results)
                    
                    st.subheader("📊 Résultats Détaillés par article")
                    
                    # Calcul du sentiment général
                    sentiment_counts = df['Sentiment'].value_counts()
                    general_sentiment = sentiment_counts.idxmax()
                    
                    # Affichage du sentiment général de manière bien visible
                    if general_sentiment == 'Positif':
                        st.success(f"### Sentiment Général : Positif 🟢\nLa tendance globale des actualités est à l'optimisme.")
                    elif general_sentiment == 'Négatif':
                        st.error(f"### Sentiment Général : Négatif 🔴\nLa tendance globale des actualités est au pessimisme.")
                    else:
                        st.info(f"### Sentiment Général : Neutre ⚪\nLes actualités sont globalement factuelles ou sans tendance claire.")
                        
                    st.write("") # Espace
                    
                    # Colorisation des cellules du DataFrame selon le sentiment
                    def color_sentiment(val):
                        if val == 'Positif':
                            return 'color: #28a745; font-weight: bold;'
                        elif val == 'Négatif':
                            return 'color: #dc3545; font-weight: bold;'
                        else:
                            return 'color: #6c757d; font-weight: bold;'
                    
                    # Affichage du tableau interactif avec configuration des colonnes
                    st.dataframe(
                        df.style.map(color_sentiment, subset=['Sentiment']), 
                        column_config={
                            "Date": st.column_config.DatetimeColumn("Date de publication", format="DD/MM/YYYY HH:mm"),
                            "Lien": st.column_config.LinkColumn("Accéder à l'article"),
                        },
                        hide_index=True,
                        use_container_width=True,
                        height=250
                    )
                    
                    st.divider()
                    
                    # --- Section Graphique ---
                    st.subheader("📉 Répartition des Sentiments")
                    
                    # Comptage des sentiments
                    sentiment_counts = df['Sentiment'].value_counts().reindex(["Positif", "Négatif", "Neutre"], fill_value=0)
                    
                    # Code couleur pour le graphique
                    color_map = {"Positif": "#28a745", "Négatif": "#dc3545", "Neutre": "#6c757d"}
                    
                    # Création du bar chart avec Matplotlib pour un meilleur contrôle des couleurs
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.bar(
                        sentiment_counts.index, 
                        sentiment_counts.values, 
                        color=[color_map[x] for x in sentiment_counts.index]
                    )
                    
                    # Ajouter les valeurs exactes au-dessus des barres
                    for bar in bars:
                        yval = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, int(yval), ha='center', va='bottom', fontsize=12)
                        
                    ax.set_ylabel("Nombre d'articles", fontsize=11)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.tick_params(axis='both', which='major', labelsize=11)
                    
                    # Afficher le graphique sur Streamlit
                    col_chart, _ = st.columns([2, 1])
                    with col_chart:
                        st.pyplot(fig)
                    
                else:
                    st.warning("Impossible d'extraire les titres des actualités.")
    else:
        st.error("Veuillez entrer un symbole boursier valide pour lancer l'analyse.")
