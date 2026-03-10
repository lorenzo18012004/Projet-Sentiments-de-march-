import streamlit as st
import yfinance as yf
import pandas as pd
from transformers import pipeline
import plotly.graph_objects as go
import plotly.express as px

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Terminal Quant & Sentiment", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Terminal Quantitatif : IA & Marché Boursier")

st.markdown("""
### 🎯 Décodons ensemble les dynamiques du marché.
Cette application croise l'**Analyse Technique** de l'historique d'une action financière avec l'**Analyse de Sentiment (NLP)** issue de la presse spécialisée. Comparer la tendance fondamentale chiffrée avec le ressenti médiatique (décrypté par l'Intelligence Artificielle FinBERT) permet d'isoler de potentiels signaux d'investissement.
""")

# --- INITIALISATION IA ---
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

with st.spinner("Initialisation du moteur d'IA FinBERT..."):
    sentiment_pipeline = load_model()


# --- SIDEBAR (PARAMÈTRES) ---
with st.sidebar:
    st.header("⚙️ Paramètres d'Analyse")
    ticker = st.text_input("Symbole Boursier (ex: AAPL, BTC-USD, TSLA, LVMH.PA) :", "AAPL").upper()
    period = st.selectbox("Historique Boursier :", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=1)
    
    st.markdown("---")
    analyze_btn = st.button("Lancer l'Analyse 🚀", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.info("💡 **Astuce** : Le sentiment s'applique aux actualités très récentes (souvent les 10-15 derniers articles), tandis que le graphique s'ajuste sur la durée choisie.")


if analyze_btn:
    if ticker:
        stock = yf.Ticker(ticker)
        
        # ==========================================
        # SECTION 1 : DONNÉES DE MARCHÉ & GRAPHIQUES
        # ==========================================
        st.header("📊 1. Analyse Technique du titre")
        
        with st.spinner(f"Récupération des marchés pour {ticker}..."):
            hist = stock.history(period=period)
            
            # Gestion du RateLimitError très fréquent sur Streamlit Cloud pour stock.info
            try:
                info = stock.info
                company_name = info.get('longName', ticker)
                currency = info.get('currency', 'USD')
            except Exception:
                # Si Yahoo bloque (Rate Limit), on utilise le ticker par défaut
                company_name = ticker
                currency = "USD"
            
        if hist.empty:
            st.error(f"❌ Impossible de trouver l'historique pour le symbole **{ticker}**. Assurez-vous que le ticker est reconnu par Yahoo Finance.")
        else:
            
            # --- KPIs boursiers ---
            current_price = hist['Close'].iloc[-1]
            previous_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            pct_change = ((current_price - previous_price) / previous_price) * 100
            
            col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
            col_kpi1.metric(f"Prix Action ({company_name})", f"{current_price:.2f} {currency}", f"{pct_change:.2f} % (1j)")
            col_kpi2.metric("Plus Haut (High)", f"{hist['High'].max():.2f} {currency}")
            col_kpi3.metric("Plus Bas (Low)", f"{hist['Low'].min():.2f} {currency}")
            col_kpi4.metric("Volume Moyen", f"{int(hist['Volume'].mean()):,}")

            # --- Graphique Candlestick Interactif ---
            fig_price = go.Figure()
            fig_price.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'],
                name="Cours",
                increasing_line_color='#26a69a', decreasing_line_color='#ef5350' # Couleurs trading pro
            ))
            
            # Formater le layout
            fig_price.update_layout(
                title=f"Action de l'entreprise : {ticker} (sur {period})", 
                yaxis_title=f"Prix ({currency})",
                xaxis_rangeslider_visible=False,
                margin=dict(l=0, r=0, t=40, b=0),
                height=400
            )
            st.plotly_chart(fig_price, use_container_width=True)
            
        st.divider()
        
        # ==========================================
        # SECTION 2 : INTELLIGENCE ARTIFICIELLE
        # ==========================================
        st.header("🧠 2. Moteur de Sentiment (Couverture Médiatique)")
        
        with st.spinner(f"Extraction et passage des textes dans les réseaux de neurones..."):
            news = stock.news
            
            if not news:
                st.warning("⚠️ Aucune actualité récente à analyser de la part des grands médias économiques.")
            else:
                results = []
                for item in news:
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
                        
                    formatted_date = date
                    if isinstance(date, (int, float)):
                        try:
                            formatted_date = pd.to_datetime(date, unit='s')
                        except:
                            pass
                    elif isinstance(date, str) and 'T' in date:
                        try:
                            formatted_date = pd.to_datetime(date)
                        except:
                            pass
                    
                    # Traitement ML : inférence FinBERT
                    analysis = sentiment_pipeline(title)[0]
                    sentiment = analysis['label']
                    confidence = analysis['score']
                    sentiment_fr = {"positive": "Positif", "negative": "Négatif", "neutral": "Neutre"}.get(sentiment.lower(), sentiment)
                    
                    results.append({
                        "Date": formatted_date,
                        "Titre de l'Article": title,
                        "Accès": link,
                        "Sentiment": sentiment_fr,
                        "Confiance (%)": round(confidence * 100, 1) # Pour la progress column
                    })
                
                if results:
                    df = pd.DataFrame(results)
                    sentiment_counts = df['Sentiment'].value_counts()
                    general_sentiment = sentiment_counts.idxmax()
                    
                    # Mise en page en 2 colonnes
                    col_chart, col_table = st.columns([1, 2.5])
                    
                    with col_chart:
                        st.subheader("Bilan des Médias")
                        
                        if general_sentiment == 'Positif':
                            st.success(f"**Dominante : POSITIF** 📈")
                        elif general_sentiment == 'Négatif':
                            st.error(f"**Dominante : NÉGATIF** 📉")
                        else:
                            st.info(f"**Dominante : NEUTRE** ➖")
                            
                        # Graphique en donut
                        fig_pie = px.pie(names=sentiment_counts.index, values=sentiment_counts.values, 
                                         color=sentiment_counts.index, hole=0.5,
                                         color_discrete_map={"Positif": "#26a69a", "Négatif": "#ef5350", "Neutre": "#9e9e9e"})
                        fig_pie.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5), margin=dict(l=0, r=0, t=10, b=0))
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                    with col_table:
                        st.subheader("Détail du Newsfeed")
                        
                        def color_sentiment(val):
                            if val == 'Positif':
                                return 'color: #26a69a; font-weight: bold;'
                            elif val == 'Négatif':
                                return 'color: #ef5350; font-weight: bold;'
                            else:
                                return 'color: #9e9e9e; font-weight: bold;'
                        
                        # Affichage du tableau interactif Pro
                        st.dataframe(
                            df.style.map(color_sentiment, subset=['Sentiment']), 
                            column_config={
                                "Date": st.column_config.DatetimeColumn("Publication", format="DD/MM/YYYY HH:mm"),
                                "Accès": st.column_config.LinkColumn("Lien Source"),
                                "Confiance (%)": st.column_config.ProgressColumn(
                                    "Fiabilité de l'IA", 
                                    help="Niveau de certitude avec lequel le réseau neuronal FinBERT a classifié cet article.",
                                    min_value=0, 
                                    max_value=100,
                                    format="%f %%"
                                )
                            },
                            hide_index=True,
                            use_container_width=True,
                            height=350
                        )
    else:
        st.warning("Veuillez saisir un symbole boursier dans la barre latérale pour démarrer.")
