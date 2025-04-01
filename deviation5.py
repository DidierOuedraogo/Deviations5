import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO

# Initialiser l'état de session pour le suivi de l'entraînement des modèles
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model_azimuth' not in st.session_state:
    st.session_state.model_azimuth = None
if 'model_inclinaison' not in st.session_state:
    st.session_state.model_inclinaison = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'columns_mapped' not in st.session_state:
    st.session_state.columns_mapped = False

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Déviation des Forages Miniers",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un look moderne
st.markdown("""
<style>
    /* Couleurs personnalisées */
    :root {
        --primary: #4F8BF9;
        --secondary: #1E293B;
        --accent: #FF4B4B;
        --background: #F8F9FA;
        --text: #1E293B;
        --light-text: #64748B;
        --card: white;
        --success: #0CCE6B;
    }
    
    /* Corps du document */
    .main {
        background-color: var(--background);
        color: var(--text);
        font-family: 'Roboto', sans-serif;
    }
    
    /* En-têtes */
    h1, h2, h3, h4 {
        color: var(--secondary);
        font-weight: 700;
    }
    
    h1 {
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f0f0;
    }
    
    h2 {
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #f0f0f0;
    }
    
    h3 {
        font-size: 1.4rem;
        margin-top: 1.5rem;
    }
    
    /* Cards */
    .stDataFrame, .css-1r6slb0, div[data-testid="stBlock"] {
        background-color: var(--card);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Widgets */
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #3A7BD5;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    .stSidebar .stButton>button {
        width: 100%;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 6px 6px 0 0;
        padding: 0 20px;
        font-weight: 500;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1wrcr25 {
        background-color: var(--secondary);
    }
    
    .css-1d391kg .css-163ttbj, .css-1wrcr25 .css-163ttbj {
        color: white;
    }
    
    .css-1d391kg label, .css-1wrcr25 label {
        color: #E2E8F0;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--primary) !important;
    }
    
    /* Author Banner */
    .author-banner {
        background-color: var(--secondary);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .author-info {
        display: flex;
        flex-direction: column;
    }
    
    .author-name {
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .author-title {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
    }
    
    .status-trained {
        background-color: var(--success);
    }
    
    .status-untrained {
        background-color: var(--accent);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #E6F0FF;
        border-left: 4px solid var(--primary);
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    
    /* Progress bar */
    .stProgress .st-bo {
        background-color: var(--primary);
    }
    
    /* Plot styling */
    .js-plotly-plot {
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        background-color: white;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Titre et auteur
st.markdown("""
<div class="author-banner">
    <div class="author-info">
        <span class="author-name">Prédiction de Déviation des Forages Miniers</span>
        <span class="author-title">Application de machine learning pour anticiper les déviations</span>
    </div>
    <div class="author-info" style="text-align: right;">
        <span class="author-name">Didier Ouedraogo, P.Geo.</span>
        <span class="author-title">Géologue & Data Scientist</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Fonction pour créer un icône
def get_icon_html(icon_name, color="white", size=24):
    return f'<i class="material-icons" style="color: {color}; font-size: {size}px;">{icon_name}</i>'

# Fonction pour charger les données
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Sidebar pour les options
with st.sidebar:
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <h3 style="margin: 0; color: white;">Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Statut du modèle
    model_status = "trained" if st.session_state.model_trained else "untrained"
    columns_status = "mapped" if st.session_state.columns_mapped else "unmapped"
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 1.5rem; background-color: #2D3748; padding: 0.8rem; border-radius: 6px;">
        <span class="status-indicator status-{model_status}"></span>
        <span style="color: white; font-size: 0.9rem;">Statut du modèle: {'Entraîné' if st.session_state.model_trained else 'Non entraîné'}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Option pour uploader les données ou utiliser des données de démonstration
    st.markdown('<p style="color: #E2E8F0; font-weight: 600; margin-bottom: 0.5rem;">Source des données</p>', unsafe_allow_html=True)
    data_option = st.radio(
        "",
        ["Charger mes données", "Utiliser données démo"],
        label_visibility="collapsed"
    )
    
    if data_option == "Charger mes données":
        uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
        
        if uploaded_file is not None and st.session_state.raw_df is None:
            # Charger les données brutes
            st.session_state.raw_df = load_data(uploaded_file)
            st.session_state.columns_mapped = False
    
    # Séparateur visuel
    st.markdown('<hr style="margin: 1.5rem 0; border-color: #4A5568;">', unsafe_allow_html=True)
    
    # Sélection du modèle
    st.markdown('<p style="color: #E2E8F0; font-weight: 600; margin-bottom: 0.5rem;">Modèle de machine learning</p>', unsafe_allow_html=True)
    model_option = st.selectbox(
        "",
        ["Random Forest", "SVM", "Régression Linéaire", "Réseau de Neurones"],
        label_visibility="collapsed"
    )
    
    # Bouton d'entraînement
    train_button = st.button("Entraîner le modèle")

# Initialisation des données
df = None

if data_option == "Charger mes données" and st.session_state.raw_df is not None and not st.session_state.columns_mapped:
    st.markdown("## Mappage des colonnes")
    st.markdown("""
    <div class="info-box">
        <b>Mappage requis</b>: Veuillez associer les colonnes de votre fichier CSV aux colonnes attendues par l'application.
    </div>
    """, unsafe_allow_html=True)
    
    # Afficher un aperçu des données brutes
    st.markdown("### Aperçu de vos données")
    st.dataframe(st.session_state.raw_df.head(), use_container_width=True)
    
    # Colonnes requises par l'application
    required_columns = {
        'profondeur_finale': 'Profondeur finale du forage (mètres)',
        'azimuth_initial': 'Azimuth initial du forage (degrés)',
        'inclinaison_initiale': 'Inclinaison initiale du forage (degrés)',
        'lithologie': 'Type de roche traversée',
        'vitesse_rotation': 'Vitesse de rotation de la tige (tr/min)',
        'deviation_azimuth': 'Déviation mesurée en azimuth (degrés)',
        'deviation_inclinaison': 'Déviation mesurée en inclinaison (degrés)'
    }
    
    st.markdown("### Associer les colonnes")
    st.markdown("""
    <p>Pour chaque paramètre requis, sélectionnez la colonne correspondante dans votre fichier CSV.</p>
    """, unsafe_allow_html=True)
    
    # Créer un dictionnaire pour stocker les mappages
    column_mapping = {}
    
    # Créer une liste des colonnes disponibles dans le CSV
    available_columns = st.session_state.raw_df.columns.tolist()
    
    # Ajouter une option "Non disponible" pour les colonnes facultatives
    available_columns_with_na = ['Non disponible'] + available_columns
    
    # Créer des sélecteurs pour chaque colonne requise
    col1, col2 = st.columns(2)
    
    with col1:
        for i, (required_col, description) in enumerate(list(required_columns.items())[:4]):
            # Suggérer une correspondance basée sur des mots-clés
            suggested_index = 0  # Par défaut "Non disponible"
            for j, col in enumerate(available_columns):
                if required_col.lower() in col.lower() or any(word in col.lower() for word in required_col.split('_')):
                    suggested_index = j + 1  # +1 car nous avons ajouté "Non disponible" en première position
                    break
            
            column_mapping[required_col] = st.selectbox(
                f"{description}",
                available_columns_with_na,
                index=suggested_index,
                help=f"Sélectionnez la colonne de votre CSV qui correspond à '{required_col}'"
            )
    
    with col2:
        for i, (required_col, description) in enumerate(list(required_columns.items())[4:]):
            # Suggérer une correspondance basée sur des mots-clés
            suggested_index = 0  # Par défaut "Non disponible"
            for j, col in enumerate(available_columns):
                if required_col.lower() in col.lower() or any(word in col.lower() for word in required_col.split('_')):
                    suggested_index = j + 1  # +1 car nous avons ajouté "Non disponible" en première position
                    break
            
            column_mapping[required_col] = st.selectbox(
                f"{description}",
                available_columns_with_na,
                index=suggested_index,
                help=f"Sélectionnez la colonne de votre CSV qui correspond à '{required_col}'"
            )
    
    # Vérifier si toutes les colonnes obligatoires sont mappées
    missing_required = [col for col, mapped in column_mapping.items() 
                       if mapped == 'Non disponible' and col not in ['lithologie']]
    
    if len(missing_required) > 0:
        st.warning(f"⚠️ Certaines colonnes obligatoires n'ont pas été mappées: {', '.join(missing_required)}")
        can_proceed = False
    else:
        can_proceed = True
    
    # Bouton pour valider le mappage
    mapping_col1, mapping_col2, mapping_col3 = st.columns([1, 2, 1])
    with mapping_col2:
        if st.button("Valider le mappage", disabled=not can_proceed, use_container_width=True):
            # Créer un nouveau DataFrame avec les colonnes mappées
            mapped_df = pd.DataFrame()
            
            for required_col, source_col in column_mapping.items():
                if source_col != 'Non disponible':
                    mapped_df[required_col] = st.session_state.raw_df[source_col]
                else:
                    # Si la colonne est facultative, on peut générer des valeurs par défaut
                    if required_col == 'lithologie':
                        mapped_df[required_col] = 'Inconnu'  # Valeur par défaut pour la lithologie
            
            # Stocker le DataFrame mappé dans la session
            st.session_state.df = mapped_df
            st.session_state.columns_mapped = True
            st.success("✅ Mappage validé! Vous pouvez maintenant explorer et modéliser vos données.")
            st.experimental_rerun()

elif data_option == "Charger mes données" and st.session_state.columns_mapped:
    # Utiliser le DataFrame déjà mappé
    df = st.session_state.df
    
elif data_option == "Utiliser données démo":
    # Données de démonstration
    st.markdown("""
    <div class="info-box">
        <b>Mode démo</b>: Utilisation de données synthétiques pour illustrer le fonctionnement de l'application.
    </div>
    """, unsafe_allow_html=True)
    
    # Créer des données synthétiques pour la démonstration
    np.random.seed(42)
    n_samples = 1000
    
    prof_finale = np.random.uniform(100, 1000, n_samples)
    azimuth_initial = np.random.uniform(0, 360, n_samples)
    inclinaison_initiale = np.random.uniform(-90, 0, n_samples)
    vitesse_rotation = np.random.uniform(50, 200, n_samples)
    
    # Lithologies possibles
    lithologies = ['Granite', 'Schiste', 'Gneiss', 'Calcaire', 'Basalte']
    lithologie = np.random.choice(lithologies, n_samples)
    
    # Créer une relation entre les entrées et les déviations (simplifiée)
    azimuth_deviation = (
        0.05 * prof_finale 
        + 0.02 * azimuth_initial 
        + 0.1 * inclinaison_initiale 
        + 0.03 * vitesse_rotation 
        + np.random.normal(0, 10, n_samples)
    )
    
    inclinaison_deviation = (
        0.03 * prof_finale 
        - 0.01 * azimuth_initial 
        + 0.05 * inclinaison_initiale 
        + 0.02 * vitesse_rotation 
        + np.random.normal(0, 5, n_samples)
    )
    
    # Ajouter un effet de la lithologie (différent pour chaque type)
    lithology_effect = {
        'Granite': (2.0, 1.0),
        'Schiste': (-1.5, 3.0),
        'Gneiss': (0.5, -2.0),
        'Calcaire': (-1.0, -1.5),
        'Basalte': (3.0, 2.5)
    }
    
    for i, lith in enumerate(lithologie):
        effect_az, effect_inc = lithology_effect[lith]
        azimuth_deviation[i] += effect_az
        inclinaison_deviation[i] += effect_inc
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'profondeur_finale': prof_finale,
        'azimuth_initial': azimuth_initial,
        'inclinaison_initiale': inclinaison_initiale,
        'lithologie': lithologie,
        'vitesse_rotation': vitesse_rotation,
        'deviation_azimuth': azimuth_deviation,
        'deviation_inclinaison': inclinaison_deviation
    })
    
    # Stocker dans la session state
    st.session_state.df = df
    st.session_state.columns_mapped = True

# Si des données sont disponibles, afficher l'application principale
if df is not None:
    # Onglets pour les différentes sections
    tabs = st.tabs(["📊 Exploration", "🧠 Modélisation", "🔮 Prédiction"])
    
    with tabs[0]:  # Exploration des données
        st.markdown("## Exploration des données")
        
        # Affichage des données en deux colonnes
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Aperçu des données")
            st.dataframe(df.head(), use_container_width=True)
        
        with col2:
            st.markdown("### Statistiques descriptives")
            st.dataframe(df.describe().style.highlight_max(axis=0), use_container_width=True)
        
        # Distribution des lithologies et métriques globales
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### Distribution des lithologies")
            fig_litho = px.histogram(df, x='lithologie', color='lithologie', 
                                    color_discrete_sequence=px.colors.qualitative.Bold,
                                    template="plotly_white")
            fig_litho.update_layout(
                xaxis_title="Lithologie",
                yaxis_title="Nombre de forages",
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_litho, use_container_width=True)
            
        with col2:
            st.markdown("### Métriques globales")
            
            # Calculer des métriques intéressantes
            mean_az_dev = df['deviation_azimuth'].abs().mean()
            max_az_dev = df['deviation_azimuth'].abs().max()
            mean_inc_dev = df['deviation_inclinaison'].abs().mean()
            max_inc_dev = df['deviation_inclinaison'].abs().max()
            
            # Afficher les métriques
            c1, c2 = st.columns(2)
            c1.metric("Déviation moyenne d'azimuth", f"{mean_az_dev:.2f}°")
            c2.metric("Déviation max. d'azimuth", f"{max_az_dev:.2f}°")
            
            c1, c2 = st.columns(2)
            c1.metric("Déviation moyenne d'inclinaison", f"{mean_inc_dev:.2f}°")
            c2.metric("Déviation max. d'inclinaison", f"{max_inc_dev:.2f}°")
            
            # Calculer la lithologie avec la déviation la plus importante
            lithology_deviation = df.groupby('lithologie')[['deviation_azimuth', 'deviation_inclinaison']].apply(
                lambda x: (x['deviation_azimuth']**2 + x['deviation_inclinaison']**2).mean()**0.5
            ).sort_values(ascending=False)
            
            most_deviated = lithology_deviation.index[0]
            deviation_value = lithology_deviation.iloc[0]
            
            # Définir la couleur pour la lithologie la plus déviée
            lithology_colors = {
                'Granite': '#FF6B6B', 
                'Schiste': '#4ECDC4', 
                'Gneiss': '#45B7D1', 
                'Calcaire': '#FFBE0B', 
                'Basalte': '#9F84BD'
            }
            color = lithology_colors.get(most_deviated, '#4F8BF9')
            
            st.markdown(f"""
            <div style="margin-top: 1rem; background-color: #F8F9FA; padding: 1rem; border-radius: 6px; border-left: 4px solid {color};">
                <p style="margin: 0; font-size: 0.9rem;">Lithologie avec le plus de déviation :</p>
                <p style="margin: 0; font-weight: 600; font-size: 1.1rem;">{most_deviated} ({deviation_value:.2f}°)</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Analyse exploratoire détaillée
        st.markdown("### Analyse exploratoire approfondie")
        
        explore_tabs = st.tabs(["Corrélations", "Déviations par lithologie", "Relations"])
        
        with explore_tabs[0]:
            # Matrice de corrélation
            numeric_cols = df.select_dtypes(include=np.number).columns
            corr_matrix = df[numeric_cols].corr()
            
            fig_corr = px.imshow(corr_matrix, 
                                text_auto=True, 
                                color_continuous_scale='RdBu_r',
                                title="Matrice de corrélation",
                                template="plotly_white")
            fig_corr.update_layout(
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Interprétation automatique des corrélations
            strong_correlations = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if abs(corr_matrix.iloc[i, j]) > 0.3:  # Seuil de corrélation
                        strong_correlations.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'corr': corr_matrix.iloc[i, j]
                        })
            
            if strong_correlations:
                st.markdown("#### Corrélations significatives")
                for corr in sorted(strong_correlations, key=lambda x: abs(x['corr']), reverse=True):
                    relation = "positive" if corr['corr'] > 0 else "négative"
                    strength = "forte" if abs(corr['corr']) > 0.7 else "modérée"
                    st.markdown(f"- Corrélation {strength} {relation} ({corr['corr']:.2f}) entre **{corr['var1']}** et **{corr['var2']}**")
        
        with explore_tabs[1]:
            # Déviations par lithologie
            col1, col2 = st.columns(2)
            
            with col1:
                fig_box1 = px.box(df, x='lithologie', y='deviation_azimuth', 
                                title="Déviation d'azimuth par lithologie", 
                                color='lithologie',
                                color_discrete_sequence=px.colors.qualitative.Bold,
                                template="plotly_white")
                fig_box1.update_layout(
                    xaxis_title="Lithologie",
                    yaxis_title="Déviation d'azimuth (°)",
                    showlegend=False,
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_box1, use_container_width=True)
            
            with col2:
                fig_box2 = px.box(df, x='lithologie', y='deviation_inclinaison', 
                                title="Déviation d'inclinaison par lithologie", 
                                color='lithologie',
                                color_discrete_sequence=px.colors.qualitative.Bold,
                                template="plotly_white")
                fig_box2.update_layout(
                    xaxis_title="Lithologie",
                    yaxis_title="Déviation d'inclinaison (°)",
                    showlegend=False,
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_box2, use_container_width=True)
            
            # Résumé statistique par lithologie
            st.markdown("#### Résumé statistique par lithologie")
            
            litho_stats = df.groupby('lithologie')[['deviation_azimuth', 'deviation_inclinaison']].agg(
                ['mean', 'std', 'min', 'max']
            ).round(2)
            
            litho_stats.columns = ['Azimuth Moy', 'Azimuth Std', 'Azimuth Min', 'Azimuth Max', 
                                   'Inclinaison Moy', 'Inclinaison Std', 'Inclinaison Min', 'Inclinaison Max']
            
            st.dataframe(litho_stats, use_container_width=True)
        
        with explore_tabs[2]:
            # Relations entre paramètres et déviations
            features = ['profondeur_finale', 'azimuth_initial', 'inclinaison_initiale', 'vitesse_rotation']
            
            selected_feature = st.selectbox(
                "Sélectionner un paramètre pour explorer sa relation avec les déviations",
                features
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_scatter1 = px.scatter(df, x=selected_feature, y='deviation_azimuth', 
                                        color='lithologie', opacity=0.7,
                                        title=f"Déviation d'azimuth vs {selected_feature}",
                                        color_discrete_sequence=px.colors.qualitative.Bold,
                                        trendline="ols",
                                        template="plotly_white")
                fig_scatter1.update_layout(
                    xaxis_title=selected_feature,
                    yaxis_title="Déviation d'azimuth (°)",
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_scatter1, use_container_width=True)
            
            with col2:
                fig_scatter2 = px.scatter(df, x=selected_feature, y='deviation_inclinaison', 
                                        color='lithologie', opacity=0.7,
                                        title=f"Déviation d'inclinaison vs {selected_feature}",
                                        color_discrete_sequence=px.colors.qualitative.Bold,
                                        trendline="ols",
                                        template="plotly_white")
                fig_scatter2.update_layout(
                    xaxis_title=selected_feature,
                    yaxis_title="Déviation d'inclinaison (°)",
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_scatter2, use_container_width=True)
            
            # Distribution du paramètre sélectionné
            fig_hist = px.histogram(df, x=selected_feature, color='lithologie',
                                   title=f"Distribution de {selected_feature}",
                                   color_discrete_sequence=px.colors.qualitative.Bold,
                                   marginal="box",
                                   template="plotly_white")
            fig_hist.update_layout(
                xaxis_title=selected_feature,
                yaxis_title="Nombre de forages",
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with tabs[1]:  # Modélisation
        st.markdown("## Modélisation des déviations")
        
        # Définir les caractéristiques et cibles
        X = df[['profondeur_finale', 'azimuth_initial', 'inclinaison_initiale', 'lithologie', 'vitesse_rotation']]
        y_azimuth = df['deviation_azimuth']
        y_inclinaison = df['deviation_inclinaison']
        
        # Préparation pour la modélisation
        numeric_features = ['profondeur_finale', 'azimuth_initial', 'inclinaison_initiale', 'vitesse_rotation']
        categorical_features = ['lithologie']
        
        # Préprocesseurs
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Choix du modèle
        if model_option == "Random Forest":
            model_azimuth = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            
            model_inclinaison = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            
        elif model_option == "SVM":
            model_azimuth = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', SVR())
            ])
            
            model_inclinaison = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', SVR())
            ])
            
        elif model_option == "Régression Linéaire":
            model_azimuth = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
            ])
            
            model_inclinaison = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
            ])
            
        else:  # Réseau de Neurones
            model_azimuth = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', MLPRegressor(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42))
            ])
            
            model_inclinaison = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', MLPRegressor(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42))
            ])
        
        # Diviser les données
        X_train, X_test, y_azimuth_train, y_azimuth_test, y_inclinaison_train, y_inclinaison_test = train_test_split(
            X, y_azimuth, y_inclinaison, test_size=0.2, random_state=42
        )
        
        # Description du modèle sélectionné
        model_descriptions = {
            "Random Forest": """
                **Random Forest** est un algorithme d'ensemble qui utilise plusieurs arbres de décision pour améliorer 
                la précision et réduire le surapprentissage. Il est efficace pour capturer les relations non linéaires 
                dans les données.
                
                **Avantages**:
                - Bonne performance sur les données complexes
                - Gère bien les valeurs manquantes
                - Fournit des mesures d'importance des variables
                
                **Complexité du modèle**: Moyenne à élevée
            """,
            "SVM": """
                **Support Vector Machine (SVM)** est un algorithme qui trouve un hyperplan optimal pour séparer les données.
                Dans sa version régression (SVR), il cherche à trouver une fonction qui s'écarte le moins possible des points.
                
                **Avantages**:
                - Efficace dans les espaces de grande dimension
                - Polyvalent grâce aux différents noyaux
                - Bonne capacité de généralisation
                
                **Complexité du modèle**: Moyenne
            """,
            "Régression Linéaire": """
                **Régression Linéaire** est un modèle simple qui établit une relation linéaire entre les variables 
                d'entrée et la sortie. Il est facile à interpréter mais limité pour capturer des relations complexes.
                
                **Avantages**:
                - Simple et interprétable
                - Rapide à entraîner
                - Faible variance
                
                **Complexité du modèle**: Faible
            """,
            "Réseau de Neurones": """
                **Réseau de Neurones** est un modèle inspiré du cerveau humain, composé de couches de neurones artificiels.
                Il peut modéliser des relations très complexes et non linéaires.
                
                **Avantages**:
                - Capacité à modéliser des relations très complexes
                - Peut apprendre des représentations hiérarchiques
                - Très flexible
                
                **Complexité du modèle**: Élevée
            """
        }
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### Modèle sélectionné: {model_option}")
            st.markdown(model_descriptions[model_option])
        
        with col2:
            st.markdown("### Configuration")
            st.markdown(f"""
            <div style="background-color: #F8F9FA; padding: 1rem; border-radius: 6px; margin-bottom: 1rem;">
                <p style="margin: 0; font-weight: 600;">Répartition des données</p>
                <p style="margin: 0;">80% Entraînement / 20% Test</p>
            </div>
            
            <div style="background-color: #F8F9FA; padding: 1rem; border-radius: 6px;">
                <p style="margin: 0; font-weight: 600;">Variables d'entrée</p>
                <p style="margin: 0;">- Profondeur finale</p>
                <p style="margin: 0;">- Azimuth initial</p>
                <p style="margin: 0;">- Inclinaison initiale</p>
                <p style="margin: 0;">- Lithologie</p>
                <p style="margin: 0;">- Vitesse de rotation</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Entraîner les modèles si l'utilisateur clique sur le bouton
        if train_button:
            st.markdown("### Entraînement en cours...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Étape 1: Préparation des données
            status_text.text("Préparation des données...")
            progress_bar.progress(20)
            
            # Étape 2: Entraînement du modèle d'azimuth
            status_text.text("Entraînement du modèle pour la déviation d'azimuth...")
            model_azimuth.fit(X_train, y_azimuth_train)
            y_azimuth_pred = model_azimuth.predict(X_test)
            progress_bar.progress(50)
            
            # Étape 3: Entraînement du modèle d'inclinaison
            status_text.text("Entraînement du modèle pour la déviation d'inclinaison...")
            model_inclinaison.fit(X_train, y_inclinaison_train)
            y_inclinaison_pred = model_inclinaison.predict(X_test)
            progress_bar.progress(80)
            
            # Étape 4: Évaluation des performances
            status_text.text("Évaluation des performances...")
            azimuth_rmse = np.sqrt(mean_squared_error(y_azimuth_test, y_azimuth_pred))
            azimuth_r2 = r2_score(y_azimuth_test, y_azimuth_pred)
            
            inclinaison_rmse = np.sqrt(mean_squared_error(y_inclinaison_test, y_inclinaison_pred))
            inclinaison_r2 = r2_score(y_inclinaison_test, y_inclinaison_pred)
            progress_bar.progress(100)
            
            # Stocker les modèles dans la session state
            st.session_state.model_azimuth = model_azimuth
            st.session_state.model_inclinaison = model_inclinaison
            st.session_state.model_trained = True
            
            status_text.text("Entraînement terminé!")
            
            # Affichage des résultats
            st.markdown("### Résultats de l'entraînement")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Déviation d'azimuth")
                
                # Métrique avec évaluation de la performance
                r2_color = '#0CCE6B' if azimuth_r2 > 0.7 else ('#FFA500' if azimuth_r2 > 0.5 else '#FF4B4B')
                
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                    <div style="background-color: #F8F9FA; padding: 1rem; border-radius: 6px; width: 48%;">
                        <p style="margin: 0; font-size: 0.9rem;">RMSE</p>
                        <p style="margin: 0; font-weight: 600; font-size: 1.5rem; color: #4F8BF9;">{azimuth_rmse:.4f}°</p>
                    </div>
                    <div style="background-color: #F8F9FA; padding: 1rem; border-radius: 6px; width: 48%;">
                        <p style="margin: 0; font-size: 0.9rem;">R²</p>
                        <p style="margin: 0; font-weight: 600; font-size: 1.5rem; color: {r2_color};">{azimuth_r2:.4f}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Graphique des prédictions vs réelles
                fig_pred_az = px.scatter(x=y_azimuth_test, y=y_azimuth_pred, 
                                        labels={'x': 'Valeurs réelles (°)', 'y': 'Prédictions (°)'},
                                        title="Prédictions vs Réelles - Déviation d'azimuth",
                                        template="plotly_white")
                fig_pred_az.add_shape(type='line', line=dict(dash='dash', color='rgba(0,0,0,0.3)'),
                                    x0=y_azimuth_test.min(), y0=y_azimuth_test.min(),
                                    x1=y_azimuth_test.max(), y1=y_azimuth_test.max())
                fig_pred_az.update_layout(
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_pred_az, use_container_width=True)
            
            with col2:
                st.markdown("#### Déviation d'inclinaison")
                
                # Métrique avec évaluation de la performance
                r2_color = '#0CCE6B' if inclinaison_r2 > 0.7 else ('#FFA500' if inclinaison_r2 > 0.5 else '#FF4B4B')
                
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                    <div style="background-color: #F8F9FA; padding: 1rem; border-radius: 6px; width: 48%;">
                        <p style="margin: 0; font-size: 0.9rem;">RMSE</p>
                        <p style="margin: 0; font-weight: 600; font-size: 1.5rem; color: #4F8BF9;">{inclinaison_rmse:.4f}°</p>
                    </div>
                    <div style="background-color: #F8F9FA; padding: 1rem; border-radius: 6px; width: 48%;">
                        <p style="margin: 0; font-size: 0.9rem;">R²</p>
                        <p style="margin: 0; font-weight: 600; font-size: 1.5rem; color: {r2_color};">{inclinaison_r2:.4f}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Graphique des prédictions vs réelles
                fig_pred_inc = px.scatter(x=y_inclinaison_test, y=y_inclinaison_pred, 
                                        labels={'x': 'Valeurs réelles (°)', 'y': 'Prédictions (°)'},
                                        title="Prédictions vs Réelles - Déviation d'inclinaison",
                                        template="plotly_white")
                fig_pred_inc.add_shape(type='line', line=dict(dash='dash', color='rgba(0,0,0,0.3)'),
                                    x0=y_inclinaison_test.min(), y0=y_inclinaison_test.min(),
                                    x1=y_inclinaison_test.max(), y1=y_inclinaison_test.max())
                fig_pred_inc.update_layout(
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_pred_inc, use_container_width=True)
            
            # Interprétation des résultats
            st.markdown("### Interprétation des résultats")
            
            # Calculer la performance moyenne
            avg_r2 = (azimuth_r2 + inclinaison_r2) / 2
            
            if avg_r2 > 0.8:
                performance_text = "excellente"
                performance_detail = """
                    Le modèle capture très bien les facteurs influençant les déviations. Vous pouvez utiliser ces prédictions 
                    avec un haut niveau de confiance pour la planification des forages.
                """
            elif avg_r2 > 0.7:
                performance_text = "bonne"
                performance_detail = """
                    Le modèle capture bien les tendances principales des déviations. Les prédictions sont fiables pour
                    la plupart des conditions de forage.
                """
            elif avg_r2 > 0.5:
                performance_text = "modérée"
                performance_detail = """
                    Le modèle capture les tendances générales mais manque de précision dans certains cas. Utilisez les 
                    prédictions comme indicateurs mais prévoyez des marges de sécurité.
                """
            else:
                performance_text = "limitée"
                performance_detail = """
                    Le modèle a du mal à capturer la complexité des facteurs influençant les déviations. Les prédictions
                    doivent être utilisées avec prudence et des facteurs supplémentaires pourraient être nécessaires.
                """
            
            st.markdown(f"""
            <div style="background-color: #F8F9FA; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
                <h4 style="margin-top: 0;">Performance globale: {performance_text.capitalize()}</h4>
                <p>{performance_detail}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Si Random Forest, afficher l'importance des caractéristiques
            if model_option == "Random Forest":
                st.markdown("### Importance des caractéristiques")
                
                # Extraire l'importance des caractéristiques pour l'azimuth
                rf_azimuth = model_azimuth.named_steps['regressor']
                preprocessor_azimuth = model_azimuth.named_steps['preprocessor']
                
                # Obtenir les noms des caractéristiques après transformation
                cat_features = preprocessor_azimuth.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
                feature_names = np.concatenate([numeric_features, cat_features])
                
                # Obtenir l'importance des caractéristiques
                feature_importance_azimuth = rf_azimuth.feature_importances_
                
                # Pour l'inclinaison
                rf_inclinaison = model_inclinaison.named_steps['regressor']
                feature_importance_inclinaison = rf_inclinaison.feature_importances_
                
                # Créer un DataFrame pour l'affichage
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance_Azimuth': feature_importance_azimuth,
                    'Importance_Inclinaison': feature_importance_inclinaison
                })
                
                # Afficher sous forme de graphique
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_imp_az = px.bar(
                        importance_df.sort_values('Importance_Azimuth', ascending=False),
                        y='Feature', x='Importance_Azimuth',
                        title="Importance des facteurs - Déviation d'azimuth",
                        template="plotly_white"
                    )
                    fig_imp_az.update_layout(
                        yaxis_title="",
                        xaxis_title="Importance relative",
                        margin=dict(l=20, r=20, t=50, b=20),
                    )
                    st.plotly_chart(fig_imp_az, use_container_width=True)
                
                with col2:
                    fig_imp_inc = px.bar(
                        importance_df.sort_values('Importance_Inclinaison', ascending=False),
                        y='Feature', x='Importance_Inclinaison',
                        title="Importance des facteurs - Déviation d'inclinaison",
                        template="plotly_white"
                    )
                    fig_imp_inc.update_layout(
                        yaxis_title="",
                        xaxis_title="Importance relative",
                        margin=dict(l=20, r=20, t=50, b=20),
                    )
                    st.plotly_chart(fig_imp_inc, use_container_width=True)
    
    with tabs[2]:  # Prédiction
        st.markdown("## Prédiction pour un nouveau forage")
        
        # Vérification si un modèle est entraîné
        if not st.session_state.model_trained:
            st.markdown("""
            <div style="background-color: #FEF2F2; border-left: 4px solid #FF4B4B; padding: 1rem; border-radius: 4px; margin-bottom: 1.5rem;">
                <p style="margin: 0; color: #7F1D1D; font-weight: 600;">Modèle non entraîné</p>
                <p style="margin: 0; color: #7F1D1D;">Veuillez d'abord entraîner un modèle depuis l'onglet "Modélisation" ou la barre latérale.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Formulaire de prédiction
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Paramètres du forage")
            
            prof_finale_input = st.number_input("Profondeur finale (m)", min_value=50.0, max_value=2000.0, value=500.0, step=50.0)
            azimuth_initial_input = st.number_input("Azimuth initial (degrés)", min_value=0.0, max_value=360.0, value=90.0, step=5.0)
            inclinaison_initiale_input = st.number_input("Inclinaison initiale (degrés)", min_value=-90.0, max_value=0.0, value=-45.0, step=5.0)
            
            lithologies_uniques = df['lithologie'].unique().tolist()
            lithologie_input = st.selectbox("Lithologie", lithologies_uniques)
            
            vitesse_rotation_input = st.number_input("Vitesse de rotation (tr/min)", min_value=20.0, max_value=300.0, value=120.0, step=10.0)
        
        with col2:
            st.markdown("### Illustration schématique")
            
            # Visualisation schématique du forage initial
            def generate_drill_illustration(azimuth, inclination):
                fig = go.Figure()
                
                # Calculer les coordonnées pour une représentation simple
                depth = 100
                x = depth * np.cos(np.radians(inclination)) * np.sin(np.radians(azimuth))
                y = depth * np.cos(np.radians(inclination)) * np.cos(np.radians(azimuth))
                z = depth * np.sin(np.radians(inclination))
                
                # Surface (grille)
                x_surface = np.linspace(-100, 100, 5)
                y_surface = np.linspace(-100, 100, 5)
                z_surface = np.zeros((5, 5))
                
                fig.add_trace(go.Surface(x=x_surface, y=y_surface, z=z_surface, 
                                        colorscale=[[0, 'lightgreen'], [1, 'lightgreen']],
                                        showscale=False, opacity=0.3))
                
                # Point de départ du forage
                fig.add_trace(go.Scatter3d(
                    x=[0], y=[0], z=[0],
                    mode='markers',
                    marker=dict(size=10, color='green'),
                    name='Départ'
                ))
                
                # Direction initiale
                fig.add_trace(go.Scatter3d(
                    x=[0, x], y=[0, y], z=[0, z],
                    mode='lines',
                    line=dict(color='blue', width=5),
                    name='Direction initiale'
                ))
                
                # Paramètres de visualisation
                fig.update_layout(
                    title=f"Orientation initiale: Azimuth {azimuth:.1f}°, Inclinaison {inclination:.1f}°",
                    scene = dict(
                        xaxis_title='Est (m)',
                        yaxis_title='Nord (m)',
                        zaxis_title='Profondeur (m)',
                        aspectmode='manual',
                        aspectratio=dict(x=1, y=1, z=1),
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.2)
                        ),
                    ),
                    margin=dict(l=0, r=0, t=30, b=0),
                    template="plotly_white",
                    height=300
                )
                
                return fig
            
            drill_fig = generate_drill_illustration(azimuth_initial_input, inclinaison_initiale_input)
            st.plotly_chart(drill_fig, use_container_width=True)
            
            # Informations supplémentaires sur la lithologie
            lithology_info = {
                'Granite': "Roche ignée à grains grossiers, abrasive et résistante.",
                'Schiste': "Roche métamorphique feuilletée de dureté moyenne.",
                'Gneiss': "Roche métamorphique à bandes alternées, dure et résistante.",
                'Calcaire': "Roche sédimentaire tendre à moyennement dure.",
                'Basalte': "Roche volcanique dense, dure et abrasive."
            }
            
            st.markdown(f"""
            <div style="background-color: #F8F9FA; padding: 1rem; border-radius: 6px; margin-top: 1rem;">
                <p style="margin: 0; font-weight: 600;">Lithologie sélectionnée: {lithologie_input}</p>
                <p style="margin: 0;">{lithology_info.get(lithologie_input, "")}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Bouton de prédiction
        predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
        
        with predict_col2:
            predict_button = st.button("⚡ Prédire les déviations", use_container_width=True, type="primary", disabled=not st.session_state.model_trained)
        
        # Faire la prédiction
        if predict_button and st.session_state.model_trained:
            # Créer un dataframe avec les données d'entrée
            input_data = pd.DataFrame({
                'profondeur_finale': [prof_finale_input],
                'azimuth_initial': [azimuth_initial_input],
                'inclinaison_initiale': [inclinaison_initiale_input],
                'lithologie': [lithologie_input],
                'vitesse_rotation': [vitesse_rotation_input]
            })
            
            # Faire les prédictions avec les modèles stockés dans session_state
            predicted_azimuth = st.session_state.model_azimuth.predict(input_data)[0]
            predicted_inclinaison = st.session_state.model_inclinaison.predict(input_data)[0]
            
            # Calculer les valeurs finales
            azimuth_final = azimuth_initial_input + predicted_azimuth
            inclinaison_final = inclinaison_initiale_input + predicted_inclinaison
            
            # Normalization pour azimuth (0-360°)
            azimuth_final = azimuth_final % 360
            
            # Contraindre l'inclinaison entre -90 et 0
            inclinaison_final = max(-90, min(0, inclinaison_final))
            
            # Afficher les résultats
            st.markdown("### Résultats de la prédiction")
            
            # Créer un cadre moderne pour les résultats
            st.markdown("""
            <div style="background-color: white; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); padding: 1.5rem; margin: 1.5rem 0;">
                <h4 style="margin-top: 0; text-align: center; margin-bottom: 1.5rem;">Déviations prédites</h4>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Déviation d'azimuth", f"{predicted_azimuth:.2f}°")
                st.metric("Azimuth final", f"{azimuth_final:.2f}°")
            
            with col2:
                st.metric("Déviation d'inclinaison", f"{predicted_inclinaison:.2f}°")
                st.metric("Inclinaison finale", f"{inclinaison_final:.2f}°")
            
            # Calcul de l'intensité de la déviation pour le texte d'interprétation
            deviation_magnitude = (predicted_azimuth**2 + predicted_inclinaison**2)**0.5
            
            if deviation_magnitude < 5:
                deviation_text = "faible"
                deviation_impact = "minime"
            elif deviation_magnitude < 15:
                deviation_text = "modérée"
                deviation_impact = "à considérer"
            else:
                deviation_text = "importante"
                deviation_impact = "significatif"
            
            st.markdown(f"""
                <p style="text-align: center; margin: 1rem 0; padding-top: 1rem; border-top: 1px solid #f0f0f0;">
                    La déviation prédite est <strong>{deviation_text}</strong>, avec un impact <strong>{deviation_impact}</strong> sur la position finale du forage.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualisation 3D de la trajectoire du forage
            st.markdown("### Visualisation de la trajectoire")
            
            # Conversion des coordonnées polaires en coordonnées cartésiennes
            def sph_to_cart(depth, azimuth, inclination):
                # Conversion des degrés en radians
                azimuth_rad = np.radians(azimuth)
                inclination_rad = np.radians(inclination)
                
                # x pointe vers l'est, y vers le nord, z vers le haut
                x = depth * np.cos(inclination_rad) * np.sin(azimuth_rad)
                y = depth * np.cos(inclination_rad) * np.cos(azimuth_rad)
                z = depth * np.sin(inclination_rad)  # z est négatif car inclination est négative
                
                return x, y, z
            
            # Calculer plusieurs points le long de la trajectoire
            num_points = 100
            depths = np.linspace(0, prof_finale_input, num_points)
            
            # Interpolation linéaire entre les angles initiaux et finaux
            azimuth_interp = np.linspace(azimuth_initial_input, azimuth_final, num_points)
            inclination_interp = np.linspace(inclinaison_initiale_input, inclinaison_final, num_points)
            
            # Calculer les coordonnées cartésiennes pour chaque point
            x_coords, y_coords, z_coords = [], [], []
            for i in range(num_points):
                x, y, z = sph_to_cart(depths[i], azimuth_interp[i], inclination_interp[i])
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(z)
            
            # Surface (grille)
            x_surface = np.linspace(min(x_coords)-50, max(x_coords)+50, 10)
            y_surface = np.linspace(min(y_coords)-50, max(y_coords)+50, 10)
            x_surface_grid, y_surface_grid = np.meshgrid(x_surface, y_surface)
            z_surface_grid = np.zeros_like(x_surface_grid)
            
            # Créer la visualisation 3D
            fig = go.Figure()
            
            # Ajouter la surface
            fig.add_trace(go.Surface(
                x=x_surface_grid,
                y=y_surface_grid,
                z=z_surface_grid,
                colorscale=[[0, 'lightgreen'], [1, 'lightgreen']],
                showscale=False,
                opacity=0.3
            ))
            
            # Ajouter la trajectoire
            fig.add_trace(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='lines',
                line=dict(
                    color='blue',
                    width=6
                ),
                name='Trajectoire prédite'
            ))
            
            # Ajouter la position de départ
            fig.add_trace(go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode='markers',
                marker=dict(
                    size=8,
                    color='green'
                ),
                name='Départ'
            ))
            
            # Ajouter la position finale
            fig.add_trace(go.Scatter3d(
                x=[x_coords[-1]],
                y=[y_coords[-1]],
                z=[z_coords[-1]],
                mode='markers',
                marker=dict(
                    size=8,
                    color='red'
                ),
                name='Arrivée'
            ))
            
            # Ajouter la trajectoire idéale (ligne droite) avec une meilleure visibilité
            x_ideal, y_ideal, z_ideal = sph_to_cart(prof_finale_input, azimuth_initial_input, inclinaison_initiale_input)
            
            fig.add_trace(go.Scatter3d(
                x=[0, x_ideal],
                y=[0, y_ideal],
                z=[0, z_ideal],
                mode='lines',
                line=dict(
                    color='rgba(255, 0, 0, 0.6)',  # Rouge plus visible
                    width=4,                        # Ligne plus épaisse
                    dash='dash'                     # Conserver le style tiret
                ),
                name='Trajectoire idéale'
            ))
            
            # Calculer l'écart final en mètres
            final_deviation = ((x_coords[-1] - x_ideal)**2 + (y_coords[-1] - y_ideal)**2 + (z_coords[-1] - z_ideal)**2)**0.5
            
            fig.update_layout(
                title=f"Trajectoire du forage (Écart final: {final_deviation:.2f} m)",
                scene=dict(
                    xaxis_title='Est (m)',
                    yaxis_title='Nord (m)',
                    zaxis_title='Profondeur (m)',
                    aspectmode='data'
                ),
                template="plotly_white",
                height=700,
                margin=dict(l=0, r=0, t=50, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Ajouter une section d'interprétation et de recommandation
            st.markdown("### Interprétation et recommandations")
            
            # Déterminer les recommandations basées sur la déviation
            if deviation_magnitude < 5:
                recommendations = """
                - La déviation prédite est faible et ne devrait pas nécessiter d'ajustements particuliers.
                - Procéder au forage selon les paramètres planifiés.
                - Surveiller régulièrement l'orientation pendant l'opération.
                """
            elif deviation_magnitude < 15:
                recommendations = """
                - Une déviation modérée est anticipée, des ajustements préventifs peuvent être envisagés.
                - Considérer une légère compensation de l'orientation initiale.
                - Prévoir des mesures de contrôle plus fréquentes pendant le forage.
                - Réduire la vitesse de rotation dans les zones critiques.
                """
            else:
                recommendations = """
                - Une déviation importante est prévue, des mesures correctives sont nécessaires.
                - Ajuster significativement l'orientation initiale pour compenser la déviation.
                - Utiliser des stabilisateurs supplémentaires pour maintenir la trajectoire.
                - Envisager des techniques de forage dirigé si disponibles.
                - Effectuer des mesures de contrôle très fréquentes.
                """
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div style="background-color: white; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); padding: 1.5rem; height: 100%;">
                    <h4 style="margin-top: 0;">Recommandations</h4>
                    <p>{recommendations}</p>
                    <p style="margin-top: 1rem; font-style: italic;">Note: Ces recommandations sont basées sur les prédictions du modèle et doivent être adaptées aux conditions spécifiques du site.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Créer une jauge pour visualiser l'intensité de la déviation
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=deviation_magnitude,
                    title={'text': "Intensité de la déviation (°)"},
                    gauge={
                        'axis': {'range': [None, 30], 'tickwidth': 1},
                        'bar': {'color': "rgba(0,0,0,0)"},
                        'steps': [
                            {'range': [0, 5], 'color': "lightgreen"},
                            {'range': [5, 15], 'color': "orange"},
                            {'range': [15, 30], 'color': "salmon"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': deviation_magnitude
                        }
                    }
                ))
                
                gauge_fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                    template="plotly_white"
                )
                
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Ajouter une option pour télécharger le rapport
            st.markdown("### Télécharger le rapport")
            
            # Créer un rapport PDF (simulé avec un texte formaté)
            report_text = f"""
            Rapport de prédiction de déviation de forage
            Date: {pd.Timestamp.now().strftime('%d/%m/%Y')}
            
            Paramètres du forage:
            - Profondeur finale: {prof_finale_input} m
            - Azimuth initial: {azimuth_initial_input}°
            - Inclinaison initiale: {inclinaison_initiale_input}°
            - Lithologie: {lithologie_input}
            - Vitesse de rotation: {vitesse_rotation_input} tr/min
            
            Résultats de la prédiction:
            - Déviation d'azimuth: {predicted_azimuth:.2f}°
            - Déviation d'inclinaison: {predicted_inclinaison:.2f}°
            - Azimuth final prévu: {azimuth_final:.2f}°
            - Inclinaison finale prévue: {inclinaison_final:.2f}°
            - Écart final estimé: {final_deviation:.2f} m
            
            Recommandations:
            {recommendations}
            
            Rapport généré par l'application "Prédiction de Déviation des Forages Miniers"
            Auteur: Didier Ouedraogo, P.Geo.
            """
            
            # Créer un bouton de téléchargement
            st.download_button(
                label="📄 Télécharger le rapport",
                data=report_text,
                file_name=f"rapport_deviation_forage_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

else:
    # Message pour guider l'utilisateur si aucune donnée n'est encore chargée
    if data_option == "Charger mes données" and not st.session_state.columns_mapped and st.session_state.raw_df is None:
        st.markdown("""
        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 70vh; text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem; color: #E2E8F0;">
                📊
            </div>
            <h2>Bienvenue dans l'application de prédiction de déviation des forages miniers</h2>
            <p style="max-width: 600px; margin: 1rem auto;">
                Veuillez charger un fichier CSV depuis la barre latérale pour commencer l'analyse et la modélisation.
            </p>
            <div style="background-color: #EBF5FF; padding: 1rem; border-radius: 6px; max-width: 600px; margin-top: 1rem;">
                <p style="margin: 0; color: #1A56DB;">
                    <strong>Format attendu</strong>: Un fichier CSV contenant des données sur les paramètres de forage et les déviations mesurées.
                    Vous pourrez mapper vos colonnes aux données requises par l'application après le chargement.
                </p>
            </div>
            <p style="margin-top: 2rem; color: #718096;">
                ou sélectionnez "Utiliser données démo" pour explorer l'application avec des données synthétiques.
            </p>
        </div>
        """, unsafe_allow_html=True)