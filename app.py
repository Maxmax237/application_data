# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Configuration de la page
st.set_page_config(page_title="MarketSight - Data Collector", layout="wide")
st.title("📊 MarketSight - Collecte & Analyse de Données Commerciales")
st.markdown("*Application de collecte organisée et analyse descriptive des données*")

# Sidebar pour la navigation
st.sidebar.title("🗂️ Navigation")
section = st.sidebar.radio(
    "Aller à :",
    ["📝 Collecte de données", "📈 Analyse descriptive", "📉 Régression linéaire", 
     "🔍 ACP (Réduction dim.)", "🏷️ Classification supervisée", "🎯 Classification non-supervisée"]
)

# Initialisation des données
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=[
        'Age', 'Revenu_annuel_k€', 'Temps_site_min', 'Nb_visites_mois',
        'Panier_moyen_€', 'Achats_12_mois', 'Churn', 'Segment'
    ])

# ==================== 1. COLLECTE DE DONNÉES ====================
if section == "📝 Collecte de données":
    st.header("📝 Collecte structurée de données clients")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Âge", min_value=18, max_value=90, value=30)
        revenu = st.number_input("Revenu annuel (k€)", min_value=10, max_value=500, value=50)
        temps_site = st.number_input("Temps sur site (minutes)", min_value=0.0, max_value=120.0, value=10.0)
        nb_visites = st.number_input("Nombre de visites/mois", min_value=0, max_value=50, value=5)
    
    with col2:
        panier_moyen = st.number_input("Panier moyen (€)", min_value=0.0, max_value=1000.0, value=50.0)
        achats_12m = st.number_input("Achats 12 derniers mois", min_value=0, max_value=100, value=10)
        churn = st.selectbox("Churn (client perdu)", ["Non", "Oui"])
        segment = st.selectbox("Segment client", ["Standard", "Premium", "Occasionnel"])
    
    if st.button("➕ Ajouter client", type="primary"):
        new_row = pd.DataFrame([{
            'Age': age,
            'Revenu_annuel_k€': revenu,
            'Temps_site_min': temps_site,
            'Nb_visites_mois': nb_visites,
            'Panier_moyen_€': panier_moyen,
            'Achats_12_mois': achats_12m,
            'Churn': 1 if churn == "Oui" else 0,
            'Segment': segment
        }])
        st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
        st.success(f"✅ Client ajouté ! Total : {len(st.session_state.data)} enregistrements")
    
    st.subheader("📋 Données collectées")
    st.dataframe(st.session_state.data, use_container_width=True)
    
    # Upload CSV
    st.subheader("📁 Import/Export")
    uploaded_file = st.file_uploader("Importer un fichier CSV", type="csv")
    if uploaded_file:
        imported_df = pd.read_csv(uploaded_file)
        st.session_state.data = pd.concat([st.session_state.data, imported_df], ignore_index=True)
        st.success(f"Importé ! Total : {len(st.session_state.data)} lignes")
    
    csv = st.session_state.data.to_csv(index=False)
    st.download_button("📥 Exporter en CSV", csv, "market_data.csv", "text/csv")

# ==================== 2. ANALYSE DESCRIPTIVE ====================
elif section == "📈 Analyse descriptive":
    st.header("📈 Analyse descriptive des données")
    
    if len(st.session_state.data) == 0:
        st.warning("Aucune donnée. Veuillez d'abord collecter des données.")
    else:
        df = st.session_state.data
        
        # Stats descriptives
        st.subheader("📊 Statistiques descriptives")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # Matrice de corrélation
        st.subheader("🔥 Matrice de corrélation")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        
        # Visualisations
        st.subheader("📊 Distributions")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x="Revenu_annuel_k€", title="Distribution des revenus")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, y="Panier_moyen_€", title="Boxplot - Panier moyen")
            st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        with col3:
            fig = px.scatter(df, x="Age", y="Panier_moyen_€", color="Segment", title="Age vs Panier moyen")
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            churn_counts = df["Churn"].value_counts().rename({0: "Non churn", 1: "Churn"})
            fig = px.pie(values=churn_counts.values, names=churn_counts.index, title="Proportion Churn")
            st.plotly_chart(fig, use_container_width=True)

# ==================== 3. RÉGRESSION LINÉAIRE ====================
elif section == "📉 Régression linéaire":
    st.header("📉 Régression linéaire - Prédiction des dépenses")
    
    if len(st.session_state.data) < 5:
        st.warning("Minimum 5 enregistrements requis pour l'analyse.")
    else:
        df = st.session_state.data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        target = st.selectbox("Variable cible (à prédire)", numeric_cols, index=list(numeric_cols).index("Panier_moyen_€") if "Panier_moyen_€" in numeric_cols else 0)
        features = st.multiselect("Variables explicatives", [c for c in numeric_cols if c != target], default=["Age", "Revenu_annuel_k€", "Temps_site_min", "Nb_visites_mois"])
        
        if features and st.button("Lancer la régression"):
            X = df[features].dropna()
            y = df[target].loc[X.index]
            
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            st.success(f"R² = {r2_score(y, y_pred):.4f}")
            st.metric("RMSE", f"{np.sqrt(mean_squared_error(y, y_pred)):.2f}")
            
            st.subheader("📈 Visualisation")
            fig = px.scatter(x=y, y=y_pred, labels={"x": "Valeurs réelles", "y": "Prédictions"}, title="Réel vs Prédit")
            fig.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], mode="lines", name="Idéal"))
            st.plotly_chart(fig)
            
            st.subheader("Coefficients")
            coef_df = pd.DataFrame({"Variable": features, "Coefficient": model.coef_})
            st.dataframe(coef_df)

# ==================== 4. ACP (RÉDUCTION DIMENSIONNALITÉ) ====================
elif section == "🔍 ACP (Réduction dim.)":
    st.header("🔍 Analyse en Composantes Principales (ACP)")
    
    if len(st.session_state.data) < 5:
        st.warning("Minimum 5 enregistrements requis.")
    else:
        df = st.session_state.data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[numeric_cols].dropna())
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        st.write(f"**Variance expliquée :** Composante 1 = {pca.explained_variance_ratio_[0]:.2%}, Composante 2 = {pca.explained_variance_ratio_[1]:.2%}")
        st.write(f"**Variance totale conservée :** {pca.explained_variance_ratio_.sum():.2%}")
        
        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], title="Projection ACP", labels={"x": "CP1", "y": "CP2"})
        st.plotly_chart(fig)
        
        # Cercle des corrélations
        st.subheader("🔘 Cercle des corrélations")
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        for i, var in enumerate(numeric_cols):
            ax2.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], head_width=0.05, color="red")
            ax2.text(pca.components_[0, i]*1.1, pca.components_[1, i]*1.1, var, fontsize=10)
        ax2.add_patch(plt.Circle((0,0), 1, color="blue", fill=False))
        ax2.set_xlim(-1.1, 1.1)
        ax2.set_ylim(-1.1, 1.1)
        ax2.axhline(0, color="gray", linestyle="--")
        ax2.axvline(0, color="gray", linestyle="--")
        ax2.set_aspect("equal")
        st.pyplot(fig2)

# ==================== 5. CLASSIFICATION SUPERVISÉE ====================
elif section == "🏷️ Classification supervisée":
    st.header("🏷️ Classification supervisée - Prédiction du Churn")
    
    if len(st.session_state.data) < 10:
        st.warning("Minimum 10 enregistrements requis.")
    else:
        df = st.session_state.data
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Churn"]
        
        X = df[numeric_cols].dropna()
        y = df.loc[X.index, "Churn"]
        
        if len(y.unique()) < 2:
            st.error("Besoin de classes Churn=0 et Churn=1 dans les données.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.metric("Précision (Accuracy)", f"{accuracy_score(y_test, y_pred):.2%}")
            
            st.subheader("Rapport de classification")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            
            st.subheader("Importance des variables")
            imp_df = pd.DataFrame({"Variable": numeric_cols, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
            fig = px.bar(imp_df, x="Importance", y="Variable", orientation="h", title="Importance des features")
            st.plotly_chart(fig)

# ==================== 6. CLASSIFICATION NON-SUPERVISÉE ====================
elif section == "🎯 Classification non-supervisée":
    st.header("🎯 Segmentation clients - K-Means")
    
    if len(st.session_state.data) < 10:
        st.warning("Minimum 10 enregistrements requis.")
    else:
        df = st.session_state.data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != "Churn"]
        
        k = st.slider("Nombre de segments (k)", 2, 8, 3)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[numeric_cols].dropna())
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        df_clust = df.iloc[:len(clusters)].copy()
        df_clust["Segment_KMeans"] = clusters
        
        st.subheader("📊 Centres des clusters")
        centers_scaled = scaler.inverse_transform(kmeans.cluster_centers_)
        centers_df = pd.DataFrame(centers_scaled, columns=numeric_cols)
        centers_df.index = [f"Cluster {i}" for i in range(k)]
        st.dataframe(centers_df.style.background_gradient(cmap="Blues"), use_container_width=True)
        
        # Visualisation PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=clusters.astype(str), 
                         title=f"Segments K-Means (k={k})", labels={"color": "Cluster"},
                         color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig)
        
        # Profil des clusters
        st.subheader("📋 Profil moyen par cluster")
        profile = df_clust.groupby("Segment_KMeans")[numeric_cols].mean()
        st.dataframe(profile.style.background_gradient(cmap="viridis"), use_container_width=True)
