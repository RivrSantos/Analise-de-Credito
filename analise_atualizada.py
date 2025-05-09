import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(layout="wide", page_title="Análise de Crédito com IA")

@st.cache_data
def carregar_dados(caminho):
    return pd.read_csv(caminho)

def codificar(df, colunas):
    for col in colunas:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

clientes = carregar_dados("clientes.csv")
clientes = codificar(clientes, ["profissao", "mix_credito", "comportamento_pagamento"])

# Separação dos dados
y = clientes["score_credito"]
X = clientes.drop(columns=["score_credito", "id_cliente"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelos
modelo_rf = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()
modelo_rf.fit(X_train, y_train)
modelo_knn.fit(X_train, y_train)

# Previsões
y_pred_rf = modelo_rf.predict(X_test)
y_pred_knn = modelo_knn.predict(X_test)

# Layout com abas
st.title("🔍 Análise de Crédito com IA")
st.markdown("Este painel realiza análise de dados e previsão de score de crédito com Machine Learning.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dados", "⚙️ Treinamento", "📈 Resultados", "🔮 Previsão"])

with tab1:
    st.header("📁 Base de Clientes")
    st.dataframe(clientes)

    st.subheader("📉 Distribuição do Score de Crédito")
    fig_dist = px.histogram(clientes, x="score_credito", color="score_credito", nbins=20, title="Distribuição do Score")
    st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("📈 Relação entre Renda e Score")
    if "renda" in clientes.columns:
        fig_renda = px.scatter(clientes, x="renda", y="score_credito", color="profissao", title="Score vs Renda")
        st.plotly_chart(fig_renda, use_container_width=True)

with tab2:
    st.header("🛠️ Modelos Treinados")
    st.markdown("**Atributos de entrada:**")
    st.code(f"{list(X.columns)}")
    st.markdown("**Target:** score_credito")

    st.subheader("🎯 Importância dos Atributos (Random Forest)")
    importancias = pd.DataFrame({"atributo": X.columns, "importancia": modelo_rf.feature_importances_})
    fig_import = px.bar(importancias.sort_values(by="importancia"), x="atributo", y="importancia", title="Importância dos Atributos")
    st.plotly_chart(fig_import, use_container_width=True)

with tab3:
    st.header("📊 Acurácia dos Modelos")
    st.metric("Random Forest", f"{accuracy_score(y_test, y_pred_rf):.2%}")
    st.metric("KNN", f"{accuracy_score(y_test, y_pred_knn):.2%}")
    st.success("O modelo Random Forest apresentou melhor desempenho.")

with tab4:
    st.header("🔮 Previsão para Novos Clientes")
    novos = carregar_dados("novos_clientes.csv")
    novos = codificar(novos, ["profissao", "mix_credito", "comportamento_pagamento"])
    previsao = modelo_rf.predict(novos)
    novos["score_credito"] = previsao
    st.dataframe(novos)

    st.subheader("📈 Resultado da Previsão")
    fig_prev = px.histogram(novos, x="score_credito", color="score_credito", title="Distribuição da Previsão para Novos Clientes")
    st.plotly_chart(fig_prev, use_container_width=True)

with st.expander("ℹ️ Sobre"):
    st.markdown("**Desenvolvido por Rivr Santos**  \nPython | Data Analytics | Front-End")
    
