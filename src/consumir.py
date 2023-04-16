import json
import matplotlib.pyplot as plt
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import streamlit as st
import typer

app = typer.Typer()

@app.command()
def consumir():

    #CARREGANDO DATASET DAS METRICAS DO MELHOR MODELO PELO PYCARET E REALIZANDO ALGUNS FILTROS E TRANSFORMAÇÕES
    metrics_best_model = pd.read_parquet('../data/08_reporting/metricas_best_model.parquet')
    metrics_transform_best = metrics_best_model.loc['value',['Precision_Best_Model','Recall_Best_Model','F1_Score_Best_Model','Accuracy_Best_Model']] * 100
    metrics_transform_best['LogLoss_Best_Model'] = metrics_best_model.loc['value','LogLoss_Best_Model'] * 10
    metrics_transform_best = pd.DataFrame(metrics_transform_best).T.rename(index={'value':'Best by PyCaret'})
    metrics_transform_best = metrics_transform_best.rename(columns = {'LogLoss_Best_Model':'LogLoss',
                                                                  'Precision_Best_Model':'Precision',
                                                                  'Recall_Best_Model':'Recall',
                                                                  'F1_Score_Best_Model':'F1_Score',
                                                                  'Accuracy_Best_Model':'Accuracy'})

    #CARREGANDO DATASET DAS METRICAS DA REGRESSÃO LOGISTICA E REALIZANDO ALGUNS FILTROS E TRANSFORMAÇÕES
    metrics_lr_model = pd.read_parquet('../data/08_reporting/metricas_lr_model.parquet')
    metrics_transform_lr = metrics_lr_model.loc['value',['Precision','Recall','F1_Score','Accuracy']] * 100
    metrics_transform_lr['LogLoss'] = metrics_lr_model.loc['value','LogLoss_Tuned_Lr_Model'] * 10
    metrics_transform_lr = pd.DataFrame(metrics_transform_lr).T.rename(index={'value':'Logistic Regression'})

    #JUNTANDO OS DOIS DATASETS PARA COMPARAÇÃO
    metrics_all = metrics_transform_lr.append(metrics_transform_best)

    #DEFININDO CONFIGURAÇÕES DO STREAMLIT
    st.set_page_config(page_title="_THE BLACK MAMBA_", layout='wide')
    st.title("🏀 THE BLACK MAMBA 🏀")
    
    #SIDEBAR PARA FILTROS E SELEÇÕES
    st.sidebar.markdown('# Filtro para os resultados')
    st.sidebar.markdown('### Filtro para modelos treinados')
    modelo = st.sidebar.selectbox('Selecione o Modelo', options= ['Logistic Regression', 'Best by PyCaret', 'Compare'])
    st.sidebar.markdown('### Inferir DataSet de 3 Pontos')
    
    #PLOTANDO OS GRAFICOS DAS METRICAS
    if modelo == 'Logistic Regression':
        st.header("Resultados do Modelo {}".format(modelo))
        ax1 = metrics_transform_lr.T.plot(kind='bar', figsize=(8,6))
        plt.title('Desempenho dos Modelos')
        plt.xlabel('Métricas')
        plt.ylabel('Pontuações')
        fig_lr = ax1.get_figure()
        st.pyplot(fig = fig_lr)
    
    if modelo == 'Best by PyCaret':
        st.header("Resultados do Modelo {}".format(modelo))
        ax2 = metrics_transform_best.T.plot(kind='bar', figsize=(8,6))
        plt.title('Desempenho dos Modelos')
        plt.xlabel('Métricas')
        plt.ylabel('Pontuações')
        fig_best = ax2.get_figure()
        st.pyplot(fig = fig_best)
    
    if modelo == 'Compare':
        st.header("Comparando os Resultados")
        ax3 = metrics_all.T.plot(kind='bar', figsize=(8,6))
        plt.title('Desempenho dos Modelos')
        plt.xlabel('Métricas')
        plt.ylabel('Pontuações')
        fig_all = ax3.get_figure()
        st.pyplot(fig = fig_all)

    
    try_model = st.sidebar.button('Começar!')

    if try_model == 1:
        st.sidebar.text('Carregando DataSet')
        data_pd = pd.read_parquet('../data/03_primary/3fg_dataset.parquet')
        data_3pt = data_pd.drop(columns= 'shot_made_flag')
        st.sidebar.text('DataSet carregado')

        st.sidebar.text('Conectando com a API via')
        st.sidebar.text('http://localhost:5000/invocations')
        url = 'http://localhost:5000/invocations'
        data_serialized = data_3pt.to_dict(orient='records')
        results = requests.post(url, 
                                json={'dataframe_records':data_serialized},
                                headers= {'Content-type': 'application/json'})
        
        st.sidebar.text('Conectado!')

        response = json.loads(results.content.decode('utf-8'))
        y_pred = pd.DataFrame(response)
        y_true = data_pd[['shot_made_flag']]

        accuracy = accuracy_score(y_true, y_pred)
        f1score = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        metrics_3pt = pd.DataFrame({
            'Precision':{'value':precision},
            'Recall':{'value':recall},
            'F1_Score':{'value':f1score},
            'Accuracy':{'value':accuracy}
        })

        st.header("Resultados da inferência")
        ax4 = metrics_3pt.T.plot(kind='bar', figsize=(8,6))
        plt.title('Desempenho do Modelo')
        plt.xlabel('Métricas')
        plt.ylabel('Pontuações')
        fig_3pt = ax4.get_figure()
        st.pyplot(fig = fig_3pt)

if __name__ == "__main__":
    app()
