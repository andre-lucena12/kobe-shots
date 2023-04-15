import json
import matplotlib
import mlflow
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
import typer

app = typer.Typer()

@app.command()
def consumir():

    st.set_page_config(page_title="_THE BLACK MAMBA_", layout='wide')

    st.title("_THE BLACK MAMBA_")
    st.header("Metricas de Inferência 🛠")

    run_id = mlflow.active_run()
    #data_return = mlflow.get_run(run_id).info.lifecycle_stage
    st.text(run_id)


    data_pd = pd.read_parquet('../data/03_primary/3fg_dataset.parquet')
    data_3pt = data_pd.drop(columns= 'shot_made_flag')
    url = 'http://localhost:5000/invocations'

    data_serialized = data_3pt.to_dict(orient='records')
         
    results = requests.post(url, 
                            json={'dataframe_records':data_serialized},
                            headers= {'Content-type': 'application/json'})
    response = json.loads(results.content.decode('utf-8'))
    data_response = pd.DataFrame(response)
    print(data_response.value_counts())

if __name__ == "__main__":
    app()
