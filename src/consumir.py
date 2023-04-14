import json
import numpy as np
import pandas as pd
import requests
import typer

app = typer.Typer()

@app.command()
def consumir():
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
