import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import os
import pickle
import base64
import io
import requests
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
DATA_PATH = os.path.join(PROJECT_ROOT, "data")

try:
    model_files = [f for f in os.listdir(MODELS_PATH) if f.endswith('.pkl')]
    MODEL_OPTIONS = [{'label': f.replace('.pkl', ''), 'value': f} for f in model_files]
except FileNotFoundError:
    print(f"Avertissement : Le dossier 'models' n'a pas été trouvé à {MODELS_PATH}")
    MODEL_OPTIONS = []
except Exception as e:
    print(f"Erreur lors du listage des modèles : {e}")
    MODEL_OPTIONS = []


def load_real_spot_data(data_path):

    try:
        df_spot = pd.read_csv(os.path.join(data_path, "France.csv"))
        df_spot['datetime'] = pd.to_datetime(df_spot['Datetime (Local)'])
        df_spot = df_spot.rename(columns={'Price (EUR/MWhe)': 'Prix_SPOT_Reel'})
        df_spot = df_spot[['datetime', 'Prix_SPOT_Reel']].set_index('datetime')
        return df_spot
    except FileNotFoundError:
        print(f"ERREUR: Fichier 'data/France.csv' introuvable pour la comparaison.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Erreur lors du chargement de France.csv : {e}")
        return pd.DataFrame()


prev_spot_layout = html.Div([
    html.H3("Prévision de prix SPOT"),
    
    dcc.Store(id='store-prevision-data'),
    
    dbc.Tabs(
        id="tabs-data-source",
        active_tab="tab-api",
        children=[
            dbc.Tab(
                label="API éco2mix",
                tab_id="tab-api",
                children=[
                    dbc.Row([
                        dbc.Col(
                            dcc.DatePickerRange(
                                id='api-date-picker',
                                display_format='DD/MM/YYYY',
                                start_date=datetime.now().date(),
                                end_date=datetime.now().date(),
                            ),
                            width="auto"
                        ),
                        dbc.Col(
                            dbc.Button("Charger les données", id="btn-load-api", color="success"),
                            width="auto"
                        ),
                        dbc.Col(
                            dbc.Spinner(html.Div(id="api-feedback")),
                            width="auto"
                        )
                    ], className="mt-3 align-items-center")
                ]
            ),
            
            dbc.Tab(
                label="Fichier CSV/XLS",
                tab_id="tab-csv",
                children=[
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Glissez-déposez ou ',
                            html.A('Sélectionnez un fichier éco2mix')
                        ]),
                        style={
                            'width': '100%', 'height': '60px', 'lineHeight': '60px',
                            'borderWidth': '1px', 'borderStyle': 'dashed',
                            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
                        },
                        multiple=False
                    ),
                    dbc.Spinner(html.Div(id="upload-feedback"))
                ]
            ),
        ]
    ),
    
    html.Div(
        id="predict-section",
        style={'display': 'none'},
        children=[
            html.Hr(),
            html.H4("Modèle et prévisions"),
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(
                        id='model-dropdown',
                        options=MODEL_OPTIONS,
                        placeholder="Sélectionner un modèle..."
                    ),
                    width=6
                ),
                dbc.Col(
                    dbc.Button("Lancer les prévisions", id="btn-predict", color="primary"),
                    width="auto"
                )
            ], className="align-items-center")
        ]
    ),
    
    dbc.Spinner(
        dcc.Graph(id='prediction-graph')
    )
])



def parse_uploaded_file(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        df = pd.read_csv(
            io.StringIO(decoded.decode('latin1')), 
            sep='\t', 
            index_col=False
        )
        
        if not df.empty and 'Copyright' in str(df.iloc[-1, 0]):
             df = df.iloc[:-1]

    except Exception as e:
        return None, f"Erreur de parsing CSV/XLS : {e}"

    try:
        if 'Date' not in df.columns or 'Heures' not in df.columns:
            return None, "Erreur : Les colonnes 'Date' et 'Heures' sont manquantes dans le fichier."
        
        return df.to_json(date_format='iso', orient='split'), f"Fichier chargé : {len(df)} lignes prêtes."
    except Exception as e:
        return None, f"Erreur lors de la conversion en JSON : {e}"

def fetch_api_data(start_date, end_date):
    try:
        url = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-tr/exports/json"
        params = {
            "limit": -1,
            "where": f"date_heure >= '{pd.to_datetime(start_date)}' AND date_heure <= '{pd.to_datetime(end_date)}T23:59:59'",
            "order_by": "date_heure"
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()

        data = pd.DataFrame(resp.json())
        
        if data.empty:
            return None, "Aucune donnée trouvée sur l'API pour cette période."

        data = data.rename(columns={
            "date": "Date", "heure": "Heures",
            'consommation': 'Consommation',
            'prevision_j1': 'Prévision J-1',
            'prevision_j': 'Prévision J',
            'fioul': 'Fioul',
            'charbon': 'Charbon',
            'gaz': 'Gaz',
            'nucleaire': 'Nucléaire',
            'eolien': 'Eolien',
            'eolien_terrestre': 'Eolien terrestre',
            'eolien_offshore': 'Eolien offshore',
            'solaire': 'Solaire',
            'hydraulique': 'Hydraulique',
            'pompage': 'Pompage',
            'bioenergies': 'Bioénergies',
            'ech_physiques': 'Ech. physiques',
            'taux_co2': 'Taux de Co2',
            'ech_comm_angleterre': 'Ech. comm. Angleterre',
            'ech_comm_espagne': 'Ech. comm. Espagne',
            'ech_comm_italie': 'Ech. comm. Italie',
            'ech_comm_suisse': 'Ech. comm. Suisse',
            'ech_comm_allemagne_belgique': 'Ech. comm. Allemagne-Belgique',
            'fioul_tac': 'Fioul - TAC',
            'fioul_cogen': 'Fioul - Cogén.',
            'fioul_autres': 'Fioul - Autres',
            'gaz_tac': 'Gaz - TAC',
            'gaz_cogen': 'Gaz - Cogén.',
            'gaz_ccg': 'Gaz - CCG',
            'gaz_autres': 'Gaz - Autres',
            'hydraulique_fil_eau_eclusee': 'Hydraulique - Fil de l?eau + éclusée',
            'hydraulique_lacs': 'Hydraulique - Lacs',
            'hydraulique_step_turbinage': 'Hydraulique - STEP turbinage',
            'bioenergies_dechets': 'Bioénergies - Déchets',
            'bioenergies_biomasse': 'Bioénergies - Biomasse',
            'bioenergies_biogaz': 'Bioénergies - Biogaz',
            "stockage_batterie": " Stockage batterie",
            "destockage_batterie": "Déstockage batterie"
        })
        
        if 'Date' not in data.columns or 'Heures' not in data.columns:
            if 'date_heure' in data.columns:
                print("Fallback API : conversion de 'date_heure' en 'Date' et 'Heures'")
                dt_col = pd.to_datetime(data['date_heure']).dt.tz_convert('Europe/Paris').dt.tz_localize(None)
                data['Date'] = dt_col.dt.strftime('%Y-%m-%d')
                data['Heures'] = dt_col.dt.strftime('%H:%M')
            else:
                return None, "Erreur API: Colonnes 'Date' et 'Heures' non trouvées après renommage."
        
        return data.to_json(date_format='iso', orient='split'), f"API chargée : {len(data)} lignes prêtes."

    except Exception as e:
        return None, f"Erreur API : {e}"   

@callback(
    Output('store-prevision-data', 'data'),
    Output('upload-feedback', 'children'),
    Input('upload-data', 'contents')
)
def update_store_from_upload(contents):
    if contents is None:
        return None, "En attente d'un fichier..."
    
    data_json, feedback_msg = parse_uploaded_file(contents)
    
    if data_json:
        return data_json, dbc.Alert(feedback_msg, color="success", duration=4000)
    else:
        return None, dbc.Alert(feedback_msg, color="danger")

@callback(
    Output('store-prevision-data', 'data', allow_duplicate=True),
    Output('api-feedback', 'children'),
    Input('btn-load-api', 'n_clicks'),
    [State('api-date-picker', 'start_date'),
     State('api-date-picker', 'end_date')],
    prevent_initial_call=True
)
def update_store_from_api(n_clicks, start_date, end_date):
    if n_clicks is None or not start_date or not end_date:
        return None, "Veuillez sélectionner une période."
        
    data_json, feedback_msg = fetch_api_data(start_date, end_date)
    
    if data_json:
        return data_json, dbc.Alert(feedback_msg, color="success", duration=4000)
    else:
        return None, dbc.Alert(feedback_msg, color="danger")

@callback(
    Output('predict-section', 'style'),
    Input('store-prevision-data', 'data')
)
def toggle_predict_section(data):
    if data:
        return {'display': 'block'}
    return {'display': 'none'}

@callback(
    Output('prediction-graph', 'figure'),
    Input('btn-predict', 'n_clicks'),
    [State('store-prevision-data', 'data'),
     State('model-dropdown', 'value')],
    prevent_initial_call=True
)
def run_prediction_and_plot(n_clicks, data_json, model_filename):
    if not data_json or not model_filename:
        return go.Figure().update_layout(title="Veuillez charger des données et sélectionner un modèle.")

    try:
        X_test_raw = pd.read_json(data_json, orient='split')
        if X_test_raw.empty:
            return go.Figure().update_layout(title="Erreur : Aucune donnée n'a été chargée.")
    except Exception as e:
        return go.Figure().update_layout(title=f"Erreur de lecture des données en cache : {e}")

    model_path = os.path.join(MODELS_PATH, model_filename)
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        model_features = model.feature_names_in_
    except Exception as e:
        return go.Figure().update_layout(title=f"Erreur lors du chargement du modèle {model_filename}: {e}")

    try:
        X_test = X_test_raw.replace("ND", np.nan)
        
        missing_cols = [col for col in model_features if col not in X_test.columns]
        if missing_cols:
            return go.Figure().update_layout(title=f"Erreur : Colonnes manquantes dans les données : {', '.join(missing_cols)}")
        X_test = X_test.dropna(subset=model_features)
        if X_test.empty:
            return go.Figure().update_layout(title="Erreur : Aucune donnée valide restante après nettoyage (NaNs).")
        prediction_datetimes = pd.to_datetime(X_test['Date'].astype(str) + ' ' + X_test['Heures'].astype(str))
        X_test_final = X_test[model_features]
        predictions = model.predict(X_test_final)

    except Exception as e:
        return go.Figure().update_layout(title=f"Erreur lors de la préparation des données ou de la prédiction : {e}")
    df_real_spot = load_real_spot_data(DATA_PATH)
    if df_real_spot.empty:
        print("Avertissement : Données réelles (France.csv) introuvables. Affichage des prévisions uniquement.")
        df_results = pd.DataFrame({
            'datetime': prediction_datetimes,
            'Prévision': predictions
        })
        fig = px.line(df_results, x='datetime', y='Prévision', title=f"Prévision ({model_filename.replace('.pkl','')})")
        fig.update_layout(template='plotly_white')
        return fig
    df_results = pd.DataFrame({
        'datetime': prediction_datetimes,
        'Prévision': predictions
    }).set_index('datetime')
    
    df_plot = df_real_spot.join(df_results, how='inner')
    
    if df_plot.empty:
        return go.Figure().update_layout(title="Aucune donnée commune entre les prévisions et les données réelles sur cette période.")

    df_plot_melted = df_plot.reset_index().melt(
        id_vars='datetime',
        value_vars=['Prix_SPOT_Reel', 'Prévision'],
        var_name='Type',
        value_name='Prix (EUR/MWhe)'
    )

    fig = px.line(
        df_plot_melted,
        x='datetime',
        y='Prix (EUR/MWhe)',
        color='Type',
        title=f"Comparaison Prévision ({model_filename.replace('.pkl','')}) vs Réel"
    )
    
    fig.update_layout(template='plotly_white', legend_title_text='Données')
    
    return fig