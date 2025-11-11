import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "eCO2mix_RTE_En-cours-Consolide.csv")

FILIERE_OPTIONS = [
    "Consommation",
    "Fioul",
    "Charbon",
    "Gaz",
    "Nucléaire",
    "Eolien",
    "Solaire",
    "Hydraulique",
    "Bioénergies"
]

MIX_COLUMNS = [
    "Fioul",
    "Charbon",
    "Gaz",
    "Nucléaire",
    "Eolien",
    "Solaire",
    "Hydraulique",
    "Bioénergies"
]

def load_eco2mix_data(filepath):
    """
    Charge et nettoie les données éco2mix.
    """
    try:
        df = pd.read_csv(
            filepath,
            sep='\t',
            encoding='latin-1', 
            index_col=False,
        )
        df = df.dropna()
    except FileNotFoundError:
        print(f"ERREUR: Fichier de données non trouvé à {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"ERREUR lors du chargement : {e}")
        return pd.DataFrame()

    try:
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Heures'], format='%Y-%m-%d %H:%M')
        df = df.set_index('datetime')
    except KeyError:
        return pd.DataFrame()

    all_needed_cols = list(set(FILIERE_OPTIONS + MIX_COLUMNS))
    for col in all_needed_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = 0

    df_hourly = df[all_needed_cols].resample('h').mean()
    return df_hourly

df_eco2mix = load_eco2mix_data(DATA_FILE_PATH)

if df_eco2mix.empty:
    eco2mix_layout = dbc.Container([
        dbc.Alert(
            [
                html.H4("Erreur de chargement des données", className="alert-heading"),
                html.P(f"Le fichier de données 'eCO2mix_RTE_En-cours-Consolide.csv' n'a pas pu être chargé depuis :"),
                html.P(f"{DATA_FILE_PATH}"),
                html.Hr(),
            ],
            color="danger",
            className="mt-4"
        )
    ], fluid=True, style={'backgroundColor': 'white'})

else:

    eco2mix_layout = dbc.Container(
        fluid=True,
        style={'backgroundColor': 'white'},
        children=[
            

            dbc.Row(
                [
                    dbc.Col(
                        width=3, 
                        className="p-3",
                        children=[
                            html.H4("Contrôles", className="mt-4"),
                            html.Hr(),
                            
                            html.Label("Sélection de la période :", className="fw-bold"),
                            dcc.DatePickerRange(
                                id='eco2mix-date-picker',
                                min_date_allowed=df_eco2mix.index.min().date(),
                                max_date_allowed=df_eco2mix.index.max().date(),
                                start_date=max(df_eco2mix.index.min(), df_eco2mix.index.max() - pd.Timedelta(days=30)).date(),
                                end_date=df_eco2mix.index.max().date(),
                                display_format='DD/MM/YYYY',
                                className="mb-3 w-100" 
                            ),
                            
                            html.Label("Sélection des filières ('Total' = Consommation) :", className="fw-bold"),
                            dcc.Dropdown(
                                id='eco2mix-filiere-dropdown',
                                options=[{'label': f, 'value': f} for f in FILIERE_OPTIONS],
                                value=['Nucléaire', 'Eolien', 'Solaire', 'Hydraulique'], 
                                multi=True,
                                className="mb-3"
                            ),
                            
                            html.Label("Type de visualisation :", className="fw-bold"),
                            dcc.RadioItems(
                                id='eco2mix-chart-type',
                                options=[
                                    {'label': ' Superposition des courbes', 'value': 'line'},
                                    {'label': ' Courbes sommées', 'value': 'area'}
                                ],
                                value='area', 
                                labelStyle={'display': 'block'}, 
                                className="mb-3"
                            )
                        ]
                    ),
                    
                    dbc.Col(

                        width=9, 
                        children=[
                            html.H4("Évolution de la production sur la période", className="mt-4 text-center"),
                            dcc.Graph(id='eco2mix-main-graph', style={'height': '80vh'}) 
                        ]
                    ),
                ]
            ),
            
            html.Hr(),

            dbc.Row(
                justify="center", 
                children=[
                    dbc.Col(
                        width=6, 
                        className="p-3",
                        children=[
                            html.H4("Mix énergétique sur la période", className="mt-4 text-center"),
                            dcc.Graph(id='eco2mix-pie-chart', style={'height': '45vh'})
                        ]
                    )
                ]
            )
        ]
    )

    
    @callback(
        [Output('eco2mix-main-graph', 'figure'),
         Output('eco2mix-pie-chart', 'figure')],
        [Input('eco2mix-date-picker', 'start_date'),
         Input('eco2mix-date-picker', 'end_date'),
         Input('eco2mix-filiere-dropdown', 'value'),
         Input('eco2mix-chart-type', 'value')]
    )
    def update_eco2mix_graphs(start_date, end_date, filiere_values, chart_type):
        
        if not start_date or not end_date or not filiere_values:
            return go.Figure(), go.Figure()

        dff = df_eco2mix.loc[start_date:end_date]
        
        if chart_type == 'line':
            fig_main = px.line(dff, x=dff.index, y=filiere_values)
            fig_main.update_traces(hovertemplate='<b>%{y:,.0f} MW</b><br>%{x}')
        else:
            fig_main = px.area(dff, x=dff.index, y=filiere_values)
            fig_main.update_traces(hovertemplate='<b>%{y:,.0f} MW</b><br>%{x}')
        
        fig_main.update_layout(
            title_text=f"Évolution de la production ({chart_type})",
            xaxis_title="Date",
            yaxis_title="Production (MW)",
            legend_title_text='Filières',
            template='plotly_white'
        )
        
        pie_data = dff[MIX_COLUMNS].sum()
        
        pie_data = pie_data[pie_data > 0]
        
        fig_pie = px.pie(
            pie_data, 
            names=pie_data.index, 
            values=pie_data.values,
            hole=0.3,
            title="Répartition du mix énergétique"
        )
        
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(
            template='plotly_white',
            showlegend=False
        )
        
        return fig_main, fig_pie