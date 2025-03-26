import pandas as pd
import plotly.express as px
from django_plotly_dash import DjangoDash
from dash import html, dcc
from ai_models.models import sale_forecast

# Initialize the Dash app
app = DjangoDash('SalesForecast', serve_locally=True)

# Load data safely
try:
    queryset = sale_forecast.objects.all().values()
    df = pd.DataFrame.from_records(queryset)

    # Convert date string to datetime if needed
    # df['ds'] = pd.to_datetime(df['ds'])

    print(f"Loaded {len(df)} records")  # Debug data loading

except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame()

# Create a default figure even if data is empty
if not df.empty:
    fig = px.line(df, x='ds', y='prediction',
                  title='Sales Forecast',
                  labels={'ds': 'Date', 'prediction': 'Predicted Sales'})
    fig.add_scatter(x=df['ds'], y=df['prediction_upper'],
                    mode='lines', line=dict(color='gray'), name='Upper Bound')
    fig.add_scatter(x=df['ds'], y=df['prediction_lower'],
                    mode='lines', line=dict(color='gray'), fill='tonexty', name='Lower Bound')
else:
    fig = px.line(title='No Data Available')  # Empty figure

# Define the layout using proper Dash components
app.layout = html.Div([
    dcc.Graph(
        id='sales-forecast-graph',
        figure=fig
    ),
    html.Div(id='output-container', children='Data loaded successfully!')
], style={'padding': '20px'})