import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Load your data
data = pd.read_csv('name_data.csv', header=0)

# Initialize the Dash app
app = dash.Dash(__name__)

server=app.server # used for deployment on render.com

# Define the layout of your app
app.layout = html.Div([
    html.H1("Visualization of baby name evolution through time in Quebec"),
    
    dcc.Input(
        id='name-input',
        type='text',
        value='',
        placeholder='Enter a name...',
        debounce=True
    ),
    
    dcc.Graph(id='name-graph')
])

# Define a callback function to update the graph when a name is entered
@app.callback(
    Output('name-graph', 'figure'),
    Input('name-input', 'value')
)
def update_graph(prenom):
    prenom = prenom.upper()
    identifier = data[data.iloc[:, 0] == prenom]
    identifier = identifier.drop(identifier.columns[0], axis=1)
    identifier = identifier.drop(identifier.columns[-1], axis=1)
    identifier = identifier.sum(axis=0)
    identifier = identifier.to_frame()
    identifier.reset_index(inplace=True)
    identifier.columns = ['Year', 'Count']
    fig = px.line(identifier, x='Year', y='Count', title=prenom.upper() + " through the years", hover_data=['Count'])
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)