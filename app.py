from flask import Flask, render_template
import joblib
import pandas as pd 
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = Flask(__name__)

df = pd.read_csv('https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/iris.csv')
dash1 = dash.Dash(
        __name__, 
        server=app, 
        routes_pathname_prefix='/dash1/')

dash1.layout = html.Div([
    dcc.Link("back", href='/', refresh="True"),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict("rows"),
        )
    ])

dash3 = dash.Dash(
        __name__, 
        server=app, 
        routes_pathname_prefix='/dash3/')

model = joblib.load('model.dmp')

@app.route('/')
def index():
    return render_template('index.html')

dash3.layout = html.Div(children=[
    html.H1(children='AdaBoost', style={'textAlign': 'center'}),

    html.Div(children=[
        html.Label('Input your Features : '),
        dcc.Input(id='input1', placeholder='features 1', type='text'),
        dcc.Input(id='input2', placeholder='features 2', type='text'),
        dcc.Input(id='input3', placeholder='features 3', type='text'),
        dcc.Input(id='input4', placeholder='features 4', type='text'),
        html.Div(id='result')
    ], style={'textAlign': 'center'}),
])

@dash3.callback(
    Output(component_id='result', component_property='children'),
    [Input(component_id='input1', component_property='value'),
    Input(component_id='input2', component_property='value'),
    Input(component_id='input3', component_property='value'),
    Input(component_id='input4', component_property='value')])

def update_years_of_experience_input(input1, input2, input3, input4):
    try:
        prediction = model.predict([[input1,input2,input3,input4]])
        return 'Prediction is {}'.format(prediction)
    except ValueError:
        return 'Unable to give prediction'


if __name__ == '__main__':
    app.run(debug=True)