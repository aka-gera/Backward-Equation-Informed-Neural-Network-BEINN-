
import os, sys, dash 
sys.path.append(os.path.abspath(os.getcwd()))

import dash_bootstrap_components as dbc
from dash import Dash, html
import webbrowser, threading
from neld_fun_0.side_bar import header_navbar

forbidden_page=('/dsa/test', '/test/dnn-pinn-neld---shear-harmonic-2-200-5-1000-test/press-0--mean-05/', '/test/dnn-pinn-neld---shear-harmonic-2-200-5-1000-train/posi--mean-05/', '/test/dnn-pinn-neld---shear-harmonic-2-200-5-1000-test/posi--mean-05/', '/test/dnn-pinn-neld---shear-harmonic-2-200-5-1000-test/press--mean-05/', '/test/dnn-pinn-neld---shear-harmonic-2-200-5-1000-train/press--mean-05/', '/test/dnn-pinn-neld---shear-harmonic-2-200-5-1000-test/momen--mean-05/', '/dsa', '/test/dnn-pinn-neld---shear-harmonic-2-200-5-1000-train/press-0--mean-05/', '/test/dnn-pinn-neld---shear-harmonic-2-200-5-1000-train/momen--mean-05/')
forbidden_endswith='None'
forbidden_endswith = None if forbidden_endswith in (None, 'None') else forbidden_endswith
head_navbar={'BEINN': 'dsa', 'DNN': 'dnn'} 

 
app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.DARKLY])
server = app.server 

header=header_navbar(head_navbar,forbidden_page)



def open_browser():
    webbrowser.open('http://127.0.0.1:8050/DSA-2')

app.layout = dbc.Container([header, dash.page_container], fluid=False)

if __name__ == '__main__':
    threading.Timer(1.25, open_browser).start()
    app.run(debug=False)
