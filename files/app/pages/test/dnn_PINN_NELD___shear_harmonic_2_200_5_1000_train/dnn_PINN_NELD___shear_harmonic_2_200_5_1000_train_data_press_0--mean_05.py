

import os, sys ,dash 
sys.path.append(os.getcwd() ) 
from beinn.neld_fun_0.side_bar import sidebar ,get_dnn,dnn_page 


page_dir= '/test/dnn-pinn-neld---shear-harmonic-2-200-5-1000-train/press-0--mean-05/dnn-pinn-neld---shear-harmonic-2-200-5-1000-train-data-press-0--mean-05' 
page_name='press_0--mean_05'  
page_name_view=dnn_page()['results']
forbidden_endswith=None 
forbidden_endswith = None if forbidden_endswith in (None, 'None') else forbidden_endswith


dash.register_page(__name__, title=page_name, name=page_name,order=0) 

def layout():
    return get_dnn(page_dir,page_name_view,forbidden_endswith,)
