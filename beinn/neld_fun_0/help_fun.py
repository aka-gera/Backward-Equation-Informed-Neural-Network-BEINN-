import numpy as np 
import plotly.graph_objs as go
import plotly.io as pio
  
def plotly_scatter(points, marker=None, mode='markers', color='red', symbol=None, size=2, opacity=1.0, name=None, showlegend=True): 
    dim = points.shape[1]
    if marker is None:
        markerr = dict(
            size=size,
            color=color,
            symbol=symbol,
            opacity=opacity  ,
        )
    else:
        markerr = marker 
        if 'opacity' not in markerr:
            markerr['opacity'] = opacity

    if dim == 3:
        return go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode=mode,
            marker=markerr,
            name=name,
            showlegend=showlegend
        ) 
    else:
        return go.Scatter(
            x=points[:, 0],
            y=points[:, 1], 
            mode=mode,
            marker=markerr,
            name=name,
            showlegend=showlegend,
        )



def Lines_plot(v1,v2):
    dim = v1.shape[1]
    x = np.column_stack((v1[:, 0], v2[:, 0])).flatten()
    y = np.column_stack((v1[:, 1], v2[:, 1])).flatten()
    if dim ==3:
        z = np.column_stack((v1[:, 2], v2[:, 2])).flatten()
 
    x = np.insert(x, slice(2, None, 2), None)
    y = np.insert(y, slice(2, None, 2), None)
    if dim == 3:
        z = np.insert(z, slice(2, None, 2), None)
        return np.column_stack((x,y,z))
    else:
        return  np.column_stack((x,y))


def plotly_lines(points,points_proj,dash='dash',color='grey',width=2,mode='lines',showlegend=False): 
    connecting_lines = []
    for i in range(points.shape[0]):
        connecting_lines.append(go.Scatter3d(
            x=[points[i, 0], points_proj[i, 0]],
            y=[points[i, 1], points_proj[i, 1]],
            z=[points[i, 2], points_proj[i, 2]],
            mode=mode,
            line=dict(color=color, width=width, dash=dash),
            showlegend=showlegend,
        ))
    return connecting_lines
  



import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import numpy as np

def create_scatter_plot(cluu,dend, geo_cluster ): 
    geo_cluster.Cluster_PCA(cluster_index=cluu)
    geo_cluster.Lines()

    intensity = dend.mean_curv()[cluu]
    vertices_ = dend.vertices[cluu]
    point = geo_cluster.cluster_pca_points
    line_points_1 = geo_cluster.pca_line_1
    line_points_2 = geo_cluster.pca_line_2
    line_points_3 = geo_cluster.pca_line_3
    far_point = geo_cluster.cluster_farthest_point
    close_point = geo_cluster.cluster_closest_point

    scatter = [
        plotly_scatter(points=vertices_, color=intensity),
        plotly_scatter(points=line_points_1, color='blue'),
        plotly_scatter(points=line_points_2, color='black'),
        plotly_scatter(points=line_points_3, color='black'),
        plotly_scatter(points=point, color='green'),
        plotly_scatter(points=far_point.reshape(1, -1), color='yellow', size=7),
        plotly_scatter(points=close_point.reshape(1, -1), color='orange', size=7),
        plotly_scatter(points=point[-1].reshape(1, -1), color='grey', size=7),
    ]
    
    return scatter

def create_subplots(clu, clu_new ,dend,geo_cluster , width=500, height=400):
    # Create subplots
    fig = make_subplots(rows=1, 
                        cols=2, 
                        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                        horizontal_spacing=0.0)
 
    scatter_1 = create_scatter_plot(clu, dend,geo_cluster )
    for scatter in scatter_1:
        fig.add_trace(scatter, row=1, col=1)
 
    scatter_2 = create_scatter_plot(clu_new, dend,geo_cluster )
    for scatter in scatter_2:
        fig.add_trace(scatter, row=1, col=2)

    # Update layout
    fig.update_layout(
        width=2 * width,  
        height=height
    )

    return fig

 