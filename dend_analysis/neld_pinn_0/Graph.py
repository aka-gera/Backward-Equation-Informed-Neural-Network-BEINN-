 
import numpy as np
import plotly.graph_objects as go
import scipy
from scipy import stats
import plotly.io as pio
from plotly.subplots import make_subplots



 
class mygraph:
    def __init__(self,tcouleur,bcouleur,fcouleur,fsize):
        self.tcouleur = tcouleur
        self.bcouleur = bcouleur
        self.fcouleur = fcouleur
        self.fsize = fsize


    def plot_history_save_video(self, Q1, Q2, dt, bheight, bwidth, filename):
        m = Q1.shape[0]

        QQ1 = Q1.T
        QQ2 = Q2.T
        xa = np.min(np.min(QQ1))
        xb = np.max([np.max(np.max(QQ1)),1])
        ya = np.min(np.min(QQ2))
        yb = np.max([np.max(np.max(QQ2)),1]) 

        minColorLimit = 0
        maxColorLimit = 1000 

        frames = []
        frame_titles = []


        x = Q1[0, :]
        y = Q2[0, :]
        
        trace = go.Histogram2d(
            x=x,
            y=y,
            autobinx=False,
            xbins=dict(start=xa, end=xb, size=0.1),
            autobiny=False,
            ybins=dict(start=ya, end=yb, size=0.1),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
        title="Counts",
                # titleside="top",
                tickmode="array",
                tickvals=list(range(int(minColorLimit), int(maxColorLimit)))),
        ) 

        fig = go.Figure(data=[trace],
                        layout=dict(xaxis=dict(range=[xa, xb]),
                                    yaxis=dict(range=[ya, yb]),
                                    showlegend=False, )
                        ) 

        fig.update_layout(
            title=f'Time {dt:.2f}'
        )


        for i in range(m):
            x = Q1[i, :]
            y = Q2[i, :]
            
            trace = go.Histogram2d(
                x=x,
                y=y,
                autobinx=False,
                xbins=dict(start=xa, end=xb, size=0.1),
                autobiny=False,
                ybins=dict(start=ya, end=yb, size=0.1),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    titleside="top",
                    tickmode="array",
                    tickvals=list(range(int(minColorLimit), int(maxColorLimit)))),
            )
 
            frames.append({'data': [trace]})

            frame_titles.append(f'Time {dt*i:.2f}') 
            # fig.add_trace(trace)

        # Create subplots
        # fig = make_subplots(rows=1, cols=1, subplot_titles=[f'Time {0:.2f}'])
 
        fig.update_layout(
                barmode='overlay',
            # paper_bgcolor=bcouleur,  
            font=dict(color=self.fcouleur,size=self.fsize),  # Set the font color 
            title_x=0.5,
            title_y=0.95,
            template=self.tcouleur,
            autosize=False,
            height=bheight,
            width=bwidth,
            hovermode='closest', 
        )

        for i, _ in enumerate(frame_titles):
            frames[i].update(layout=dict(title=f'Time {dt*i:.2f}'))
 
        fig.update(frames=frames)
        
        fig.update_layout(
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(label='Play',
                            method='animate',
                            args=[None, dict(frame=dict(duration=1000, redraw=True), fromcurrent=True, mode='immediate')]),
                        dict(label='Pause',  # Add a pause button
                            method='animate',
                            args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
                    ],
                    x=0.1,
                    xanchor='right',
                    y=1.2,
                    yanchor='top',
                )
            ],
        )

        fig.update_xaxes(
            title_text='x1-x2',
            title_font = {"size": 18},
            title_standoff = 25,
            side='bottom')
        fig.update_yaxes(
                title_text = 'y1-y2',
                title_font = {"size": 18},
                title_standoff = 25)
        ###################################################################
        fig.data[0].visible = True

        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="restyle",
                args=["visible", [False] * len(fig.data)],
                label=str(i),
            )
            step["args"][1][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)

        sliders = [dict(
            active=10,
            currentvalue={"prefix": "Frequency: "},
            pad={"t": 50},
            steps=steps
        )]
 
        fig.update_layout(sliders=sliders)
        #########################################################################

        # Write the animation to an HTML file
        fig.write_html(file=filename, auto_open=False) 









        

    def HistoryMatrix(self,sav,Ndata,ShowTime):

        QQ1 = sav['Q1'].T
        QQ2 = sav['Q2'].T
        xa = np.min(np.min(QQ1))
        xb =  np.max([np.max(np.max(QQ1)),1])
        ya = np.min(np.min(QQ2))
        yb =  np.max([np.max(np.max(QQ2)),1])
        xedges = np.linspace(xa, xb, Ndata+1)
        yedges = np.linspace(ya, yb, Ndata+1)

        minColorLimit = 0
        maxColorLimit = 0
        histMa = np.zeros((Ndata, Ndata, len(ShowTime)))

        for k in ShowTime:
            # histmat, _ , _,_ = binned_statistic_2d(QQ1[:, k], QQ2[:, k], values=None, statistic='count', bins=[xedges, yedges])
            ret = stats.binned_statistic_2d(QQ1[:, k], QQ2[:, k], values=None, statistic='count', bins=[xedges, yedges])
            histmat = ret.statistic
            minColorLimit = min([np.min(np.min(histmat)), minColorLimit])
            maxColorLimit = max([np.max(np.max(histmat)), maxColorLimit])
            histMa[:, :, k] = histmat.T

        return histMa,minColorLimit,maxColorLimit,xa,xb,ya,yb


    def plot_history_vector_timed(self,Q1,Q2,dt,bheight,bwidth):
        
        QQ1 = Q1.T
        QQ2 = Q2.T
        xa = np.min(np.min(QQ1))
        xb = np.max([np.max(np.max(QQ1)),1])
        ya = np.min(np.min(QQ2))
        yb = np.max([np.max(np.max(QQ2)),1]) 

        minColorLimit = 0
        maxColorLimit = 1000
        
        m = Q1.shape[0]
        frames = []
        frame_titles = []


        x = Q1[0, :]
        y = Q2[0, :]
        
        trace = go.Histogram2d(
            x=x,
            y=y,
            autobinx=False,
            xbins=dict(start=xa, end=xb, size=0.1),
            autobiny=False,
            ybins=dict(start=ya, end=yb, size=0.1),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                titleside="top",
                tickmode="array",
                tickvals=list(range(int(minColorLimit), int(maxColorLimit)))),
        ) 

        fig = go.Figure(data=[trace],
                        layout=dict(xaxis=dict(range=[xa, xb]),
                                    yaxis=dict(range=[ya, yb]),
                                    showlegend=False, )
                        ) 

        fig.update_layout(
            title=f'Time {dt:.2f}'
        )


        for i in range(m):
            x = Q1[i, :]
            y = Q2[i, :]
            
            trace = go.Histogram2d(
                x=x,
                y=y,
                autobinx=False,
                xbins=dict(start=xa, end=xb, size=0.1),
                autobiny=False,
                ybins=dict(start=ya, end=yb, size=0.1),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    titleside="top",
                    tickmode="array",
                    tickvals=list(range(int(minColorLimit), int(maxColorLimit)))),
            )
 
            frames.append({'data': [trace]})

            frame_titles.append(f'Time {dt*i:.2f}') 
            # fig.add_trace(trace)
 
        fig.update_layout(
                barmode='overlay',
            # paper_bgcolor=bcouleur,  
            font=dict(color=self.fcouleur,size=self.fsize),  # Set the font color 
            title_x=0.5,
            title_y=0.95,
            template=self.tcouleur,
            autosize=False,
            height=bheight,
            width=bwidth,
            hovermode='closest', 
        )

        for i, _ in enumerate(frame_titles):
            frames[i].update(layout=dict(title=f'Time {dt*i:.2f}'))
 
        fig.update(frames=frames)
        
        fig.update_layout(
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(label='Play',
                            method='animate',
                            args=[None, dict(frame=dict(duration=0, redraw=True), fromcurrent=True, mode='immediate')]),
                        dict(label='Pause',  # Add a pause button
                            method='animate',
                            args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
                    ],
                    x=0.1,
                    xanchor='right',
                    y=1.2,
                    yanchor='top',
                )
            ],
        )

        fig.update_xaxes(
            title_text='x1-x2',
            title_font = {"size": 18},
            title_standoff = 25,
            side='bottom')
        fig.update_yaxes(
                title_text = 'y1-y2',
                title_font = {"size": 18},
                title_standoff = 25)
        ###################################################################
        fig.data[0].visible = True

        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="restyle",
                args=["visible", [False] * len(fig.data)],
                label=str(i),
            )
            step["args"][1][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)

        sliders = [dict(
            active=10,
            currentvalue={"prefix": "Frequency: "},
            pad={"t": 50},
            steps=steps
        )]
 
        fig.update_layout(sliders=sliders)
        #########################################################################


        return fig
    

    def plot_history_matrixxy(self,Q1,Q2,time,bheight,bwidth):
        
        QQ1 = Q1.T
        QQ2 = Q2.T
        xa = np.min(np.min(QQ1))
        xb = np.max(np.max(QQ1))
        ya = np.min(np.min(QQ2))
        yb = np.max(np.max(QQ2))

        minColorLimit = 0
        maxColorLimit = 1000
        
        m = Q1.shape[0]
        frames = []
        frame_titles = []


        x = Q1[0, :]
        y = Q2[0, :]
        
        trace = go.Histogram2d(
            x=x,
            y=y,
            autobinx=False,
            xbins=dict(start=xa, end=xb, size=0.1),
            autobiny=False,
            ybins=dict(start=ya, end=yb, size=0.1),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Counts",
                # titleside="top",
                tickmode="array",
                tickvals=list(range(int(minColorLimit), int(maxColorLimit)))),
        ) 

        fig = go.Figure(data=[trace],
                        layout=dict(xaxis=dict(range=[xa, xb]),
                                    yaxis=dict(range=[ya, yb]),
                                    showlegend=False, )
                        ) 

        fig.update_layout(
            title=f'Time {time[0,0]:.2f}'
        )


        for i in range(1,m):
            x = Q1[i, :]
            y = Q2[i, :]
            
            trace = go.Histogram2d(
                x=x,
                y=y,
                autobinx=False,
                xbins=dict(start=xa, end=xb, size=0.1),
                autobiny=False,
                ybins=dict(start=ya, end=yb, size=0.1),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Counts",
                    # titleside="top",
                    tickmode="array",
                    tickvals=list(range(int(minColorLimit), int(maxColorLimit)))),
            )
 
            frames.append({'data': [trace]})

            frame_titles.append(f'Time {time[i,0]:.2f}') 
            # fig.add_trace(trace)
 
        fig.update_layout(
                barmode='overlay',
            # paper_bgcolor=bcouleur,  
            font=dict(color=self.fcouleur,size=self.fsize),  # Set the font color 
            title_x=0.5,
            title_y=0.95,
            template=self.tcouleur,
            autosize=False,
            height=bheight,
            width=bwidth,
            hovermode='closest', 
        )

        for i, _ in enumerate(frame_titles):
            frames[i].update(layout=dict(title=f'Time {time[i,0]:.2f}'))
 
        fig.update(frames=frames)
        
        fig.update_layout(
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(label='Play',
                            method='animate',
                            args=[None, dict(frame=dict(duration=0, redraw=True), fromcurrent=True, mode='immediate')]),
                        dict(label='Pause',  # Add a pause button
                            method='animate',
                            args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
                    ],
                    x=0.1,
                    xanchor='right',
                    y=1.2,
                    yanchor='top',
                )
            ],
        )

        fig.update_xaxes(
            title_text='x1-x2',
            title_font = {"size": 18},
            title_standoff = 25,
            side='bottom')
        fig.update_yaxes(
                title_text = 'y1-y2',
                title_font = {"size": 18},
                title_standoff = 25)
        ###################################################################
        fig.data[0].visible = True

        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="restyle",
                args=["visible", [False] * len(fig.data)],
                label=str(i),
            )
            step["args"][1][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)

        sliders = [dict(
            active=10,
            currentvalue={"prefix": "Frequency: "},
            pad={"t": 0},
            steps=steps
        )]
 
        fig.update_layout(sliders=sliders)
        #########################################################################


        return fig
    
    def plot_history_matrixxy2(self,PBCs,sav,dat,param,pbc,bheight,bwidth):
        
        QQ1 = sav['Q1'].T
        QQ2 = sav['Q2'].T
        
        # Calculate L matrix
        L =  pbc['L0']

        q1 = QQ1[:, 0].reshape(-1,1)
        q2 = QQ2[:, 0].reshape(-1,1)
        q3 = np.zeros((QQ1.shape[0],1))
        
        qqX = np.hstack((q1, q2, q3)).T
        qq = PBCs.data_replicas(L, qqX, dat, param) 
        x =qq[0,:].reshape(-1) 
        y =qq[1,:].reshape(-1) 
        xa = -dat['xmax']# np.min(np.min(x))
        xb =  dat['xmax']# np.max(np.max(x)) 
        ya = - dat['ymax']#np.min(np.min(y))
        yb = dat['ymax']#np.max(np.max(y))  

        minColorLimit = 0
        maxColorLimit = 1000

        m = sav['Q1'].shape[0]
        frames = []
        frame_titles = []
        
        # Create a heatmap trace for the frame
        trace = go.Histogram2d(
            x=x,
            y=y,
            autobinx=False,
            xbins=dict(start=xa, end=xb, size=0.1),
            autobiny=False,
            ybins=dict(start=ya, end=yb, size=0.1),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                    title="Counts",
                # titleside="top",
                tickmode="array",
                tickvals=list(range(int(minColorLimit), int(maxColorLimit)))),
        ) 

        fig = go.Figure(data=[trace],
                        layout=dict(xaxis=dict(range=[xa, xb]),
                                    yaxis=dict(range=[ya, yb]),
                                    showlegend=False, )
                        ) 

        fig.update_layout(
            title=f'Time {0:.2f}'
        )
 
        for i in range(m):
            
            theta = i * pbc['dt'] - np.floor(i * pbc['dt'] / pbc['T']) * pbc['T']

            # Calculate L matrix
            L = scipy.linalg.expm(theta * pbc['A']).dot(pbc['L0'])

            q1 = QQ1[:, i].reshape(-1,1)
            q2 = QQ2[:, i].reshape(-1,1) 

            qqX = np.hstack((q1, q2, q3)).T
            qq = PBCs.data_replicas(L, qqX, dat, param) 
            x =qq[0,:].reshape(-1) 
            y =qq[1,:].reshape(-1)  
            
            trace = go.Histogram2d(
                x=x,
                y=y,
                autobinx=False,
                xbins=dict(start=xa, end=xb, size=0.1),
                autobiny=False,
                ybins=dict(start=ya, end=yb, size=0.1),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Counts",
                    # titleside="top",
                    tickmode="array",
                    tickvals=list(range(int(minColorLimit), int(maxColorLimit)))),
            )

        
            frames.append({'data': [trace]})
            dt =pbc['dt']
            frame_titles.append(f'Time { dt*i:.2f}') 
            # fig.add_trace(trace)
    
        fig.update_layout(
                barmode='overlay',
            # paper_bgcolor=bcouleur,  
            font=dict(color=self.fcouleur,size=self.fsize),  # Set the font color 
            title_x=0.5,
            title_y=0.95,
            template=self.tcouleur,
            autosize=False,
            height=bheight,
            width=bwidth,
            hovermode='closest', 
        )

        for i, _ in enumerate(frame_titles):
            frames[i].update(layout=dict(title=f'Time {dt*i:.2f}'))


        fig.update(frames=frames)
        
        fig.update_layout(
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(label='Play',
                            method='animate',
                            args=[None, dict(frame=dict(duration=0, redraw=True), fromcurrent=True, mode='immediate')]),
                        dict(label='Pause',  # Add a pause button
                            method='animate',
                            args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
                    ],
                    x=0.1,
                    xanchor='right',
                    y=1.2,
                    yanchor='top',
                )
            ],
        )

        fig.update_xaxes(
            title_text='x1-x2',
            title_font = {"size": 18},
            title_standoff = 25,
            side='bottom')
        fig.update_yaxes(
                title_text = 'y1-y2',
                title_font = {"size": 18},
                title_standoff = 25)
        ###################################################################
        fig.data[0].visible = True

        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="restyle",
                args=["visible", [False] * len(fig.data)],
                label=str(i),
            )
            step["args"][1][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)

        sliders = [dict(
            active=10,
            currentvalue={"prefix": "Frequency: "},
            pad={"t": 50},
            steps=steps
        )]
        

        fig.update_layout(sliders=sliders)
        #########################################################################


        return fig


