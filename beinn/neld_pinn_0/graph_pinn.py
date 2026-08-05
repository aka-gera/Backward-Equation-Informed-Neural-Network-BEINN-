



import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np


metadata = dict(artist='AKA_GERA' ) 
colmap = 'hot' # 'viridis'
file_name = 'file_name'


class graph_pinn():
    def __init__(self) -> None:
        pass

    def scatter(self,u_sav,x_sav,y_sav,temps,ax_lim,
                file_name=file_name,
                fps=10,
                metadata=metadata,
                figsize=(10,8),
                colmap=colmap):
        
        FFMpegWriter = manimation.writers['ffmpeg']
        writer = FFMpegWriter(fps=fps, metadata=metadata)

        fig = plt.figure(figsize=figsize)
        ax2 = fig.add_subplot(111)
        with writer.saving(fig, f"{file_name}.mp4", 75):
            for i,itm in enumerate(temps):
                ax2.cla()

                ax2.axes.set_xlim(ax_lim['xmin'],ax_lim['xmax'])
                ax2.axes.set_ylim(ax_lim['ymin'],ax_lim['ymax'])
                ax2.set_xticks(np.linspace(ax_lim['xmin'],ax_lim['xmax'],4))
                ax2.set_yticks(np.linspace(ax_lim['ymin'],ax_lim['ymax'],4))
                ax2.set_xlabel("x")
                ax2.set_ylabel("y")


                ax2.set_title(f'Time {itm:.3f}')
                surf=ax2.scatter(x_sav[i], y_sav[i], c=u_sav[i], cmap=colmap,facecolors=None, s=100.0, marker='s',   alpha=0.7, edgecolors='none')

                cbar = fig.colorbar(mappable=surf, ax= ax2, orientation='vertical')
                cbar.set_label('u predicted')
                cbar.mappable.set_clim(ax_lim['zmin'],ax_lim['zmax'])

                writer.grab_frame()
                cbar.remove()

