import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_clrs
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from scomplex.simulations import ThreeBodySystem
from scomplex.metrics.metrics import DHSIC

from pathlib import Path

def plot_initial_condition(trajectories: np.ndarray, filename:Path) -> None:
    with sns.axes_style('white'):
        with plt.style.context('./plotstyle_presentation.mplstyle'):

            # Get the default color cycle specified in the loaded style sheet
            prop_cycle = plt.rcParams['axes.prop_cycle']

            # Extract the colors from the cycle
            colors = prop_cycle.by_key()['color']

            # Get the first three colors
            c1, c2, c3 = colors[:3]

            r1 = trajectories[:, 0:2]
            #v1 = trajectories[:, 2:4]
            r2 = trajectories[:, 4:6]
            #v2 = trajectories[:, 6:8]
            r3 = trajectories[:, 8:10]
            #v3 = trajectories[:, 10:12]

            fig, ax = plt.subplots()
            fig.set_size_inches(3.375, 3.375)

            ax.plot(r1[0, 0], r1[0, 1], marker='o', markersize=5, color=c1, zorder=3)
            ax.plot(r2[0, 0], r2[0, 1], marker='o', markersize=5, color=c2, zorder=3)
            ax.plot(r3[0, 0], r3[0, 1], marker='o', markersize=5, color=c3, zorder=3)
            
            ax.set_xlim(-2.0, 2.0)
            ax.set_ylim(-2.0, 2.0)
            ax.set_aspect('equal', 'box')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_ylabel(r'$y$')
            ax.set_xlabel(r'$x$')

            fig.tight_layout()

            pdf_filename = filename.parent / Path(filename.stem + "_initial_confition")
            pdf_filename = pdf_filename.with_suffix('.pdf')
            fig.savefig(pdf_filename)





def create_three_body_animation(trajectories: np.ndarray, filename: Path, N: int=500, fps: int=30, state_space_on: bool=True) -> None:
    with sns.axes_style('white'):
        with plt.style.context('./plotstyle_presentation.mplstyle'):

            # Get the default color cycle specified in the loaded style sheet
            prop_cycle = plt.rcParams['axes.prop_cycle']

            # Extract the colors from the cycle
            colors = prop_cycle.by_key()['color']

            # Get the first three colors
            c1, c2, c3 = colors[:3]

            r1 = trajectories[:, 0:2]
            #v1 = trajectories[:, 2:4]
            r2 = trajectories[:, 4:6]
            #v2 = trajectories[:, 6:8]
            r3 = trajectories[:, 8:10]
            #v3 = trajectories[:, 10:12]

            N_frames = len(r1) # number of frames


            fig, ax = plt.subplots()
            fig.set_size_inches(3.375, 3.375)
            if state_space_on:
                ax.plot(*r1.T, ":", color=c1, linewidth=0.7, zorder=1, alpha=0.5)
                ax.plot(*r2.T, ":", color=c2, linewidth=0.7, zorder=1, alpha=0.5)
                ax.plot(*r3.T, ":", color=c3, linewidth=0.7, zorder=1, alpha=0.5)


            point_r1, = ax.plot([], [], marker='o', markersize=5, color=c1, zorder=3)
            point_r2, = ax.plot([], [], marker='o', markersize=5, color=c2, zorder=3)
            point_r3, = ax.plot([], [], marker='o', markersize=5, color=c3, zorder=3)

            
            # Create LineCollections for trails (one per body)
            trail_r1 = LineCollection([], linewidths=2, zorder=2)
            trail_r2 = LineCollection([], linewidths=2, zorder=2)
            trail_r3 = LineCollection([], linewidths=2, zorder=2)
            ax.add_collection(trail_r1)
            ax.add_collection(trail_r2)
            ax.add_collection(trail_r3)

            def get_fading_segments(positions, color, N):
                    start = max(0, len(positions) - N)
                    pts = positions[start:]
                    if len(pts) < 2:
                        return [], []
                    segments = np.stack([pts[:-1], pts[1:]], axis=1)
                    alphas = np.linspace(0.1, 1.0, len(segments))
                    rgb = mpl_clrs.to_rgb(color)
                    rgba = np.tile(rgb, (len(segments), 1))
                    rgba = np.concatenate([rgba, alphas[:, None]], axis=1)
                    return segments, rgba
            
            def init():
                ax.set_xlim(-2.0, 2.0)
                ax.set_ylim(-2.0, 2.0)
                ax.set_aspect('equal', 'box')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_ylabel(r'$y$')
                ax.set_xlabel(r'$x$')

                point_r1.set_data([], [])
                point_r2.set_data([], [])
                point_r3.set_data([], [])
                
                trail_r1.set_segments([])
                trail_r2.set_segments([])
                trail_r3.set_segments([])

                fig.tight_layout()
                return point_r1, point_r2, point_r3, trail_r1, trail_r2, trail_r3


            def update(frame):
                
                segs1, trail_colors1 = get_fading_segments(r1[frame-N:frame], c1, N)
                segs2, trail_colors2 = get_fading_segments(r2[frame-N:frame], c2, N)
                segs3, trail_colors3 = get_fading_segments(r3[frame-N:frame], c3, N)
                trail_r1.set_segments(segs1)
                trail_r2.set_segments(segs2)
                trail_r3.set_segments(segs3)
                trail_r1.set_color(trail_colors1)
                trail_r2.set_color(trail_colors2)
                trail_r3.set_color(trail_colors3)

                point_r1.set_data([r1[frame, 0]],[r1[frame, 1]])
                point_r2.set_data([r2[frame, 0]],[r2[frame, 1]])
                point_r3.set_data([r3[frame, 0]],[r3[frame, 1]])
                return point_r1, point_r2, point_r3, trail_r1, trail_r2, trail_r3


            ani = FuncAnimation(
                fig, 
                update,
                frames=np.arange(N, N_frames), # adjusting to trail length that is supposed to be shown
                init_func=init,
                blit=True,
                interval=50,
            )

            # save as mp4 animation
            mp4_filename = filename.parent / filename.stem
            mp4_filename = mp4_filename.with_suffix('.mp4')
            ani.save(mp4_filename, writer='ffmpeg', fps=fps)


if __name__ == "__main__":

    # set parameters
    parameter_set = {
        "G": 0.9, # coupling strength
        "m1": 1.0, # masses
        "m2": 1.0,
        "m3": 1.0,
    }


    #############################################
    #   unstable initial conditions
    #############################################

    animation_filename = Path("./animations/three_body_stable.mp4")
    metric_filename = Path("./data/three_body_stable_dhsic_joined.csv")

    # initial position and momentum for figure-eight orbit (from Chenciner & Montgomery 2000)
    r1 = np.array([0.97000436, -0.24308753])
    r2 = -r1
    r3 = np.array([0.0, 0.0])

    v1 = np.array([0.4662036850, 0.4323657300])
    v3 = -2 * v1
    # figure eight orbit or stable dynamic is set by the relation between 
    # two bodies
    
    #v2 = 1.03*v1 # stable dynamic
    v2 = v1 # figure eight momentum

    # define initial position
    y0 = np.array([r1, v1, r2, v2, r3, v3]).flatten(order='C')

    # specify duration
    t0 = 0.0
    dt = 0.01
    number_of_steps = 5000
    T = number_of_steps*dt

    # run simulation
    tbs = ThreeBodySystem(parameter_set)
    trajectories = tbs.run(y0, T, t0=t0, dt=dt)

    # create plots and animations
    plot_initial_condition(trajectories, animation_filename)
    create_three_body_animation(trajectories, animation_filename, fps=100, N=500)

    # calculate and save the distance statistic
    r1 = trajectories[:, 0:2]
    v1 = trajectories[:, 2:4]
    r2 = trajectories[:, 4:6]
    v2 = trajectories[:, 6:8]
    r3 = trajectories[:, 8:10]
    v3 = trajectories[:, 10:12]

    dhsic = DHSIC()
    system_state = np.array([r1, v1, r2, v2, r3, v3])
    dhsic_metric = dhsic(system_state)

    # save as panda dataframes instead of npy files
    column_keys = ['r_1', 'v_1', 'r_2', 'v_2', 'r_3', 'v_3']
    df = pd.DataFrame(dhsic_metric, columns=column_keys)
    df.to_csv(metric_filename)