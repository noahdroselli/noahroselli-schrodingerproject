import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

class animate_heat_map():
    def __init__(self, c, diff, dt):
        fig = plt.figure(figsize=(10,8))

        d = np.real(c.data * np.conj(c.data))
        norm = np.linalg.norm(d)
        d = d/norm

        ax = sns.heatmap(d)
        plt.xticks([])
        plt.yticks([])

        def init():
            plt.clf()
            ax = sns.heatmap(d)

        def animate(i):
            plt.clf()
            diff.step(dt)

            d = np.real(c.data * np.conj(c.data))
            norm = np.linalg.norm(d)
            d = d/norm
            ax = sns.heatmap(d)

        anim = animation.FuncAnimation(fig, animate, interval = 1, frames = 30)
        plt.xticks([])
        plt.yticks([])
        plt.show()