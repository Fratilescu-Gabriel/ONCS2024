import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
DEFAULT_COLORS = ['b', 'g', 'r', 'y', 'c', 'm', 'k']

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Use 'TkAgg' backend
matplotlib.use('TkAgg')


class Plotter:
    """
    A versatile plotting class that simplifies creating and managing multiple figures with matplotlib.

    Key Features:
        - Create multiple figures with minimal setup.
        - Add different plot types: lines, scatter plots, stars, and interpolated lines.
        - Customize the plot: grids, styles, legends, axis labels, and title.
        - Save and show plots easily.
    """
    
    def __init__(self, style='ggplot', grid=True):
        """Initialize the class with global style and grid settings."""
        plt.style.use(style)
        self.grid_enabled = grid
        self.figures = []
        self.current_figure = None

    def new_figure(self, title=None, xlabel=None, ylabel=None, legend=True):
        """
        Create a new figure with the given properties.

        Args:
            title (str, optional): Title for the figure. Defaults to None.
            xlabel (str, optional): Label for the x-axis. Defaults to None.
            ylabel (str, optional): Label for the y-axis. Defaults to None.
            legend (bool, optional): Whether to show the legend. Defaults to True.

        Example:
            new_figure("Example Plot", "Time (s)", "Amplitude")
        """
        fig, ax = plt.subplots()
        ax.grid(self.grid_enabled, linestyle='--', color='gray', alpha=0.7)
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if legend:
            ax.legend()

        self.figures.append((fig, ax))
        self.current_figure = (fig, ax)

    def add_line_plot(self, x, y, label=None, linestyle='-', color=None, marker=None):
        """Add a line plot to the current figure."""
        _, ax = self.current_figure
        ax.plot(x, y, label=label, linestyle=linestyle, color=color, marker=marker)

    def add_scatter_plot(self, x, y, label=None, color=None, marker='o'):
        """Add a scatter plot to the current figure."""
        _, ax = self.current_figure
        ax.scatter(x, y, label=label, color=color, marker=marker)

    def add_stars(self, x, y, label=None, color='gold'):
        """Add star markers to the current figure to highlight specific points."""
        _, ax = self.current_figure
        ax.scatter(x, y, label=label, color=color, marker='*', s=150)

    def add_interpolated_line(self, x, y, label=None, kind='linear', num_points=500, linestyle='--', color=None):
        """Add an interpolated line to the current figure."""
        _, ax = self.current_figure
        interpolator = interp1d(x, y, kind=kind)
        x_new = np.linspace(min(x), max(x), num_points)
        y_new = interpolator(x_new)
        ax.plot(x_new, y_new, label=label, linestyle=linestyle, color=color)

    def show_legend(self, location='upper right'):
        """Show the legend on the current figure."""
        _, ax = self.current_figure
        ax.legend(loc=location)

    def show(self):
        """Display all figures."""
        plt.show()

    def save(self, filename='plot.png', dpi=300):
        """
        Save the current figure as an image file.

        Args:
            filename (str, optional): Name of the file to save. Defaults to 'plot.png'.
            dpi (int, optional): Dots per inch for the saved file. Defaults to 300.
        """
        fig, _ = self.current_figure
        fig.savefig(filename, dpi=dpi)




class Plotter3D:
    """
    A comprehensive 3D plotting class using matplotlib.

    Key Features:
        - Multiple 3D plot types: scatter, wireframe, surface, contour, bars.
        - Adjustable viewing angles, colors, labels, and titles.
        - Efficient and user-friendly API.
    """
    
    def __init__(self, style='ggplot'):
        """Initialize the class with a global style setting."""
        plt.style.use(style)
        self.figures = []
        self.current_figure = None

    def new_figure(self, title=None, xlabel=None, ylabel=None, zlabel=None):
        """
        Create a new 3D figure with the given properties.

        Args:
            title (str, optional): Title for the figure. Defaults to None.
            xlabel (str, optional): Label for the x-axis. Defaults to None.
            ylabel (str, optional): Label for the y-axis. Defaults to None.
            zlabel (str, optional): Label for the z-axis. Defaults to None.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if zlabel:
            ax.set_zlabel(zlabel)

        self.figures.append((fig, ax))
        self.current_figure = (fig, ax)

    def add_scatter_plot(self, x, y, z, label=None, color=None, marker='o'):
        """Add a 3D scatter plot to the current figure."""
        _, ax = self.current_figure
        ax.scatter(x, y, z, label=label, color=color, marker=marker)

    def add_wireframe(self, x, y, z, label=None, color='blue'):
        """Add a 3D wireframe plot to the current figure."""
        _, ax = self.current_figure
        ax.plot_wireframe(x, y, z, color=color, label=label)

    def add_surface(self, x, y, z, cmap='viridis'):
        """Add a 3D surface plot to the current figure."""
        _, ax = self.current_figure
        ax.plot_surface(x, y, z, cmap=cmap)

    def add_contour(self, x, y, z, zdir='z', offset=0, cmap='coolwarm'):
        """Add a 3D contour plot to the current figure."""
        _, ax = self.current_figure
        ax.contour3D(x, y, z, cmap=cmap, offset=offset, zdir=zdir)

    def add_bar_chart(self, x, y, z, dx=0.1, dy=0.1, dz=None, color='cyan'):
        """Add a 3D bar chart to the current figure."""
        _, ax = self.current_figure
        if dz is None:
            dz = [1] * len(x)
        ax.bar3d(x, y, z, dx, dy, dz, color=color)

    def add_vectors(self, x, y, z, u, v, w, label=None, color='green'):
        """Add 3D vectors to the current figure."""
        _, ax = self.current_figure
        ax.quiver(x, y, z, u, v, w, color=color, label=label)

    def adjust_view(self, elev=20, azim=30):
        """Adjust the viewing angles of the current 3D plot."""
        _, ax = self.current_figure
        ax.view_init(elev=elev, azim=azim)

    def show_legend(self, location='upper right'):
        """Show the legend on the current figure."""
        _, ax = self.current_figure
        ax.legend(loc=location)

    def show(self):
        """Display all figures."""
        plt.show()

    def save(self, filename='plot3d.png', dpi=300):
        """
        Save the current figure as an image file.

        Args:
            filename (str, optional): Name of the file to save. Defaults to 'plot3d.png'.
            dpi (int, optional): Dots per inch for the saved file. Defaults to 300.
        """
        fig, _ = self.current_figure
        fig.savefig(filename, dpi=dpi)
