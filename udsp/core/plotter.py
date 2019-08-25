"""
Plotting facilities

"""

try:
    import matplotlib.pyplot as _plt
    import numpy as _np
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    _HAS_MATPLOTLIB = False
else:
    _HAS_MATPLOTLIB = True

import warnings

from random import uniform as _rnd
from . import utils as _utl
from . import mtx as _mtx
from ..signal.base import Signal


if _HAS_MATPLOTLIB:

    class Plotter(object):
        """
        Abstract base class for signal plotters

        Attributes
        ----------
        _signals: list[list[Signal]]
            The signals to be plotted. This is a 2D array specifying the
            grid-layout of the plots.
        _layout: {"single","pip","grid"}
            The plot layout
        title: str
            The plot title
        shared: bool
            If true then the plots share the same x or y axis. Specifically,
            if the plots are laid out in one row then they all share the y
            axis of the first plot, while if all the plots are laid out in
            one column then they all share the x axis of the first plot.
        grid: bool
            Whether to show the plot's grid

        """

        _LAYOUT_SINGLE = "single"
        _LAYOUT_PIP = "pip"
        _LAYOUT_GRID = "grid"

        _DEFAULT_LINE_COLOR = (0.0, 0.0, 0.0)
        _DEFAULT_LINE_STYLE = "solid"
        _DEFAULT_LINE_WIDTH = 2
        _DEFAULT_GRID_COLOR = (0.8, 0.8, 0.8)
        _DEFAULT_GRID_STYLE = "dashed"
        _DEFAULT_GRID_WIDTH = 1
        _DEFAULT_BG_COLOR = (0.0, 0.0, 0.0, 0.0)
        _DEFAULT_SURF_COLOR = (1.0, 1.0, 1.0)
        _DEFAULT_EDGE_COLOR = (0.0, 0.0, 0.0)

        def __init__(self, signals=None, **kwargs):
            """
            Creates a signal plotter

            Parameters
            ----------
            signals: None, Signal, list[Signal], list[list[Signal]]
                An optional list of signals (or a single signal) to be
                plotted. Signals can be added to the plotter after creation
                too using the set() method. If it's a list of signals
                then they will be plotted in plot-in-plot mode (multiplot).
                If it's a 2D array of signals then they will be plotted
                according to the layout specified by the grid.
            kwargs:
                Optional arguments

            """
            super().__init__()
            self._layout = None
            self._signals = self._make_layout(signals)
            self.title = ""
            self.shared = False
            self.grid = False

        def set(self, signals):
            """
            Sets signals in the plotter

            Parameters
            ----------
            signals: Signal, list[Signal], list[list[Signal]]
                A list of signals (or a single signal) to be plotted.
                If it's a list of signals then they will be plotted in
                plot-in-plot mode (multiplot). If it's a 2D array of signals
                then they will be plotted according to the layout specified
                in the grid.

            Returns
            -------
            Plotter
                This plotter's instance

            """
            self._signals = self._make_layout(signals)
            return self

        def graph(self):
            """
            Plots a signal as an x-vs-y graph

            Returns
            -------
            None

            """
            raise NotImplementedError

        def map(self):
            """
            Plots a signal as a 2D scalar field

            Returns
            -------
            None

            """
            raise NotImplementedError

        def _make_layout(self, signals):
            """
            Arranges the signals in a grid layout

            This method takes the signal(s) passed by clients and arranges
            them in a grid layout. Signals may be passed as a single instance,
            as a list, or as a grid layout.

            Parameters
            ----------
            signals: None, Signal, list[Signal], list[list[Signal]]

            Returns
            -------
            list[list[Signal]]

            """
            if signals:
                # single plot
                if isinstance(signals, Signal):
                    signals = [[signals]]
                    self._layout = self._LAYOUT_SINGLE
                # plot-in-plot
                elif (isinstance(signals, list) and
                      isinstance(signals[0], Signal)):
                    signals = [signals]
                    self._layout = self._LAYOUT_PIP
                # grid plot
                elif (isinstance(signals, list) and
                      isinstance(signals[0], list)):
                    # check whether it's a valid grid layout
                    # (all rows with equal size and not empty)
                    for row in signals:
                        if (len(signals[0]) == 0 or
                             len(row) != len(signals[0])):
                            raise ValueError("Invalid layout")

                    # or, more "Pythonic" but more obscure
                    # ns = sum(map(lambda v: len(v), signals))
                    # if ns == 0 or ns % len(signals) or ns % len(signals[0]):
                    #     raise ValueError("Invalid layout")
                    self._layout = self._LAYOUT_GRID
                else:
                    signals = None

            return signals


    class Plotter1D(Plotter):
        """
        Specialized class for 1D signal plotting

        """
        def __init__(self, signals, **kwargs):
            super().__init__(signals, **kwargs)

        def graph(self):

            if not self._signals:
                raise ValueError("No signal to plot")

            nrows, ncols = len(self._signals), len(self._signals[0])

            # Plots with shared axes. Currently only possible if they
            # are in a row (shared y) or in a column (shared x).
            is_shared_y = self.shared and nrows == 1 and ncols > 1
            is_shared_x = self.shared and ncols == 1 and nrows > 1

            fig = _plt.figure()

            for i in range(nrows):
                for j in range(ncols):

                    if self._signals[i][j] is None:
                        continue

                    if self._signals[i][j].is_empty():
                        raise ValueError(
                            "No signal (%s)" % self._signals[i][j].name
                        )
                    nplot = i * ncols + j + 1

                    # default plot attributes
                    legend = None
                    title = self._signals[i][j].name
                    xlabel = self._signals[i][j].xunits
                    ylabel = self._signals[i][j].yunits
                    line_color = self._DEFAULT_LINE_COLOR
                    line_style = self._DEFAULT_LINE_STYLE
                    line_width = self._DEFAULT_LINE_WIDTH
                    grid_color = self._DEFAULT_GRID_COLOR
                    grid_style = self._DEFAULT_GRID_STYLE
                    grid_width = self._DEFAULT_GRID_WIDTH
                    shared_x = None
                    shared_y = None

                    ax = None

                    if self._layout == self._LAYOUT_GRID:
                        if is_shared_y:
                            shared_y = fig.axes[0] if nplot > 1 else None
                        if is_shared_x:
                            shared_x = fig.axes[0] if nplot > 1 else None
                        ax = fig.add_subplot(nrows, ncols, nplot,
                                             sharex=shared_x,
                                             sharey=shared_y)
                    elif self._layout == self._LAYOUT_SINGLE:
                        ax = fig.add_subplot(nrows, ncols, nplot)
                    elif self._layout == self._LAYOUT_PIP:
                        if not ax:
                            ax = fig.add_subplot(nrows, ncols, nplot)
                        # give the first plot default color and
                        # the rest random ones. Also, assume all
                        # signals have the same x-y units.
                        if nplot > 1:
                            line_color = _rnd(0, 1), _rnd(0, 1), _rnd(0, 1)
                            xlabel = ylabel = ""
                        legend = self._signals[i][j].name or "Unknown"
                        title = self.title
                    else:
                        raise RuntimeError

                    ys, xs = self._signals[i][j].get(alls=True)

                    ax.plot(xs, ys,
                            color=line_color,
                            linestyle=line_style,
                            linewidth=line_width,
                            label=legend)

                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
                    ax.set_title(title)

                    if legend:
                        ax.legend()
                    if self.grid:
                        ax.grid(color=grid_color,
                                linestyle=grid_style,
                                linewidth=grid_width)

                    if is_shared_x and nplot != nrows:
                        _plt.setp(ax.get_xticklabels(), visible=False)
                    if is_shared_y and nplot != 1:
                        _plt.setp(ax.get_yticklabels(), visible=False)
            _plt.show()

        def map(self):
            raise TypeError(
                "This signal type does not support maps"
            )


    class Plotter2D(Plotter):
        """
        Specialized class for 2D signal plotting

        Attributes
        ----------
        zrange: tuple, None, optional
            The z-axis range the signal will be plotted in. If not specified
            it will be automatically determined.
        stride: tuple, optional
            The step between points (determines the "density" of the mesh)
        colormap: str
            A colormap name used to render maps

        """
        def __init__(self, signals, **kwargs):

            super().__init__(signals, **kwargs)
            self.zrange = None
            self.stride = (1, 1)
            self.colormap = "gray"

        def graph(self):

            if not self._signals:
                raise ValueError("No signal to plot")

            nrows, ncols = len(self._signals), len(self._signals[0])

            fig = _plt.figure()

            for i in range(nrows):
                for j in range(ncols):

                    if self._signals[i][j] is None:
                        continue

                    if self._signals[i][j].is_empty():
                        raise ValueError(
                            "No signal (%s)" % self._signals[i][j].name
                        )
                    nplot = i * ncols + j + 1

                    # default plot attributes
                    # legend = None
                    title = self._signals[i][j].name
                    xlabel = self._signals[i][j].xunits[0]
                    ylabel = self._signals[i][j].xunits[1]
                    zlabel = self._signals[i][j].yunits[0]
                    edge_color = self._DEFAULT_EDGE_COLOR
                    surf_color = self._DEFAULT_SURF_COLOR

                    if self._layout == self._LAYOUT_GRID:
                        ax = fig.add_subplot(nrows, ncols, nplot,
                                             projection='3d')
                    elif self._layout == self._LAYOUT_SINGLE:
                        ax = fig.gca(projection='3d')
                    elif self._layout == self._LAYOUT_PIP:
                        ax = fig.gca(projection='3d')
                        # Give the first plot default color and the rest
                        # random ones. Add some transparency so they blend.
                        # Also, assume all signals have the same x-y units.
                        if nplot == 1:
                            edge_color = self._DEFAULT_EDGE_COLOR + (0.05,)
                            surf_color = self._DEFAULT_SURF_COLOR + (0.3,)
                        else:
                            rcol = _rnd(0, 1), _rnd(0, 1), _rnd(0, 1)
                            edge_color = rcol + (0.1,)
                            surf_color = rcol + (0.3,)
                            xlabel = ylabel = zlabel = ""
                        # if nplot == 1:
                        #     edge_color = (0.7, 0.7, 0.7, 0.7)
                        #     surf_color = (0, 0, 0, 0.1)
                        # else:
                        #     edge_color = (0.7, 0.7, 0.7, 0.7)
                        #     surf_color = (0, 0, 0, 0.1)
                        #     xlabel = ylabel = ""
                        title = self.title
                    else:
                        raise RuntimeError

                    ys, xs = self._signals[i][j].get(alls=True)
                    X, Y = _utl.to_meshgrid(xs)
                    Z = ys

                    if not self.zrange or _utl.all_same(0, self.zrange):
                        zmin, zmax = _mtx.mat_min_max(Z)
                        self.zrange = (
                            zmin - (zmax - zmin) / 10,
                            zmax + (zmax - zmin) / 10
                        )

                    X, Y, Z = _np.array(X), _np.array(Y), _np.array(Z)

                    ax.plot_surface(X, Y, Z,
                                    rstride=self.stride[0],
                                    cstride=self.stride[1],
                                    color=surf_color,
                                    shade=False,
                                    edgecolor=edge_color)

                    ax.set_title(title)
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
                    ax.set_zlabel(zlabel)
                    ax.set_zlim(self.zrange[0], self.zrange[1])
                    ax.w_xaxis.set_pane_color(self._DEFAULT_BG_COLOR)
                    ax.w_yaxis.set_pane_color(self._DEFAULT_BG_COLOR)
                    ax.w_zaxis.set_pane_color(self._DEFAULT_BG_COLOR)
                    ax.grid(False)

            _plt.show()

        def map(self):

            if not self._signals:
                raise ValueError("No signal to plot")

            nrows, ncols = len(self._signals), len(self._signals[0])

            for i in range(nrows):
                for j in range(ncols):

                    if self._signals[i][j] is None:
                        continue

                    if self._signals[i][j].is_empty():
                        raise ValueError(
                            "No signal (%s)" % self._signals[i][j].name
                        )
                    nplot = i * ncols + j + 1

                    # default plot attributes
                    # legend = None
                    title = self._signals[i][j].name
                    xlabel = self._signals[i][j].xunits[0]
                    ylabel = self._signals[i][j].xunits[1]
                    malpha = 1

                    if self._layout == self._LAYOUT_GRID:
                        _plt.subplot(nrows, ncols, nplot)
                    elif self._layout == self._LAYOUT_SINGLE:
                        pass
                    elif self._layout == self._LAYOUT_PIP:
                        malpha = 0.5
                    else:
                        raise RuntimeError

                    _plt.imshow(self._signals[i][j].get(),
                                cmap=self.colormap,
                                alpha=malpha)
                    _plt.xlabel(xlabel)
                    _plt.ylabel(ylabel)
                    _plt.title(title)

            _plt.show()

# No Matplotlib
else:
    # Dummies
    class Plotter(object):
        def __init__(self, arg):
            warnings.warn("Plotting is disabled. Matplotlib is required "
                          "to use this functionality")

    class Plotter1D(Plotter):
        pass

    class Plotter2D(Plotter):
        pass
