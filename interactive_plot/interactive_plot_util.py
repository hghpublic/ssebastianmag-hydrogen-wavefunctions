"""
File: interactive_plot/interactive_plot_util.py
Description: Utilities for getting an interactive plot with sliders and preferences visualizing hydrogen wave functions.

Author: Jonathan Hirsch
Date: December 2025
"""
import sys
import os

import numpy as np
import seaborn as sns
from dash import Dash, dcc, html, Output, Input
import plotly.graph_objects as go

# Add the parent folder to sys.path, to import other scripts
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Get absolute path of parent folder
sys.path.insert(0, parent_dir)  # Add parent folder at the front of path. Only changes path during lifetime of process

from hydrogen_wavefunction import compute_psi_xz_slice, compute_probability_density
from hwf_plots import WaveFunction

# Global cache
_wavefunction_cache = {}


def compute_P_and_extent(wf: WaveFunction):
    """
        Compute the probability density |ψ|² for a given hydrogenic WaveFunction
        and return both the computed density grid and the spatial extent in units
        of the reduced-mass Bohr radius a_μ.

        Parameters
        ----------
        wf : WaveFunction
            WaveFunction instance containing the quantum numbers (n, l, m),
            nuclear charge Z, reduced-mass options, grid resolution and extent.

        Returns
        -------
        P : np.ndarray
            2D array of the probability density |ψ|² evaluated on the x–z slice.
        extent : tuple of float
            (x_min, x_max, z_min, z_max) in units of a_μ.

        Notes
        -----
        A global cache (`_wavefunction_cache`) is used to avoid redundant
        recomputation for identical sets of (n, l, m). Other parameters like
        resolution or extent do not affect caching.
        """

    if wf in _wavefunction_cache:
        return _wavefunction_cache[wf]

    Xg, Zg, psi, a_mu = compute_psi_xz_slice(
        n=wf.n, l=wf.l, m=wf.m,
        Z=wf.Z,
        use_reduced_mass=wf.use_reduced_mass,
        M=wf.M,
        extent_a_mu=wf.extent_a_mu,
        grid_points=wf.grid_points,
        phi_value=wf.phi_value,
        phi_mode=wf.phi_mode
    )
    P = compute_probability_density(psi)

    extent = (
        float(np.min(Xg) / a_mu),
        float(np.max(Xg) / a_mu),
        float(np.min(Zg) / a_mu),
        float(np.max(Zg) / a_mu)
    )

    _wavefunction_cache[wf] = (P, extent)
    return P, extent


def create_plotly_colorscale(name: str):
    """
    Create a Plotly-compatible colorscale from a seaborn color palette.

    Parameters
    ----------
    name : str, optional
        Name of the seaborn color palette to convert.

    Returns
    -------
    list
        Plotly colorscale list of the form
        [[pos, "rgb(r,g,b)"], ...] with 100 samples.

    Notes
    -----
    The colorscale is linearly sampled from the seaborn palette and
    normalized to the interval [0, 1] for Plotly.
    """
    pal = sns.color_palette(name, as_cmap=False, n_colors=100)
    colorscale = []
    for i, rgb in enumerate(pal):
        colorscale.append([i / 99.0, f'rgb({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)})'])
    return colorscale


_last_valid_values = {
    'nmax': 30,
    'resolution': 600
}


def create_dash_app():
    """
        Create and initialize the Dash application for visualizing hydrogens probability densities on an x–z slice.

        Returns
        -------
        dash.Dash
            A fully configured Dash application ready to be run with
            `app.run(...)`.

        """
    print('Starting app...')
    app = Dash(__name__)
    colorscale = create_plotly_colorscale('rocket')

    # Initial values
    n0, l0, m0 = 1, 0, 0
    nmax0, resolution0 = _last_valid_values['nmax'], _last_valid_values['resolution']

    @app.callback(
        Output('wf-heatmap', 'figure'),
        Output('slider-n', 'max'),
        Output('slider-l', 'max'),
        Output('slider-m', 'min'),
        Output('slider-m', 'max'),
        # Sliders
        Input('slider-n', 'value'),
        Input('slider-l', 'value'),
        Input('slider-m', 'value'),
        # Preferences
        Input('preferences-resolution', 'value'),
        Input('preferences-nmax', 'value')
    )
    def update_heatmap(n, l, m, preferences_resolution, preferences_n_max) -> tuple[go.Figure, int, int, int, int]:
        """
            Update the |ψ|² heatmap and slider limits whenever a control value changes.

            Parameters
            ----------
            n : int
                Principal quantum number selected by the user.
            l : int
                Orbital angular momentum quantum number selected by the user.
            m : int
                Magnetic quantum number selected by the user.
            preferences_resolution : int
                Number of grid points used in each dimension of the computed slice.
            preferences_n_max : int
                Maximum allowed value for the n-slider.

            Returns
            -------
            fig : plotly.graph_objects.Figure
                Updated heatmap figure of the probability density.
            slider_n_max : int
                Updated maximum for the n-slider.
            slider_l_max : int
                Updated maximum for the l-slider (= n − 1).
            slider_m_min : int
                Updated minimum for the m-slider (= −l).
            slider_m_max : int
                Updated maximum for the m-slider (= +l).

            Notes
            -----
            The function enforces the physical constraints:
                0 ≤ l ≤ n−1
                −l ≤ m ≤ l

            The spatial extent scales as 2.5 * n² * a_μ, ensuring that larger
            states remain fully captured.
            """
        # If text fields are empty use last valid values
        # Must occur before checking validity of numbers to avoid numerical comps (less, greater, etc.) with None
        resolution = preferences_resolution if preferences_resolution is not None else _last_valid_values['resolution']
        nmax = preferences_n_max if preferences_n_max is not None else _last_valid_values['nmax']

        if resolution <= 0 or nmax <= 0:
            raise ValueError('Invalid preferences. Resolution and Max n must be non zero and positive.')

        # Slider limits dynamically
        l_max = n - 1
        if l > l_max:
            l = l_max
        m_max = l
        m = max(min(m, m_max), -m_max)

        # Compute Wave Function
        extent_a_mu = 2.5 * n ** 2
        wf = WaveFunction(n=n, l=l, m=m, extent_a_mu=extent_a_mu, grid_points=resolution)
        try:
            P, extent = compute_P_and_extent(wf)
        except MemoryError as e:
            # Specify error
            raise MemoryError(
                f'Encountered MemoryError. Likely caused by too much resolution. Try reducing resolution. Error: {e}')

        x_min, x_max, z_min, z_max = extent
        # Plot
        fig = go.Figure(
            data=[go.Heatmap(
                z=P,
                x=np.linspace(x_min, x_max, P.shape[1]),
                y=np.linspace(z_min, z_max, P.shape[0]),
                colorscale=colorscale,
                colorbar=dict(title=r'|ψ|² [m⁻³]')
            )]
        )
        # Note that layout might be changed in css
        fig.update_layout(
            title=f'Hydrogen Probability Density (n,l,m)=({n},{l},{m})',
            xaxis_title=r'x / a_μ',
            yaxis_title=r'z / a_μ',
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
        )

        # Update last valid values if everything worked
        _last_valid_values['resolution'] = resolution
        _last_valid_values['nmax'] = nmax
        return fig, nmax, l_max, -l, l

    # Initialize the plot once
    fig0, _, _, _, _ = update_heatmap(n0, l0, m0, resolution0, nmax0)

    # Dash layout
    # Note that layout might be changed in css
    app.layout = html.Div(className='container', children=[
        html.H1('Hydrogen Wavefunction Probability Density'),
        # ---------------- Github link ----------------
        html.Div(className='github-link', children=[
            html.A(
                href='https://github.com/ssebastianmag/hydrogen-wavefunctions',
                target='_blank',
                children=[
                    html.Img(src='/assets/github-logo.webp', className='github-icon')
                ]
            )
        ]),
        html.Div(className='global-grid', children=[
            # ---------------- Graph ----------------
            html.Div(className='graph-card', children=[
                dcc.Graph(id='wf-heatmap', figure=fig0)
            ]),

            # ---------------- Controls ----------------
            html.Div(className='controls-grid', children=[

                # ---------------- Sliders ----------------
                html.Div(className='sliders-grid', children=[
                    html.Div(className='slider-container slider-n', children=[
                        html.Label('n:'),
                        dcc.Slider(
                            id='slider-n',
                            min=1, max=nmax0, step=1, value=n0,
                            className='slider'
                        )
                    ]),
                    html.Div(className='slider-container slider-l', children=[
                        html.Label('l:'),
                        dcc.Slider(
                            id='slider-l',
                            min=0, max=n0 - 1, step=1, value=l0,
                            className='slider'
                        )
                    ]),
                    html.Div(className='slider-container slider-m', children=[
                        html.Label('m:'),
                        dcc.Slider(
                            id='slider-m',
                            min=-l0, max=l0, step=1, value=m0,
                            className='slider'
                        )
                    ])
                ]),

                # ---------------- Preferences ----------------
                html.Div(className='preferences-box', children=[
                    html.H3('Preferences'),
                    html.Div(className='preferences-grid', children=[
                        # Resolution
                        html.Label('Resolution (px):'),
                        dcc.Input(
                            id='preferences-resolution',
                            type='number',
                            value=resolution0,
                            className='preferences-input'
                        ),
                        # Nmax
                        html.Label('Maximum n:'),
                        dcc.Input(
                            id='preferences-nmax',
                            type='number',
                            value=nmax0,
                            className='preferences-input'
                        )
                    ])
                ]),
            ])
        ])

    ])
    print('App started')
    return app
