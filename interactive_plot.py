import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import colors
import seaborn as sns
import numpy as np
from hydrogen_wavefunction import compute_psi_xz_slice, compute_probability_density
from hwf_plots import WaveFunction, plot_hydrogen_wavefunction_xz

# Global cache dictionary for the session
_wavefunction_cache = {}


def create_hydrogen_plot(ax: matplotlib.axes.Axes, wf: WaveFunction, colormap='rocket'):
    key = (wf.n, wf.l, wf.m)
    if key in _wavefunction_cache:
        # Take from cache
        P, extent = _wavefunction_cache[key]
    else:
        # Compute
        Xg, Zg, psi, a_mu = compute_psi_xz_slice(
            n=wf.n, l=wf.l, m=wf.m, Z=wf.Z,
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
        _wavefunction_cache[key] = (P, extent)

    cmap = sns.color_palette(colormap, as_cmap=True)
    im = ax.imshow(P, extent=extent, origin='lower', aspect='equal', cmap=cmap)
    return im, P


def interactive_hydrogen_plot(slider_n_max=30):
    fig, ax = plt.subplots(figsize=(10, 9))
    plt.subplots_adjust(left=0.15, bottom=0.30)
    colormap_str = "rocket"
    # Background color -> darkest color in the colormap
    pal_100 = sns.color_palette(colormap_str, n_colors=100)
    cm_sorted = sorted(pal_100, key=lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])

    text_color = "#dfdfdf"
    tick_color = "#c4c4c4"

    fig.patch.set_facecolor(cm_sorted[0])
    for spine in ax.spines.values():
        spine.set_color(tick_color)

    ax.tick_params(axis="x", colors=tick_color)
    ax.tick_params(axis="y", colors=tick_color)

    # X_Y axis
    x_z_units = r"a_\mu"
    ax.set_xlabel(rf"$x / {x_z_units}$", fontsize=12, color=text_color)
    ax.set_ylabel(rf"$z / {x_z_units}$", fontsize=12, color=text_color)
    ax.xaxis.set_label_coords(x=0.5, y=-0.075)
    ax.yaxis.set_label_coords(x=-0.08, y=0.5)

    # Slider axes
    ax_n = plt.axes((0.15, 0.20, 0.70, 0.03))
    ax_l = plt.axes((0.15, 0.15, 0.70, 0.03))
    ax_m = plt.axes((0.15, 0.10, 0.70, 0.03))
    # Sliders
    slider_l_max = slider_n_max - 1
    slider_kwargs = {'valstep': 1, 'edgecolor': cm_sorted[-1]}
    slider_n = Slider(ax_n, 'n', 1, slider_n_max, valinit=1, **slider_kwargs)
    slider_l = Slider(ax_l, 'l', 0, slider_l_max, valinit=0, **slider_kwargs)
    slider_m = Slider(ax_m, 'm', -slider_l_max, slider_l_max, valinit=0, **slider_kwargs)
    sliders = [slider_n, slider_l, slider_m]
    for slider in sliders:
        slider.label.set_color(tick_color)
        slider.valtext.set_color(tick_color)

    # Initial plot
    wf_init = WaveFunction(n=1, l=0, m=0)
    im, P = create_hydrogen_plot(ax, wf_init)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r"Probability density $|\psi|^{2}$ [m$^{-3}$]", fontsize=12, color=text_color, labelpad=34)
    cbar.ax.tick_params(labelsize=12, colors=text_color)
    cbar.ax.set_frame_on(False)

    def update(val):
        n = int(slider_n.val)
        l = int(slider_l.val)
        m = int(slider_m.val)

        if l >= n:
            l = n - 1
            slider_l.set_val(l)
        if abs(m) > l:
            m = -l if m < 0 else l
            slider_m.set_val(m)
        lmax = n - 1
        slider_l.valmax = lmax
        slider_m.valmax = lmax
        slider_m.valmin = -lmax

        extent_a_mu = 2.5 * n ** 2

        wf = WaveFunction(n=n, l=l, m=m, extent_a_mu=extent_a_mu)
        _, P_new = create_hydrogen_plot(ax, wf, colormap=colormap_str)

        # Update image data and normalization
        im.set_data(P_new)
        finite = P_new[np.isfinite(P_new)]
        vmin, vmax = 0.0, float(np.percentile(finite, 99.9))
        im.set_norm(colors.Normalize(vmin=vmin, vmax=vmax))

        ax.set_title(f'Hydrogen Probability Density (n,l,m)=({n},{l},{m})')
        fig.canvas.draw_idle()

    slider_n.on_changed(update)
    slider_l.on_changed(update)
    slider_m.on_changed(update)
    plt.show()


if __name__ == '__main__':
    interactive_hydrogen_plot(slider_n_max=30)
