#!/usr/bin/env python
"""Utility plotting functions for consistent and convenient plotting of RFs.
"""

import logging
import time
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# pylint: disable=invalid-name, logging-format-interpolation

logging.basicConfig()


def plot_rf_stack(rf_stream, time_window=(-10.0, 25.0), trace_height=0.2, stack_height=0.8, save_file=None, **kwargs):
    """Wrapper function of rf.RFStream.plot_rf() to help do RF plotting with consistent formatting and layout.

    :param rf_stream: RFStream to plot
    :type rf_stream: rf.RFStream
    :param time_window: Time window to plot, defaults to (-10.0, 25.0)
    :type time_window: tuple, optional
    :param trace_height: Height of a single trace (reduce to cram RFs closer together), defaults to 0.2
    :type trace_height: float, optional
    :param stack_height: Height of mean (stacked) RF at top of plot, defaults to 0.8
    :type stack_height: float, optional
    :param save_file: File to save resulting image into, defaults to None
    :type save_file: str to valid file path, optional
    """
    fig = rf_stream.plot_rf(fillcolors=('#000000', '#a0a0a0'), trim=time_window, trace_height=trace_height,
                            stack_height=stack_height, fname=save_file, show_vlines=True, **kwargs)
    return fig


def plot_station_rf_overlays(db_station, title=None, time_range=None):
    """Plot translucent overlaid RF traces for all traces in each channel, and overplot
    the mean signal of all the traces per channel.

    :param db_station: Dictionary with list of traces per channel for a given station.
    :type db_station: dict({str, list(RFTrace)})
    :param title: Plot title, defaults to None
    :type title: str, optional
    :return: Mean trace signal per channel
    :rtype: list(np.array)
    """
    num_channels = 0
    for ch, traces in db_station.items():
        if traces:
            num_channels += 1

    plt.figure(figsize=(16, 8*num_channels))
    colors = ["#8080a040", "#80a08040", "#a0808040"]
    min_x = 1e+20
    max_x = -1e20

    signal_means = []
    for i, (ch, traces) in enumerate(db_station.items()):
        if not traces:
            continue
        col = colors[i]
        plt.subplot(num_channels, 1, i + 1)
        sta = traces[0].stats.station
        for j, tr in enumerate(traces):
            lead_time = tr.stats.onset - tr.stats.starttime
            times = tr.times()
            plt.plot(times - lead_time, tr.data, '--', color=col, linewidth=2)
            mask = (~np.isnan(tr.data) & ~np.isinf(tr.data))
            if j == 0:
                data_mean = np.zeros_like(tr.data)
                data_mean[mask] = tr.data[mask]
                counts = mask.astype(np.float)
            else:
                data_mean[mask] += tr.data[mask]
                counts += mask.astype(np.float)
            # end if
        # end for
        data_mean = data_mean/counts
        data_mean[(counts == 0)] = np.nan
        signal_means.append(data_mean)
        plt.plot(tr.times() - lead_time, data_mean, color="#202020", linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (normalized)')
        plt.grid(linestyle=':', color="#80808020")
        title_text = '.'.join([sta, ch])
        if title is not None:
            title_text += ' ' + title
        plt.title(title_text, fontsize=14)
        x_lims = plt.xlim()
        min_x = min(min_x, x_lims[0])
        max_x = max(max_x, x_lims[1])
    # end for
    if time_range is not None:
        min_x = time_range[0]
        max_x = time_range[1]

    for i in range(num_channels):
        subfig = plt.subplot(num_channels, 1, i + 1)
        subfig.set_xlim((min_x, max_x))
    # end for

    return signal_means


def plot_hk_stack(k_grid, h_grid, hk_stack, title=None, save_file=None, num=None, clip_negative=True):
    """Plot H-k stack using data generated by function rf_stacking.computed_weighted_stack().

    This function is subject to further validation - documentation deferred.
    """
    # Call rf_stacking.computed_weighted_stack() first to combine weighted components before calling this function.
    # For best practices, use a perceptually linear color map.
    colmap = 'plasma'
    fig = plt.figure(figsize=(16, 12))
    if clip_negative:
        hk_stack = hk_stack.copy()
        hk_stack[hk_stack < 0] = 0
    # end if
    plt.contourf(k_grid, h_grid, hk_stack, levels=50, cmap=colmap)
    cb = plt.colorbar()
    cb.mappable.set_clim(0, np.nanmax(hk_stack[:]))
    cb.ax.set_ylabel('Stack sum')
    plt.contour(k_grid, h_grid, hk_stack, levels=10, colors='k', linewidths=1)
    plt.xlabel(r'$\kappa = \frac{V_p}{V_s}$ (ratio)', fontsize=14)
    plt.ylabel('H = Moho depth (km)', fontsize=14)
    if title is not None:
        plt.title(title, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tick_params(right=True, labelright=True)
    plt.yticks(fontsize=14)

    if num is not None:
        xl = plt.xlim()
        yl = plt.ylim()
        txt_x = xl[0] + 0.85*(xl[1] - xl[0])
        txt_y = yl[0] + 0.95*(yl[1] - yl[0])
        plt.text(txt_x, txt_y, "N = {}".format(num), color="#ffffff", fontsize=16, fontweight='bold')
    # end if

    if save_file is not None:
        tries = 10
        while tries > 0:
            try:
                tries -= 1
                plt.savefig(save_file, dpi=300)
                break
            except PermissionError:
                time.sleep(1)
                if tries == 0:
                    print("WARNING: Failed to save file {} due to permissions!".format(save_file))
                    break
            # end try
        # end while
    # end if

    return fig


def plot_rf_wheel(rf_stream, max_time=15.0, deg_per_unit_amplitude=45.0, plt_col='C0', title='',
                  figsize=(10, 10), cluster=True, cluster_col='#ff4000', layout=None, fontscaling=1.0):
    """Plot receiver functions around a polar plot with source direction used to position radial RF plot.

    :param rf_stream: Collection of RFs to plot. If passed as a list, then each stream in the list
        will be plotted on separate polar axes.
    :type rf_stream: rf.RFStream or list(rf.RFStream)
    :param max_time: maximum time relative to onset, defaults to 25.0
    :type max_time: float, optional
    :param deg_per_unit_amplitude: Azimuthal scaling factor for RF amplitude, defaults to 20
    :type deg_per_unit_amplitude: float, optional
    :param plt_col: Plot color for line and positive signal areas, defaults to 'C0'
    :type plt_col: str, optional
    :param title: Title for the overall plot, defaults to ''
    :type title: str, optional
    :param figsize: Size of figure area, defaults to (12, 12)
    :type figsize: tuple, optional
    :param cluster: Whether to add overlaid mean RF where there are many RFs close together.
    :type cluster: bool
    :param cluster_col: Color of clustered stacked overlay plots.
    :type cluster_col: matplotlib color specification
    :param layout: Arrangement of polar plots in grid. If None, then arranged in a column.
    :type layout: tuple(int, int)
    :return: Figure object
    :rtype: matplotlib.figure.Figure
    """
    if not isinstance(rf_stream, list):
        rf_stream = [rf_stream]
    if layout is None:
        layout = (len(rf_stream), 1)

    figsize = (figsize[0]*layout[1], figsize[1]*layout[0])
    fig = plt.figure(figsize=figsize)

    for n, stream in enumerate(rf_stream):
        ax = plt.subplot(*tuple(list(layout) + [n + 1]), projection="polar")
        # Orient with north
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(75)

        inner_radius = 0.4*max_time  # time units (e.g. sec)
        stream = stream.copy().trim2(0, max_time, reftime='onset')
        for i, tr in enumerate(stream):
            t = tr.times()
            rf_amp = tr.data
            back_azi = np.deg2rad(tr.stats.back_azimuth)
            azi_amp = back_azi - np.deg2rad(deg_per_unit_amplitude*rf_amp/
                                            np.linspace(1, (np.max(t) - np.min(t))/inner_radius, len(t)))
            plt.plot(azi_amp, t, color=plt_col, zorder=i+1)
            ax.fill_betweenx(t, azi_amp, back_azi, where=((azi_amp - back_azi) < 0), lw=0., facecolor=plt_col,
                             alpha=0.7, zorder=i+1)
            ax.fill_betweenx(t, azi_amp, back_azi, where=((azi_amp - back_azi) >= 0), lw=0., facecolor='#a0a0a080',
                             zorder=i+1)
        # end for

        ax.set_rorigin(-inner_radius)
        ax.set_rlim(0, max_time)
        ax.tick_params(labelsize=14*fontscaling)

        stream_meta = stream[0].stats
        target_id = '.'.join([stream_meta.network, stream_meta.station, stream_meta.channel])
        ax.text(0.5, 0.5, target_id, fontsize=14*fontscaling, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)

        if cluster:
            try:
                from sklearn.cluster import DBSCAN

                back_azis = np.array([tr.stats.back_azimuth for tr in stream])
                clustering = DBSCAN(eps=1.0).fit_predict(back_azis.reshape(-1, 1))
                cluster_data = defaultdict(list)
                for i, cl in enumerate(clustering):
                    if cl == -1:
                        continue
                    cluster_data[cl].append(stream[i])
                # end for

                zplus = i + 1
                for i, cl in enumerate(cluster_data.values()):
                    # Have to assume same time samples for each RFTrace.
                    t = cl[0].times()
                    mean_azi = np.deg2rad(np.mean([tr.stats.back_azimuth for tr in cl]))
                    mean_amp = np.mean([tr.data for tr in cl], axis=0)

                    azi_amp = mean_azi - np.deg2rad(deg_per_unit_amplitude*mean_amp/
                                                    np.linspace(1, (np.max(t) - np.min(t))/inner_radius, len(t)))
                    plt.plot(azi_amp, t, color=cluster_col, zorder=i+zplus)
                    ax.fill_betweenx(t, azi_amp, mean_azi, where=((azi_amp - mean_azi) < 0), lw=0.,
                                     facecolor=cluster_col, zorder=i+zplus)
                    ax.fill_betweenx(t, azi_amp, mean_azi, where=((azi_amp - mean_azi) >= 0), lw=0.,
                                     facecolor='#a0a0a080', zorder=i+zplus)
                # end for
            except Exception as e:
                logging.error("Clustering RFs failed with error: {}".format(str(e)))
            # end try
        # end if

        if title:
            fig.suptitle(title, fontsize=20*fontscaling)
        # end if

    # end for

    return fig
