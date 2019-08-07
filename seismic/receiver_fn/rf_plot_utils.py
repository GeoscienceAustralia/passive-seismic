#!/usr/bin/env python
"""Utility plotting functions for consistent and convenient plotting of RFs.
"""

import logging
import time

import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

# import rf
# import rf.imaging

logging.basicConfig()


def plot_rf_stack(rf_stream, time_window=(-10.0, 25.0), trace_height=0.2, stack_height=0.8, save_file=None):
    """Wrapper function of rf.RFStream.plot_rf() to help do RF plotting with consistent formatting and layout.

    :param rf_stream: [description]
    :type rf_stream: [type]
    :param time_window: [description], defaults to (-10.0, 25.0)
    :type time_window: tuple, optional
    :param trace_height: [description], defaults to 0.2
    :type trace_height: float, optional
    :param stack_height: [description], defaults to 0.8
    :type stack_height: float, optional
    :param save_file: [description], defaults to None
    :type save_file: [type], optional
    """
    _ = rf_stream.plot_rf(fillcolors=('#000000', '#a0a0a0'), trim=time_window, trace_height=trace_height,
                          stack_height=stack_height, fname=save_file)


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


def plot_hk_stack(k_grid, h_grid, hk_stack, title=None, save_file=None, show=True, num=None, clip_negative=True):
    """Plot H-k stack using data generated by function rf_stacking.computed_weighted_stack().

    :param k_grid: [description]
    :type k_grid: [type]
    :param h_grid: [description]
    :type h_grid: [type]
    :param hk_stack: [description]
    :type hk_stack: [type]
    :param title: [description], defaults to None
    :type title: [type], optional
    :param save_file: [description], defaults to None
    :type save_file: [type], optional
    :param show: [description], defaults to True
    :type show: bool, optional
    :param num: [description], defaults to None
    :type num: [type], optional
    :param clip_negative: [description], defaults to True
    :type clip_negative: bool, optional
    """
    # Call rf_stacking.computed_weighted_stack() first to combine weighted components before calling this function.
    # For best practices, use a perceptually linear color map.
    colmap = 'plasma'
    plt.figure(figsize=(16, 12))
    if clip_negative:
        hk_stack[hk_stack < 0] = 0
    plt.contourf(k_grid, h_grid, hk_stack, levels=50, cmap=colmap)
    cb = plt.colorbar()
    cb.mappable.set_clim(0, np.max(hk_stack[:]))
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

    if show:
        plt.show()
    else:
        plt.close()