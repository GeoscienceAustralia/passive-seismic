#!/usr/bin/env python

import os

import numpy as np
import click
import rf

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy import interpolate
from scipy.signal.windows import tukey
from scipy.signal import firwin, tf2sos, sosfiltfilt
from obspy.geodetics.base import gps2dist_azimuth

import seismic.receiver_fn.rf_util as rf_util
from seismic.ASDFdatabase import FederatedASDFDataSet
from seismic.receiver_fn.plot_ccp import run
from seismic.units_utils import KM_PER_DEG


# pylint: disable=invalid-name

LEAD_INOUT_DIST_KM = 25.0


def run_batch(transect_file, rf_waveform_file, fed_db_file, amplitude_filter=False, similarity_filter=False,
              stack_scale=0.4, width=30.0, spacing=2.0, max_depth=200.0,
              channel='R', output_folder='', colormap='seismic', annotators=None):
    """Run CCP generation in batch mode along a series of transects.

    :param transect_file: File containing specification of network and station locations of ends of transects
    :type transect_file: str or Path
    :param rf_waveform_file: HDF5 file of QA'd receiver functions for the network matching the transect file
    :type rf_waveform_file: str or Path
    :param fed_db_file: Name of file with which to initialize FederatedASDFDataBase
    :type fed_db_file: str or Path
    :param amplitude_filter: Whether to use amplitude-based filtering of waveforms beform plotting.
    :type amplitude_filter: bool
    :param similarity_filter: Whether to use RF waveform similarity filtering of waveforms beform plotting.
    :type similarity_filter: bool
    :param stack_scale: Max value to represent on color scale of CCP plot
    :type stack_scale: float
    :param width: Width of transect (km)
    :type width: float
    :param spacing: Discretization size (km) for RF ray sampling
    :type spacing: float
    :param max_depth: Maximum depth of slice below the transect line (km)
    :type max_depth: float
    :param channel: Channel component ID to source for the RF amplitude
    :type channel: str length 1
    :return: None
    """

    print("Reading HDF5 file...")
    rf_stream = rf.read_rf(rf_waveform_file, 'H5').select(component=channel)

    rf_type = rf_stream[0].stats.rotation
    if amplitude_filter:
        # Label and filter quality
        rf_util.label_rf_quality_simple_amplitude(rf_type, rf_stream)
        rf_stream = rf.RFStream([tr for tr in rf_stream if tr.stats.predicted_quality == 'a'])
    # end if

    # For similarity filtering, similarity filtering must applied to one station at a time.
    if similarity_filter:
        data_dict = rf_util.rf_to_dict(rf_stream)
        rf_stream = rf.RFStream()
        for _sta, ch_dict in data_dict:
            for _cha, ch_traces in ch_dict.items():
                if len(ch_traces) >= 3:
                    # Use short time window that cuts off by 10 sec, since we're only interested in Ps phase here.
                    filtered_traces = rf_util.filter_crosscorr_coeff(rf.RFStream(ch_traces), time_window=(-2, 10),
                                                                     apply_moveout=True)
                    rf_stream += filtered_traces
                else:
                    rf_stream += rf.RFStream(ch_traces)
                # end if
            # end for
        # end for
    # end if

    spectral_filter = {'type': 'highpass', 'freq': 0.2, 'corners': 1, 'zerophase': True}
    if spectral_filter is not None:
        rf_stream.filter(**spectral_filter)
    # end if

    db = FederatedASDFDataSet.FederatedASDFDataSet(fed_db_file)
    sta_coords = db.unique_coordinates

    if output_folder and not os.path.isdir(output_folder):
        assert not os.path.isfile(output_folder)
        os.makedirs(output_folder, exist_ok=True)
    # end if

    with open(transect_file, 'r') as f:
        net = f.readline().strip()
        for transect in f.readlines():

            if not transect.strip():
                continue

            sta_start, sta_end = transect.split(',')
            sta_start = sta_start.strip()
            sta_end = sta_end.strip()
            start = '.'.join([net, sta_start])
            end = '.'.join([net, sta_end])
            start = np.array(sta_coords[start])
            end = np.array(sta_coords[end])

            # Offset ends slightly to make sure we don't lose end stations due to truncation error.
            # Note: for simplicity this treats lat/lon like cartesian coords, but this is approximate
            # and will break down near poles, for long transects, or if transect crosses the antimeridian.
            dirn = (end - start)
            dirn = dirn/np.linalg.norm(dirn)
            start -= LEAD_INOUT_DIST_KM*dirn/KM_PER_DEG
            end += LEAD_INOUT_DIST_KM*dirn/KM_PER_DEG
            start_latlon = (start[1], start[0])
            end_latlon = (end[1], end[0])

            title = 'Network {} CCP R-stacking (profile {}-{})'.format(net, sta_start, sta_end)
            hf_main, hf_map, metadata = run(rf_stream, start_latlon, end_latlon, width, spacing, max_depth, channel,
                                            stacked_scale=stack_scale, title=title, colormap=colormap,
                                            background_model='ak135_60')

            metadata['transect_start'] = start
            metadata['transect_end'] = end
            metadata['transect_dirn'] = dirn
            if annotators is not None:
                for ant in annotators:
                    ant(hf_main, metadata)
                # end for
            # end if

            outfile_base = '{}-ZRT-R_CCP_stack_{}-{}_{}km_spacing'.format(net, sta_start, sta_end, spacing)
            outfile = outfile_base + '.pdf'
            outfile_map = outfile_base + '_MAP.pdf'

            outfile = os.path.join(output_folder, outfile)
            outfile_map = os.path.join(output_folder, outfile_map)

            if hf_main is not None:
                hf_main.savefig(outfile, dpi=300)
                plt.close(hf_main)
            # endif

            if hf_map is not None:
                hf_map.savefig(outfile_map, dpi=300)
                plt.close(hf_map)
            # endif

        # end for
    # end with
# end func


def moho_annotator(hf, metadata):
    """
    Custom plot annotator to add markers to the main figure showing locations of Moho
    estimates from other sources.

    :param hf: Figure containing main CCP plot
    :type hf: matplotlib.pyplot.Figure
    :param metadata: Dictionary from CCP plotting code containing metadata of the transect and station data
    :type metadata: dict
    """
    if hf is None or metadata is None:
        return
    # end if

    filename = "post_analysis/OA_Hk+RFinversion_moho_2019-12-12.csv"
    data = pd.read_csv(filename, usecols=['Site', 'HKM_Depth', 'Inv_Dp'], skipinitialspace=True, index_col='Site',
                       dtype={'Site': str, 'HKM_Depth': np.float64, 'Inv_Dp': np.float64,})

    x = []
    y1 = []
    y2 = []
    for stn, md in metadata.items():
        if 'transect_' in stn:
            continue  # Skip non-station metadata
        if md is None:
            continue
        x.append(md['sta_offset'])
        y1.append(data.loc[stn]['HKM_Depth'])
        y2.append(data.loc[stn]['Inv_Dp'])
    # end for
    x = np.array(x)
    y1 = np.array(y1)
    y2 = np.array(y2)
    plt.figure(hf.number)
    plt.plot(x, y1, 'o', markerfacecolor="C2", alpha=0.8, markersize=10,
             markeredgecolor="#101010", markeredgewidth=2, aa=True)
    plt.plot(x, y2, 'v', markerfacecolor="#ffd700", alpha=0.8, markersize=10,
             markeredgecolor="#101010", markeredgewidth=2, aa=True)
    plt.legend(['H-k depth', 'RF Inv. depth'], loc='lower right')

# end func


def gravity_subplot(hf, metadata, grav_map):
    """
    Custom plot modifier to add a gravity subplot along the CCP transect line.

    :param hf: Figure containing main CCP plot
    :type hf: matplotlib.pyplot.Figure
    :param metadata: Dictionary from CCP plotting code containing metadata of the transect and station data
    :type metadata: dict
    :param grav_map: Interpolator callable that takes a iterable of 2D coordinates and interpolates a gravity
        value at each, based on external data source.
    :type grav_map: Instance of 2D interpolation class from scipy.interpolator
    """
    if hf is None or metadata is None:
        return
    # end if

    def tukey_lowpass_filter(x, filter_len_km):
        filter_len = int(np.ceil(filter_len_km * RESAMPLES_PER_KM))
        if (filter_len % 2) == 0:
            filter_len += 1
        # end if
        kern = firwin(filter_len, 1.0 / filter_len_km, window=('tukey', 0.3), fs=RESAMPLES_PER_KM)
        sos_kern = tf2sos(kern, 1)
        sig_filt = sosfiltfilt(sos_kern, x)
        return sig_filt
    # end func

    def tukey_rolling_mean(x, filter_len_km):
        filter_len = int(np.ceil(filter_len_km * RESAMPLES_PER_KM))
        if (filter_len % 2) == 0:
            filter_len += 1
        # end if
        kern = tukey(filter_len, 0.3)
        kern = kern/np.sum(kern)
        sos_kern = tf2sos(kern, 1)
        sig_filt = sosfiltfilt(sos_kern, x)
        return sig_filt
    # end func

    RESAMPLES_PER_KM = 2.0

    # Move the bottom of the main axes bounding box up to make space to gravity plot beneath
    pos = hf.axes[0].get_position()
    pos.y0 += 0.2
    hf.axes[0].set_position(pos)
    # Also move colorbar
    pos_cb = hf.axes[1].get_position()
    pos_cb.y0 += 0.2
    hf.axes[1].set_position(pos_cb)

    # Add gravity plot. Ignores curvature of earth in transect projection to distance
    # from starting point, just uses Euclidean geometry.
    start = metadata['transect_start']
    end = metadata['transect_end']
    dirn = metadata['transect_dirn']
    grid_spec = gridspec.GridSpec(ncols=1, nrows=2, figure=hf, height_ratios=[4, 1])
    ax_grav = hf.add_subplot(grid_spec[1])
    pos_grav = ax_grav.get_position()
    pos_grav.x0 = pos.x0
    pos_grav.x1 = pos.x1
    ax_grav.set_position(pos_grav)
    plt.sca(ax_grav)
    plt.title("Gravity survey", fontsize=8, y=0.80)
    transect_len_km = gps2dist_azimuth(start[1], start[0], end[1], end[0])[0]/1000.0
    num_samples = int(np.ceil(transect_len_km*RESAMPLES_PER_KM))
    grav_pos = np.linspace(start, end, num_samples)
    grav_vals = grav_map(grav_pos)
    grav_dist = np.dot((grav_pos - start), dirn)*KM_PER_DEG
    grav_vals_lp_filt = tukey_lowpass_filter(grav_vals, 15.0)
    # grav_vals_tk_filt = tukey_rolling_mean(grav_vals, 15.0)
    grav_vals_lp_filt_long = tukey_lowpass_filter(grav_vals, 70.0)
    # grav_vals_tk_filt_long = tukey_rolling_mean(grav_vals, 70.0)
    plt.plot(grav_dist, grav_vals, label='Original Bouger')
    plt.plot(grav_dist, grav_vals_lp_filt, '--', alpha=0.6, label='Designed Tukey filter (15 km)')
    # plt.plot(grav_dist, grav_vals_tk_filt, '-.', alpha=0.6, label='Tukey rolling window (15 km)')
    plt.plot(grav_dist, grav_vals_lp_filt_long, '--', alpha=0.6, label='Designed Tukey filter (70 km)')
    # plt.plot(grav_dist, grav_vals_tk_filt_long, '-.', alpha=0.6, label='Tukey rolling window (70 km)')
    plt.grid("#80808080", linestyle=':')
    plt.legend(loc=3, fontsize=8)
    tickstep_x = 50.0
    xlim = hf.axes[0].get_xlim()
    plt.xticks(np.arange(0.0, xlim[1], tickstep_x), fontsize=12)
    plt.xlim(xlim)
    plt.ylabel('Gravity (mGal)')
# end func


@click.command()
@click.option('--rf-file', type=click.Path(exists=True, dir_okay=False), required=True,
              help='HDF5 file containing receiver functions')
@click.option('--waveform-database', type=click.Path(exists=True, dir_okay=False), required=True,
              help='Location of waveform source database used to generate FederatedASDFDataSet. '
                   'Provides station location metadata. e.g. "/g/data/ha3/Passive/SHARED_DATA/Index/asdf_files.txt".')
@click.option('--stack-scale', type=float, default=0.4, show_default=True,
              help='Max value to represent on color scale of CCP plot')
@click.option('--width', type=float, default=40.0, show_default=True,
              help='Width of transect (km)')
@click.option('--depth', type=float, default=200.0, show_default=True,
              help='Depth of slice below the transect line (km)')
@click.option('--spacing', type=float, default=2.0, show_default=True,
              help='Discretization size (km) of grid beneath transect line')
@click.option('--channel', type=click.Choice(['R', 'T', 'Q']), default='R', show_default=True,
              help='Channel to use for stacking, e.g. R')
@click.option('--colormap', type=str, default='seismic', show_default=True,
              help='Colormap to use. Must be recognized by matplotlib. Suggest seismic, coolwarm or jet.')
@click.option('--apply-amplitude-filter', is_flag=True, default=False, show_default=True,
              help='Apply RF amplitude filtering to the RFs.')
@click.option('--apply-similarity-filter', is_flag=True, default=False, show_default=True,
              help='Apply RF similarity filtering to the RFs.')
@click.argument('transect-file', type=click.Path(exists=True, dir_okay=False), required=True)
@click.argument('output-folder', type=click.Path(dir_okay=True, file_okay=False), required=False, default='')
def main(transect_file, output_folder, rf_file, waveform_database, stack_scale, width, depth, spacing, channel,
         colormap, apply_amplitude_filter, apply_similarity_filter):
    """
    Batch mode generation of CCP stacks.

    transect_file: File containing network code (first line) followed by transect line start/end points in
    the form of pairs on comma-separated station codes, two per line

    output_folder: Folder in which to place output files.
    """
    assert len(channel)  == 1, "Batch stack only on one channel at a time"

    # Custom plot modifiers. Leave commented for now until refactoring in ticket PST-479
    print("Loading gravity data...")
    grav = np.load('post_analysis/GravityGrid.xyz.npy')
    # Cut gravity map to region of interest
    OA_region_mask = (grav[:, 1] >= -23) & (grav[:, 1] <= -17) & (grav[:, 0] >= 132) & (grav[:, 0] <= 142)
    grav = grav[OA_region_mask, :]
    print("Creating interpolator...")
    grav_map = interpolate.CloughTocher2DInterpolator(grav[:, 0:2], grav[:, 2])
    # annotators = [moho_annotator, lambda hf, md: gravity_subplot(hf, md, grav_map)]
    annotators = [lambda hf, md: gravity_subplot(hf, md, grav_map)]
    # annotators = None
    print("Producing plot...")
    run_batch(transect_file, rf_file, waveform_database, stack_scale=stack_scale, width=width, spacing=spacing,
              max_depth=depth, channel=channel, output_folder=output_folder, colormap=colormap,
              amplitude_filter=apply_amplitude_filter, similarity_filter=apply_similarity_filter,
              annotators=annotators)
# end main


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
# end if
