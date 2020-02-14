#!/usr/bin/env python
# coding: utf-8

import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

import numpy as np
from scipy import stats
import scipy.optimize as optimize

from tqdm.auto import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from seismic.inversion.wavefield_decomp.network_event_dataset import NetworkEventDataset
from seismic.inversion.wavefield_decomp.wavefield_continuation_tao import WfContinuationSuFluxComputer
from seismic.inversion.wavefield_decomp.model_properties import LayerProps
from seismic.stream_quality_filter import curate_stream3c
from seismic.receiver_fn.rf_util import compute_vertical_snr


logging.info("Loading input file...")
src_file = (r"/g/data/ha3/am7399/shared/OA_RF_analysis/" +
            r"OA_event_waveforms_for_rf_20170911T000036-20181128T230620_rev8.h5")
data_all = NetworkEventDataset(src_file, network='OA', station='BT23', location='0M')

# Time window of original data to use for processing. All traces must have at least this extent
# about the onset time.
TIME_WINDOW = (-20, 50)
# Narrower time window used for integration of energy flux
FLUX_WINDOW = (-10, 20)
# Cut window for selecting central wavelet
CUT_WINDOW = (-5, 30)

# -----------------------------------------------------------------------------
# Apply windowinf, filtering and QC to loaded dataset before passing to Tao's algorithm.
logging.info("Cleaning input data...")


def stream_snr_compute(stream):
    stream.taper(0.05)
    compute_vertical_snr(stream)
# end func


def amplitude_nominal(stream, max_amplitude):
    return ((np.max(np.abs(stream[0].data)) <= max_amplitude) and
            (np.max(np.abs(stream[1].data)) <= max_amplitude) and
            (np.max(np.abs(stream[2].data)) <= max_amplitude))
# end func


# Trim streams to time window
data_all.apply(lambda stream:
               stream.trim(stream[0].stats.onset + TIME_WINDOW[0], stream[0].stats.onset + TIME_WINDOW[1]))

# Apply curation to streams prior to rotation
data_all.curate(lambda _, evid, stream: curate_stream3c(evid, stream))

# Rotate to ZRT coordinates
data_all.apply(lambda stream: stream.rotate('NE->RT'))

# Detrend the traces
data_all.apply(lambda stream: stream.detrend('linear'))

# Run high pass filter to remove high amplitude, low freq noise, if present.
f_min = 0.05
data_all.apply(lambda stream: stream.filter('highpass', freq=f_min, corners=2, zerophase=True))

# Compute SNR of Z component to use as a quality metric
data_all.apply(stream_snr_compute)

# Filter by SNR
data_all.curate(lambda _1, _2, stream: stream[0].stats.snr_prior >= 3.0)

# It does not make sense to filter by similarity, since these are raw waveforms, not RFs,
# and the waveform will be dominated by the source waveform which differs for each event.

# Filter streams with incorrect number of traces
discard = []
for sta, ev_db in data_all.by_station():
    num_pts = np.array([tr.stats.npts for st in ev_db.values() for tr in st])
    expected_pts = stats.mode(num_pts)[0][0]
    for evid, stream in ev_db.items():
        if ((stream[0].stats.npts != expected_pts) or
            (stream[1].stats.npts != expected_pts) or
            (stream[2].stats.npts != expected_pts)):
            discard.append(sta, evid)
        # end if
    # end for
# end for
data_all.prune(discard)

# Filter streams with spuriously high amplitude
MAX_AMP = 10000
data_all.curate(lambda _1, _2, stream: amplitude_nominal(stream, MAX_AMP))


# -----------------------------------------------------------------------------
# Pass cleaned up data set for test station to flux computer class.
data_OA = data_all.station('BT23')
fs_processing = 10.0  # Hz
logging.info("Ingesting source data streams...")
flux_comp = WfContinuationSuFluxComputer(data_OA.values(), fs_processing, TIME_WINDOW, CUT_WINDOW)

# Define bulk properties of mantle (lowermost half-space)
mantle_props = LayerProps(vp=8.0, vs=4.5, rho=3.3, thickness=np.Infinity)

# Assumed crust property constants
Vp_c = 6.4
rho_c = 2.7

# Assumed sediment property constants
# Vp_s = 2.1
# rho_s = 1.97

crust_props = LayerProps(Vp_c, 3.7, rho_c, 35)

single_layer_model = [crust_props]


# -----------------------------------------------------------------------------
# Example 1: Computing energy for a single model proposition.

logging.info("Computing single point mean SU flux...")
energy, energy_per_event, wf_mantle = flux_comp(mantle_props, single_layer_model)
logging.info(energy)


# -----------------------------------------------------------------------------
# Example 2: Computing energy across a parametric space of models.

def plot_Esu_space(H, Vs, Esu, network, station, save=True, show=True):
    colmap = 'plasma'
    plt.figure(figsize=(16, 12))
    plt.contourf(Vs, H, Esu, levels=50, cmap=colmap)
    plt.colorbar()
    plt.contour(Vs, H, Esu, levels=10, colors='k', linewidths=1, antialiased=True)
    plt.xlabel('Crust $V_s$ (km/s)', fontsize=14)
    plt.ylabel('Crust $H$ (km)', fontsize=14)
    plt.tick_params(right=True, labelright=True, which='both')
    plt.tick_params(top=True, labeltop=True, which='both')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.minorticks_on()
    plt.xlim(np.min(Vs), np.max(Vs))
    plt.ylim(np.min(H), np.max(H))
    plt.grid(linestyle=':', color="#80808080")
    plt.title('{}.{} Crust properties'.format(network, station), fontsize=20, y=1.05)
    if save:
        plt.savefig('example_{}.{}_crust_props.png'.format(network, station), dpi=300)
    if show:
        plt.show()
    plt.close()
# end func


def job_caller(i, j, callable, mantle, earth_model, flux_window):
    energy, _, _ = callable(mantle, earth_model, flux_window=flux_window)
    return (i, j, energy)
# end func


H, Vs = np.meshgrid(np.linspace(25, 45, 51), np.linspace(Vp_c/2.1, Vp_c/1.5, 51))
Esu = np.zeros(H.shape)

logging.info("Computing 2D parametric space mean SU flux...")
results = []
for i, (H_arr, Vs_arr) in tqdm(enumerate(zip(H, Vs)), total=H.shape[0], desc='Crust loop'):
    results.extend(Parallel(n_jobs=-1)(delayed(job_caller)(i, j, flux_comp, mantle_props,
                                                           [LayerProps(Vp_c, _Vs, rho_c, _H)],
                                                           FLUX_WINDOW)
                                       for j, (_H, _Vs) in enumerate(zip(H_arr, Vs_arr))))
# end for

for i, j, energy in results:
    Esu[i, j] = energy
# end for

plot_Esu_space(H, Vs, Esu, 'OA', 'BT23')


# -----------------------------------------------------------------------------
# Example 3: Using a global energy minimization solver to find solution.

def objective_fn(model, callable, mantle, Vp, rho, flux_window):
    num_layers = len(model)//2
    earth_model = []
    for i in range(num_layers):
        earth_model.append(LayerProps(Vp[i], model[2*i + 1], rho[i], model[2*i]))
    # end for
    earth_model = np.array(earth_model)
    energy, _, _ = callable(mantle, earth_model, flux_window=flux_window)
    return energy
# end func


logging.info("Computing optimal crust properties by SU flux minimization...")

# Use single layer (crust only) model in this example.
Vp = [Vp_c]
rho = [rho_c]
k_min, k_max = (1.5, 2.1)
fixed_args = (flux_comp, mantle_props, Vp, rho, FLUX_WINDOW)
H_initial = 40.0
k_initial = np.mean((k_min, k_max))
model_initial = np.array([H_initial, Vp_c/k_initial])
H_min, H_max = (25.0, 45.0)
bounds = optimize.Bounds([H_min, Vp_c/k_max], [H_max, Vp_c/k_min])

# Find local minimum relative to initial guess.
soln = optimize.minimize(objective_fn, model_initial, fixed_args, bounds=bounds)
H_crust, Vs_crust = soln.x
logging.info('Success = {}, Iterations = {}, Function evaluations = {}'.format(soln.success, soln.nit, soln.nfev))
logging.info('Solution H_crust = {}, Vs_crust = {}, SU energy = {}'.format(H_crust, Vs_crust, soln.fun))


# -----------------------------------------------------------------------------
# Demonstrate syntactic usage of scipy global optimizers:
model_initial_poor = np.array([34, Vp_c/k_initial])

# - Basin hopping
logging.info('Trying basinhopping...')
soln_bh = optimize.basinhopping(objective_fn, model_initial_poor, T=0.3,
                                minimizer_kwargs={'args': fixed_args, 'bounds': bounds})
logging.info('Result:\n{}'.format(soln_bh))

# - Differential evolution
logging.info('Trying differential_evolution...')
soln_de = optimize.differential_evolution(objective_fn, bounds, fixed_args, workers=-1)
logging.info('Result:\n{}'.format(soln_de))

# - SHGO (VERY EXPENSIVE AND/OR not convergent)
# logging.info('Trying shgo...')
# soln_shgo = optimize.shgo(objective_fn, list(zip(bounds.lb, bounds.ub)), fixed_args,
#                           options={'f_min': 0, 'f_tol': 1.0e-3, 'maxev': 10000})
# logging.info('Result:\n{}'.format(soln_shgo))

# - Dual annealing
logging.info('Trying dual_annealing...')
soln_da = optimize.dual_annealing(objective_fn, list(zip(bounds.lb, bounds.ub)), fixed_args, x0=model_initial_poor,
                                  initial_temp=2000.0, maxfun=10000)
logging.info('Result:\n{}'.format(soln_da))
