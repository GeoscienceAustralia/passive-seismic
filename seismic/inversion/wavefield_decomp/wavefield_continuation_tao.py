#!/usr/bin/env python
# coding: utf-8
"""Class encapsulating algorithm for 1D inversion using wavefield continuation.

    Based on reference:
    Kai Tao, Tianze Liu, Jieyuan Ning, Fenglin Niu, "Estimating sedimentary and crustal structure
    using wavefield continuation: theory, techniques and applications", *Geophysical Journal International*,
    Volume 197, Issue 1, April, 2014, Pages 443-457, https://doi.org/10.1093/gji/ggt515
"""

import numpy as np


class WfContinuationSuFluxComputer:
    """
    Implements computation of the upwards mean S-wave energy flux at the top of the mantle
    for an ensemble of events for one station.

    Class instance can be loaded with a dataset and then evaluated for arbitrary 1D earth models.

    Process:
    1. Load data from a station event dataset.  Copy of the data is buffered by this
        class in efficient format for energy flux calculation.
    2. Define 1D earth model and mantle half-space material properties (in external code).
    3. Call instance with models and receive energy flux results.
    """
    def __init__(self):
        pass

    def _streamdict_to_array(self, data, f_s, time_window, cut_window):
        """
        Convert dict of streams (indexed by event id) to numpy array in format required
        by function compute_su_energy(), including resampling to f_s and applying a cut
        window and sinc resampling.

        This conversion may be expensive and compute_su_energy() may need to be called
        many times, so preconverting the format to numpy array once only is important
        to overall performance.

        Any quality filtering needs to be performed prior to calling this function.

        Note: This function modifies in-place the traces in the values of data.
        """
        from seismic.receiver_fn.rf_util import sinc_resampling

        # Resample to f_s if any trace is not already as f_s
        for evid, stream in data.items():
            if np.any(np.array([tr.stats.sampling_rate != f_s for tr in stream])):
                # Resampling lowpass only, as per Tao (anti-aliasing)
                stream.filter('lowpass', freq=f_s / 2.0, corners=2, zerophase=True).interpolate(
                    f_s, method='lanczos', a=10)
            # end if
        # end for

        # Trim to time window
        for evid, stream in data.items():
            stream.trim(stream[0].stats.onset + time_window[0],
                        stream[0].stats.onset + time_window[1])
        # end for

        # Cut central data segment and resample back to original length using sinc interpolation.
        for stream in data.values():
            for tr in stream:
                times = tr.times() - (tr.stats.onset - tr.stats.starttime)
                tr_cut = tr.copy().trim(tr.stats.onset + cut_window[0], tr.stats.onset + cut_window[1])
                tr_cut.detrend('linear')
                tr_cut.taper(0.10)
                cut_times = tr_cut.times() - (tr_cut.stats.onset - tr_cut.stats.starttime)
                cut_data = tr_cut.data
                resampled_data = sinc_resampling(cut_times, cut_data, times)
                # Replace trace data with cut resampled data
                tr.data = resampled_data
            # end for
        # end for

        # Pull data arrays out into matrix format
        self.v0 = np.array([[st.select(component='R')[0].data.tolist(),
                        (-st.select(component='Z')[0].data).tolist()] for st in data.values()])

    # end func

    def __call__(self, v0, f_s, p, mantle_props, layer_props,
                      time_window=(-20, 50), flux_window=(-10, 20)):
        """Compute upgoing S-wave energy for a given set of seismic time series v0.

        :param v0: Numpy array of shape (N_events, 2, N_samples) containing the R- and
            Z-component traces for all events at sample rate f_s and covering duration
            of time_window.
        :param mantle_props: LayerProps representing mantle properties.
        :param layer_props: List of LayerProps.
        """
        # This is the callable operator that performs computations of energy flux
        dt = 1.0 / f_s
        npts = v0.shape[2]
        nevts = v0.shape[0]
        t = np.linspace(*time_window, npts)

        # Reshape to facilitate max_vz normalization using numpy broadcast rules.
        v0 = np.moveaxis(v0, 0, -1)

        # Normalize each event signal by the maximum z-component amplitude.
        # We perform this succinctly using numpy multidimensional broadcasting rules.
        max_vz = np.abs(v0[1, :, :]).max(axis=0)
        v0 = v0 / max_vz

        # Reshape back to original shape.
        v0 = np.moveaxis(v0, -1, 0)

        # Transform v0 to the spectral domain using real FFT
        fv0 = np.fft.rfft(v0, axis=-1)

        # Compute discrete frequencies
        w = 2 * np.pi * np.fft.rfftfreq(v0.shape[-1], dt)

        # Extend w to full spectral domain.
        w_full = np.hstack((w, -np.flipud(w[1:])))

        # To extend fv0, we need to flip left-right and take complex conjugate.
        fv0_full = np.dstack((fv0, np.fliplr(np.conj(fv0[:, :, 1:]))))

        # Compute mode matrices for mantle
        M_m, Minv_m, _ = WfContinuationSuFluxComputer._mode_matrices(mantle_props.Vp, mantle_props.Vs, mantle_props.rho, p)

        # Propagate from surface
        fvm = WfContinuationSuFluxComputer._propagate_layers(fv0_full, w_full, layer_props, p)
        fvm = np.matmul(Minv_m, fvm)

        num_pos_freq_terms = (fvm.shape[2] + 1) // 2
        # Velocities at top of mantle
        vm = np.fft.irfft(fvm[:, :, :num_pos_freq_terms], v0.shape[2], axis=2)

        # Compute coefficients of energy integral for upgoing S-wave
        qb_m = np.sqrt(1 / mantle_props.Vs ** 2 - p * p)
        Nsu = dt * mantle_props.rho * (mantle_props.Vs ** 2) * qb_m

        # Compute mask for the energy integral time window
        integral_mask = (t >= flux_window[0]) & (t <= flux_window[1])
        vm_windowed = vm[:, :, integral_mask]

        # Take the su component.
        su_windowed = vm_windowed[:, 3, :]

        # Integrate in time
        Esu_per_event = Nsu * np.sum(np.abs(su_windowed) ** 2, axis=1)

        # Compute mean over events
        Esu = np.mean(Esu_per_event)

        return Esu, Esu_per_event, vm

    # end func

    def _mode_matrices(Vp, Vs, rho, p):
        """Compute M, M_inv and Q for a single layer for a scalar or array of ray parameters p.

        :param Vp: P-wave body wave velocity (scalar, labeled α in Tao's paper)
        :type Vp:
        :param Vs: S-wave body wave velocity (scalar, labeled β in Tao's paper)
        :type Vs:
        :param rho: Bulk material density, ρ (scalar)
        :type rho:
        :param p: Scalar or array of ray parameters (one per event)
        :type p:
        """
        qa = np.sqrt((1 / Vp ** 2 - p * p).astype(np.complex))
        assert not np.any(np.isnan(qa)), qa
        qb = np.sqrt((1 / Vs ** 2 - p * p).astype(np.complex))
        assert not np.any(np.isnan(qb)), qb
        eta = 1 / Vs ** 2 - 2 * p * p
        mu = rho * Vs * Vs
        trp = 2 * mu * p * qa
        trs = 2 * mu * p * qb
        mu_eta = mu * eta
        # First compute without velocity factors for reduced operation count.
        M = np.array([
            [p, p, qb, qb],
            [qa, -qa, -p, p],
            [-trp, trp, -mu_eta, mu_eta],
            [-mu_eta, -mu_eta, trs, trs]
        ])
        # Then times by velocity factors
        Vfactors = np.diag([Vp, Vp, Vs, Vs])
        M = np.matmul(np.moveaxis(M, -1, 0), Vfactors)

        Q = np.dstack([np.expand_dims(np.array([-_1, _1, -_2, _2]), 1) for (_1, _2) in zip(qa, qb)])
        Q = np.moveaxis(Q, -1, 0)

        # First compute without velocity factors for reduced operation count.
        mu_p = mu * p
        Minv = (1.0 / rho) * np.array([
            [mu_p, mu_eta / 2 / qa, -p / 2 / qa, -0.5 * np.ones(p.shape)],
            [mu_p, -mu_eta / 2 / qa, p / 2 / qa, -0.5 * np.ones(p.shape)],
            [mu_eta / 2 / qb, -mu_p, -0.5 * np.ones(p.shape), p / 2 / qb],
            [mu_eta / 2 / qb, mu_p, 0.5 * np.ones(p.shape), p / 2 / qb]
        ])
        # Then times by velocity factors
        Vfactors_inv = np.diag([1 / Vp, 1 / Vp, 1 / Vs, 1 / Vs])
        Minv = np.matmul(Vfactors_inv, np.moveaxis(Minv, -1, 0))

        #     # DEBUG CHECK - verify M*Minv is close to identity
        #     for i in range(M.shape[0]):
        #         _M = M[i,:,:]
        #         _Minv = Minv[i,:,:]
        #         assert _M.shape[0] == _M.shape[1]
        #         assert np.allclose(np.matmul(_M, _Minv).flatten(), np.eye(_M.shape[0]).flatten()), i

        return (M, Minv, Q)
    # end func

    def _propagate_layers(fv0, w, layer_props, p):
        """
        layer_props is a list of LayerProps
        """
        fz = np.hstack((fv0, np.zeros_like(fv0)))
        for layer in layer_props:
            M, Minv, Q = WfContinuationSuFluxComputer._mode_matrices(layer.Vp, layer.Vs, layer.rho, p)
            fz = np.matmul(Minv, fz)
    #         phase_args = np.outer(Q - Q[1], w)
    #         phase_args = np.matmul(Q, np.expand_dims(w, 0))
            # Expanding dims on w here means that at each level of the stack, phase_args is np.outer(Q, w)
            phase_args = np.matmul(Q, np.expand_dims(np.expand_dims(w, 0), 0))
            assert np.allclose(np.outer(Q[0,:,:], w).flatten(), phase_args[0,:,:].flatten()), (Q, w)
            phase_factors = np.exp(1j*layer.H*phase_args)
            fz = phase_factors*fz  # point-wise multiplication
            fz = np.matmul(M, fz)
        # end for
        return fz
    # end func


# end class
