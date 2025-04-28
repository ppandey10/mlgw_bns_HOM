from src.model import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Sans Serif']})
rc('text', usetex=True)
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from itertools import cycle

from src.model import *

from src.higher_order_modes import (
    BarePostNewtonianModeGenerator,
    TEOBResumSModeGenerator,
    Mode,
    ModeGenerator,
    ModeGeneratorFactory,
    spherical_harmonic_spin_2,
    spherical_harmonic_spin_2_conjugate, 
    teob_mode_generator_factory,
    mode_to_k
)

from src.taylorf2 import phase_5h_post_newtonian_tidal
from src.pn_modes import (
    Mode,
   _post_newtonian_amplitudes_by_mode,
   _post_newtonian_phases_by_mode 
)
from src.neural_network import TimeshiftsGPR

modes_model = ModesModel(
    filename="fast_train_2k",
    modes=[Mode(2,2), Mode(3,3), Mode(2,1), Mode(4,4)]
)
modes_model.load()

model_22 = modes_model.models[Mode(2,2)]
model_33 = modes_model.models[Mode(3,3)]
model_21 = modes_model.models[Mode(2,1)]
model_44 = modes_model.models[Mode(4,4)]


models = {
    "model_22": model_22,
    "model_21": model_21,
    "model_33": model_33,
    "model_44": model_44
}

f_hz = np.arange(20, 2048, 0.0007153125)

time_shifts_predictor = TimeshiftsGPR().load_model(
    filename="ts_model_HOM_comp_pool.pkl"
)

def plot_lm():
    """
    Plot the amplitude & phase residuals between MLGW and EOB for single mode.
    """

    lam = np.linspace(1.0, 3.0, num=10)
    cmap = plt.cm.Spectral
    norm = mcolors.Normalize(vmin=np.min(lam), vmax=np.max(lam))
    sm = ScalarMappable(norm=norm, cmap=cmap)

    plt.figure(figsize=(6, 3))

    ax = plt.axes()

    for i in range(len(lam[:])):

        p_set = ParameterSet(np.array([[lam[i], 400, 800, 0, 0]]))
        dataset = Dataset(initial_frequency_hz=20., srate_hz=4096.)
        wp_list = p_set.waveform_parameters(dataset)

        time_shifts = time_shifts_predictor.predict([wp_list[0].array])

        log_teob_pn_amp = np.log(np.abs(amp_pred) / np.abs(amp_teob))

        phase_res = (
            (model_33.predict_waveforms_bulk(p_set, model_33.nn).phases 
            - model_33.dataset.generate_waveforms_from_params(p_set, model_33.downsampling_indices).phases)
        )[0]

        freq = model_33.dataset.frequencies_hz[model_33.downsampling_indices.phase_indices]

        phase_diff_w_ts = (
            phase_res 
            + 2 * np.pi * time_shifts * (freq - freq[0])
            - phase_res[0]
        )
        ax.set_xscale("log")
        ax.plot(freq, phase_diff_w_ts, color=sm.to_rgba(lam[i]))
        # ax.plot(f_spa, log_teob_pn_amp, color=sm.to_rgba(lam[i]))
        ax.set_title(r'$(\ell=3, m=3)$')
        ax.grid(True)
        ax.set_ylabel(r'$\phi_{lm}^{\mathrm{EOB}} - \phi_{lm}^{\mathrm{PN}} - 2 \pi f \Delta{t_{\mathrm{GPR}}}$')
        # ax.set_ylabel(r'$\log(A_{lm}^{\mathrm{EOB}}/A_{lm}^{\mathrm{PN}})$')
        ax.set_xlabel(r'$Mf$')
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label(r'$q$')
    plt.savefig('mlgw_vs_eob_lm_phi.png', bbox_inches='tight')
    print("done!")

def plotting_mlgw():
    """
    Plot the amplitude & phase residuals between MLGW and EOB for (2,1), (2,2), (3,3) and (4,4) modes.
    """

    modes_list = cycle(['(\ell=2, m=1)', '(\ell=2, m=2)', '(\ell=3, m=3)', '(\ell=4, m=4)'])
    modes = cycle([Mode(2,1), Mode(2,2), Mode(3,3), Mode(4,4)])

    lam = np.linspace(1.0, 3.0, num=10)
    cmap = plt.cm.Spectral
    norm = mcolors.Normalize(vmin=np.min(lam), vmax=np.max(lam))
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # plot
    fig, axes = plt.subplots(2, 2, figsize=(7, 3.5), constrained_layout=True)
    axes = axes.flatten()

    for ax, (model_name, model_instance) in zip (axes, models.items()):

        mode = next(modes_list)
        m = next(modes)
        for i in range(len(lam[:])):
            p_set = ParameterSet(np.array([[lam[i], 3000, 3000, 0.1, 0.2]]))
            dataset = Dataset(initial_frequency_hz=20., srate_hz=4096.)
            wp_list = p_set.waveform_parameters(dataset)

            time_shifts = time_shifts_predictor.predict([wp_list[0].array])

            amp_res = np.log(
                np.abs(model_instance.predict_waveforms_bulk(p_set, model_instance.nn).amplitudes / 
                model_instance.dataset.generate_waveforms_from_params(p_set, model_instance.downsampling_indices).amplitudes)
            )[0]
            
            phase_res = (
                (model_instance.predict_waveforms_bulk(p_set, model_instance.nn).phases 
                - model_instance.dataset.generate_waveforms_from_params(p_set, model_instance.downsampling_indices).phases)
            )[0]

            freq_hz_phi = model_instance.dataset.frequencies_hz[model_instance.downsampling_indices.phase_indices]
            freq_natural_phi = freq_hz_phi * model_instance.dataset.mass_sum_seconds
            freq_hz_amp = model_instance.dataset.frequencies_hz[model_instance.downsampling_indices.amplitude_indices]
            freq_natural_amp = freq_hz_amp * model_instance.dataset.mass_sum_seconds

            phase_diff_w_ts = (
                phase_res 
                + 2 * np.pi * time_shifts * (freq_hz_phi - freq_hz_phi[0])
                - phase_res[0]
            )

            ax.set_xscale("log")
            ax.plot(freq_natural_phi, phase_diff_w_ts, color=sm.to_rgba(lam[i]))
            # ax.plot(freq_natural_amp, amp_res, color=sm.to_rgba(lam[i]))
            ax.set_title(rf'${mode}$')
            ax.grid(True)
            ax.set_xlabel(r'$Mf$')
    cbar = fig.colorbar(sm, ax=axes, location='right', shrink=0.8, aspect=20)
    cbar.set_label(r'$q$')
    # plt.tight_layout()
    fig.text(-0.03, 0.5, r'$\phi_{lm}^{\mathrm{MLGW}} - \phi_{lm}^{\mathrm{TEOB}} - 2 \pi f \Delta{t_{\mathrm{GPR}}}$', va='center', rotation='vertical')
    # fig.text(-0.03, 0.5, r'$\log(A_{\ell m}^{\mathrm{MLGW}}/A_{\ell m}^{\mathrm{TEOB}})$', va='center', rotation='vertical')
    plt.savefig('mlgw_vs_eob_21_22_33_44_phi.pdf', bbox_inches='tight')
    print("done!")

if __name__ == "__main__":
    plotting_mlgw()


