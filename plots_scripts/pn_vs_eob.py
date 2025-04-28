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

modes_model = ModesModel(modes=[Mode(2,2), Mode(3,3), Mode(2,1), Mode(4,4)])

model_22 = modes_model.models[Mode(2,2)]
model_21 = modes_model.models[Mode(2,1)]
model_33 = modes_model.models[Mode(3,3)]
model_44 = modes_model.models[Mode(4,4)]

models = {
    "model_21": model_21,
    "model_22": model_22,
    "model_33": model_33,
    "model_44": model_44
}

f_hz = np.arange(20, 2048, 0.0007153125)

time_shifts_predictor = TimeshiftsGPR().load_model(
    filename="ts_model_HOM_comp_pool.pkl"
)

def plot_pn_vs_eob():
    """
    Plot the amplitude & phase residuals between EOB and PN for (2,1), (2,2), (3,3) and (4,4) modes.
    """
    modes_list = cycle(['(\ell=2, m=1)', '(\ell=2, m=2)', '(\ell=3, m=3)', '(\ell=4, m=4)'])
    modes = cycle([Mode(2,1), Mode(2,2), Mode(3,3), Mode(4,4)])

    # create a colormap and normalize it to the range of mass_ratios
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
            p = WaveformParameters(
                mass_ratio=lam[i],
                lambda_1=3000, lambda_2=3000,
                chi_1=0.1, chi_2=0.2,
                dataset=Dataset(20., 4096.)
            )
            f_natural = p.dataset.hz_to_natural_units(f_hz)

            time_shifts = time_shifts_predictor.predict([p.array])

            f_spa, amp_teob, phase_teob = model_instance.waveform_generator.effective_one_body_waveform(p, f_natural)
            amp_pn = _post_newtonian_amplitudes_by_mode[m](p, f_spa)
            phase_pn = _post_newtonian_phases_by_mode[m](p, f_spa)

            log_teob_pn_amp = np.log(np.abs(amp_teob) / np.abs(amp_pn))
            phase_diff = (phase_teob - phase_pn)
            phase_diff_w_ts = (
                phase_diff - 2 * np.pi * ((f_spa - f_spa[0]) / p.dataset.mass_sum_seconds) * time_shifts 
            )

            ax.set_xscale("log")
            # ax.plot(f_spa, phase_diff_w_ts, color=sm.to_rgba(lam[i]))
            ax.plot(f_spa, log_teob_pn_amp, color=sm.to_rgba(lam[i]))
            ax.set_title(rf'${mode}$')
            ax.grid(True)
            ax.set_xlabel(r'$Mf$')
    cbar = fig.colorbar(sm, ax=axes, location='right', shrink=0.8, aspect=20)
    cbar.set_label(r'$q$')
    # fig.text(-0.03, 0.5, r'$\phi_{\ell m}^{\mathrm{EOB}} - \phi_{\ell m}^{\mathrm{PN}} - 2 \pi f \Delta{t_{\mathrm{GPR}}}$', va='center', rotation='vertical')
    fig.text(-0.03, 0.5, r'$\log(A_{\ell m}^{\mathrm{EOB}}/A_{\ell m}^{\mathrm{PN}})$', va='center', rotation='vertical')
    plt.savefig('pn_vs_eob_21_22_33_44_amp.pdf', bbox_inches='tight')
    print("done!")

def plot_lm():
    """
    Plot the phase difference between EOB and PN for a given mode.
    """
    lam = np.linspace(1.0, 3.0, num=10)
    cmap = plt.cm.Spectral
    norm = mcolors.Normalize(vmin=np.min(lam), vmax=np.max(lam))
    sm = ScalarMappable(norm=norm, cmap=cmap)

    plt.figure(figsize=(6, 3))
    ax = plt.axes()
    for i in range(len(lam)):
        p = WaveformParameters(
            mass_ratio=lam[i],
            lambda_1=3000, lambda_2=3000,
            chi_1=0.1, chi_2=0.2,
            dataset=Dataset(20., 4096.)
        )
        f_natural = p.dataset.hz_to_natural_units(f_hz)

        time_shifts = time_shifts_predictor.predict([p.array])

        f_spa, amp_teob, phase_teob = model_33.waveform_generator.effective_one_body_waveform(p, f_natural)
        amp_pn = _post_newtonian_amplitudes_by_mode[Mode(3,3)](p, f_spa)
        phase_pn_unwrap = _post_newtonian_phases_by_mode[Mode(3,3)](p, f_spa)

        log_teob_pn_amp = np.log(np.abs(amp_teob) / np.abs(amp_pn))
        phase_diff = (phase_teob - phase_pn_unwrap)
        phase_diff_w_ts = (
            phase_diff 
            - 2 * np.pi * ((f_spa - f_spa[0]) / p.dataset.mass_sum_seconds) * time_shifts 
            - phase_diff[0]
        )

        ax.set_xscale("log")
        ax.plot(f_spa, phase_diff_w_ts, color=sm.to_rgba(lam[i]))
        # ax.plot(f_spa, log_teob_pn_amp, color=sm.to_rgba(lam[i]))
        ax.set_title(r'$(\ell=2, m=2)$')
        ax.grid(True)
        ax.set_ylabel(r'$\phi_{lm}^{\mathrm{EOB}} - \phi_{lm}^{\mathrm{PN}} - 2 \pi f \Delta{t_{\mathrm{GPR}}}$')
        # ax.set_ylabel(r'$\log(A_{lm}^{\mathrm{EOB}}/A_{lm}^{\mathrm{PN}})$')
        ax.set_xlabel(r'$Mf$')
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label(r'$q$')
    plt.savefig('pn_vs_eob_lm_phi.png', bbox_inches='tight')
    print("done!")

if __name__ == "__main__":
    plot_pn_vs_eob()


