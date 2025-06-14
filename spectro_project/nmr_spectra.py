import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))
import pulp as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple
from matplotlib import cm, colors
from configuration import NAMES, ENV
from masserstein import estimate_proportions
from masserstein.nmr_spectrum import NMRSpectrum


def plot_spectrum(spectrum: NMRSpectrum, out_folder: str = "spectrum_plots") -> None:
    """ Plot a given NMRSpectrum and save the plot to a specified folder.
    
    Args:
        spectrum: NMRSpectrum object to plot.
        out_folder: Folder to save the plot (created if doesn't exist).
    """
    os.makedirs(out_folder, exist_ok=True)
    fname = os.path.join(out_folder, f"{spectrum.label}.png")

    plt.figure(figsize=(12, 6))
    plt.title(spectrum.label)
    for ppm, intensity in spectrum.confs:
        plt.vlines(ppm, 0, intensity, linewidth=1)
    plt.xlabel("ppm")
    plt.ylabel("Normalized intensity")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    
    print(f"Plot saved to {fname}")

def create_spectrum(path: str,
                    spectra_files: list[str],
                    labels: list[str] = None,
                    out_folder: str = "components_spectra") -> List[NMRSpectrum]:
    """
    Create NMRSpectrum objects from given CSV files and plot each spectrum.
    
    Args:
        spectra_dirs: List of root directories to search CSV files in.
        spectra_files: List of CSV filenames to process (basename only).
        labels: Optional list of labels to assign to each spectrum.
        out_folder: Folder to save spectrum plots (created if doesn't exist).
    """
    if labels is None:
        labels = spectra_files

    os.makedirs(out_folder, exist_ok=True)
    spectra = []

    for file, label in zip(spectra_files, labels):
        filepath = os.path.join(path, file)
        if not os.path.isfile(filepath):
            print(f"File not found: {filepath}")
            continue

        # Load data
        confs = np.loadtxt(filepath, delimiter=',', skiprows=1)
        try:
            spectrum = NMRSpectrum(confs=list(zip(confs[:, 0], confs[:, 1])), label=label)
        except IndexError:
            spectrum = NMRSpectrum(confs=[(confs[0], confs[1])], label=label)

        spectrum.trim_negative_intensities()
        spectrum.normalize()
        spectra.append(spectrum)
        plot_spectrum(spectrum, out_folder) # plot each spectrum
    
    print(f"Created {len(spectra)} spectra in '{out_folder}'")
    return spectra


def plot_spectral_regression(experiment: NMRSpectrum,
                             references: list[NMRSpectrum],
                             proportions: list[float],
                             xmin: float,
                             xmax: float,
                             save: bool = False,
                             folder: str = None,
                             kappa_components: float = None,
                             kappa_mixture: float = None,
                             labels: List[str] = NAMES) -> None:
    """ Plots a comparison of the experimental spectrum with
    the fitted components.

    Args:
        experiment (NMRSpectrum): spectrum.
        references (list[NMRSpectrum]): reference spectra.
        proportions (list[float]): estimated proportions.
        xmin (float), xmax (float): ppm range on the x-axis.
        save (bool): whether to save the plot.
        folder (str): output folder (if save=True).
    """
    plt.figure(figsize=(12, 6))

    # Plot the experimental spectrum with a label
    experiment.plot(profile=True, show=False,
                    linewidth=1.5, color='black')
    n_comp = sum(p > 0 for p in proportions)
    colors_b = cm.tab20b.colors
    colors_c = cm.tab20c.colors
    combined = list(colors_b) + list(colors_c)
    cmap = colors.ListedColormap(combined, name='tab40')
    color_indices = iter(range(n_comp))
    for ref, prop, label in zip(references, proportions, labels):
        if prop <= 0:
            continue
        xs, ys = zip(*ref.confs)
        ys_scaled = np.array(ys) * prop
        color = cmap(next(color_indices))
        markerline, stemlines, _ = plt.stem(xs, ys_scaled,
                                            linefmt='-',
                                            basefmt=' ',
                                            markerfmt=' ',
                                            label=f"{label} ({prop:.2f})"
                                    )
        plt.setp(stemlines, color=color)
        plt.setp(markerline, color=color)

    # Plot
    title = f'Experimental vs Estimated — {experiment.label}'
    if kappa_mixture is not None and kappa_components is not None:
        title += f'\nκ_mixture={kappa_mixture}, κ_components={kappa_components}'

    plt.xlim(xmax, xmin)
    plt.xlabel('ppm')
    plt.ylabel('Scaled Intensity')
    plt.title(title)
    plt.legend(loc='upper right', fontsize='small', framealpha=0.9)
    plt.tight_layout(rect=(0, 0, 0.75, 1))
    plt.legend(loc='center left',
               bbox_to_anchor=(1.02, 0.5),
               fontsize='small',
               framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Display or save the plot
    if save and folder:
        os.makedirs(folder, exist_ok=True)
        safe_name = experiment.label.replace('/', '_')
        out_path = os.path.join(folder, f"{safe_name}_{kappa_mixture}_{kappa_components}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")
    else:
        plt.show()


def analyze_with_solvers(experiment_spectrum : NMRSpectrum,
                         reference_spectra: list[NMRSpectrum],
                         kappa_mixture: float = 0.25,
                         kappa_components: float = 0.22,
                         out_folder: str = "analysis_plots",
                         solvers: List[pl.LpSolver] = [pl.LpSolverDefault, pl.GUROBI(env=ENV, msg=False)],
                         labels: List[str] = NAMES) -> None:
    """ Run estimation on given spectra using different solvers,
    plot results.

    Args:
        spectrum_dir: Path to folder with mixture spectra CSV.
        spectrum_files: Names of CSV files to analyze.
        reference_spectra: List of preprocessed NMRSpectrum objects.
        labels: Optional labels corresponding to spectrum_files.
        kappa_mixture: MTD parameter for mixture (κw).
        kappa_components: MTD threshold for components (κ_components).
        out_folder: Directory to save plots.
    """
    os.makedirs(out_folder, exist_ok=True)
    for solver in solvers:
        # Estimate proportions
        results = estimate_proportions(
            spectrum=experiment_spectrum,
            query=reference_spectra,
            MTD=kappa_mixture,
            MTD_th=kappa_components,
            solver=solver,
            verbose=True,
            what_to_compare = 'area',
        )

        # Show results
        plot_spectral_regression(
            experiment=experiment_spectrum,
            references=reference_spectra,
            proportions=results['proportions'],
            xmin=8.0,
            xmax=1.0,
            save=True,
            folder=os.path.join(out_folder, solver.__class__.__name__),
            kappa_components=kappa_components,
            kappa_mixture=kappa_mixture,
            labels=labels
        )

def compare_kappas(experiment_spectrum : NMRSpectrum,
                    reference_spectra: list[NMRSpectrum],
                    kappas_mixture: List[float] = [0.25],
                    kappas_components: List[float] = [0.22],
                    out_folder: str = "analysis_plots",
                    solvers: List[pl.LpSolver] = [pl.LpSolverDefault]):
    """ Compare results for different kappas values.

    Args:
        experiment_spectrum: NMRSpectrum object for the experimental spectrum.
        reference_spectra: List of NMRSpectrum objects for reference spectra.
        kappas: List of kappa values to test.
        out_folder: Directory to save comparison plots.
        solvers: List of solvers to use for estimation.
    """
    for kappa_mixture in kappas_mixture:
        for kappa_components in kappas_components:
            print(f"Analyzing with K_mixture={kappa_mixture}, K_components={kappa_components}")
            analyze_with_solvers(
                experiment_spectrum=experiment_spectrum,
                reference_spectra=reference_spectra,
                kappa_mixture=kappa_mixture,
                kappa_components=kappa_components,
                out_folder=out_folder,
                solvers=solvers
            )

def analyze_timecourse(input_folder: str,
                       reference_spectra: List[NMRSpectrum],
                       solvent: str = "5001",
                       kappa_mixture: float = 0.25,
                       kappa_components: float = 0.22,
                       solver=pl.PULP_CBC_CMD(msg=False),
                       days_range: Tuple[int, int] = (0, 200),
                       out_folder: str = "timecourse_results",
                       labels: List[str] = NAMES) -> pd.DataFrame:
    """
    Analyzes metabolite proportion changes over time plus control QC.
    At the end saves a plot and a CSV with results. Plot have reference
    lines for QC values with markers for each experiment day.

    Args:
        input_folder: Path to folder with time course CSV files.
        reference_spectra: List of preprocessed NMRSpectrum objects.
        solvent: Solvent identifier (e.g., "V5001").
        kappa_mixture: MTD parameter for mixture (κw).
        kappa_components: MTD threshold for components (κ_components).
        solver: Solver to use for estimation.
        days_range: Tuple specifying the range of days to analyze.
        out_folder: Directory to save results and plots.
        labels: List of metabolite names corresponding to reference spectra.
    """
    pattern = re.compile(rf"^(\d+)_V{re.escape(solvent)}_D(\d+)\.csv$", flags=re.IGNORECASE)
    experiment_days = {}
    qc_file = None
    for f in os.listdir(input_folder):
        if f.lower().endswith("qc.csv"):
            qc_file = f
        else:
            m = pattern.match(f)
            if m:
                day = int(m.group(2))
                if days_range[0] <= day <= days_range[1]:
                    experiment_days[day] = f

    records = []
    qc_props = {}

    # Process QC file if available
    if qc_file:
        arr = np.loadtxt(os.path.join(input_folder, qc_file), delimiter=',', skiprows=1)
        spec_qc = NMRSpectrum(confs=list(zip(arr[:,0], arr[:,1])), label="QC")
        spec_qc.trim_negative_intensities()
        spec_qc.normalize()
        res_qc = estimate_proportions(
            spectrum=spec_qc,
            query=reference_spectra,
            MTD=kappa_mixture,
            MTD_th=kappa_components,
            solver=solver,
            verbose=False,
            what_to_compare='area'
        )
        for label, prop in zip(labels, res_qc['proportions']):
            qc_props[label] = prop
            records.append({'day': None, 'component': label, 'proportion': prop})

    # Every day's spectrum
    for day, fname in sorted(experiment_days.items()):
        arr = np.loadtxt(os.path.join(input_folder, fname), delimiter=',', skiprows=1)
        spec = NMRSpectrum(confs=list(zip(arr[:,0], arr[:,1])), label=f"D{day}")
        spec.trim_negative_intensities()
        spec.normalize()
        res = estimate_proportions(
            spectrum=spec,
            query=reference_spectra,
            MTD=kappa_mixture,
            MTD_th=kappa_components,
            solver=solver,
            verbose=False,
            what_to_compare='area'
        )
        for label, prop in zip(labels, res['proportions']):
            records.append({'day': day, 'component': label, 'proportion': prop})

    df = pd.DataFrame(records)

    # Plotting results
    plt.figure(figsize=(12, 6))
    colors_b = cm.tab20b.colors
    colors_c = cm.tab20c.colors
    combined = list(colors_b) + list(colors_c)
    cmap = colors.ListedColormap(combined, name='tab40')
    label_colors = {label: cmap(i / len(labels)) for i, label in enumerate(labels)}

    for label in labels:
        sub = df[df['component'] == label]
        qc_val = qc_props.get(label, None)
        if qc_val is not None:
            plt.hlines(qc_val, xmin=days_range[0], xmax=days_range[1],
                       colors=label_colors[label], linestyles='--', alpha=0.5)

        sub_days = sub.dropna(subset=['day'])
        plt.plot(sub_days['day'], sub_days['proportion'],
                 color=label_colors[label], marker='x', linestyle='', label=label)

    plt.xlabel("Day")
    plt.ylabel("Proportion")
    plt.title("Analysis of Metabolites Over Time")
    plt.grid(ls='--', alpha=0.3)
    plt.tight_layout(rect=(0,0,0.8,1))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.tight_layout()

    # Save results
    os.makedirs(out_folder, exist_ok=True)
    plot_path = os.path.join(out_folder, "timecourse.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {plot_path}")

    csv_path = os.path.join(out_folder, "timecourse_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved data: {csv_path}")

    return df
    