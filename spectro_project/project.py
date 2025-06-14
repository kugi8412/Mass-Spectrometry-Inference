import os
import random
import pulp as pl
import numpy as np
import pandas as pd

from nmr_spectra import create_spectrum, analyze_with_solvers, compare_kappas, analyze_timecourse
from data_processing import extract_nested_zips, extract_1D_spectra, preprocess_1D_spectra, find_best_spectra, analyze_hmdb_metabolites
from configuration import SEED, BASE_DIR, NUM_WORKERS, OUTPUT_DIR, REFERENCE_FILES, QC_FILE, EXP_REFERENCE_FILES, HMDB_ROOT, EXP_NAMES, NAMES, EXPERIMENTS_FILES


### MAIN ###
if __name__ == "__main__":
    np.random.seed(SEED)
    random.seed(SEED)
    print(f"Using {NUM_WORKERS} workers for parallel processing")

    # Create necessary directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "hmdb_extracted"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "hmdb_preprocessed"), exist_ok=True)
    
    # Extract nested zip files if needed
    if not os.path.exists(HMDB_ROOT):
        print("Extracting nested zip files:")
        extract_nested_zips(os.path.join(BASE_DIR, "data.zip"), HMDB_ROOT)
        extract_nested_zips(os.path.join(BASE_DIR, "Data_urine.zip"), HMDB_ROOT)

    # Process HMDB data
    hmdb_extracted = []
    for i in range(1, 6):
        part_dir = os.path.join(HMDB_ROOT, f"hmdb_nmr_spectra_{i}", f"hmdb_nmr_spectra_{i}")
        if os.path.exists(part_dir):
            extracted = extract_1D_spectra(
                root_folder=part_dir,
                nucleus="1H",  # or "13C"
                output_dir=os.path.join(OUTPUT_DIR, "hmdb_extracted"),
                num_workers=NUM_WORKERS
            )
            hmdb_extracted.extend(extracted)

    # Preprocess HMDB spectra
    if hmdb_extracted:
        hmdb_preprocessed = preprocess_1D_spectra(
            filenames=hmdb_extracted,
            out_folder=os.path.join(OUTPUT_DIR, "hmdb_preprocessed"),
            sig=4,
            num_workers=NUM_WORKERS
        )
    
    # Analyze HMDB metabolites
    total_detected, with_experimental, detected_quantified, detected_not_quantified = analyze_hmdb_metabolites(
        ["data/hmdb_nmr_spectra_1", "data/hmdb_nmr_spectra_2", "data/hmdb_nmr_spectra_3", "data/hmdb_nmr_spectra_4", "data/hmdb_nmr_spectra_5"],
        "1H",
        NUM_WORKERS
    )

    best_files = find_best_spectra(
        selected_metabolites=os.path.join(HMDB_ROOT, "Data_urine", "selected_metabolites.csv"),
        spectra_folder=os.path.join(OUTPUT_DIR, "hmdb_preprocessed"),
    )

    selected_metabolites = pd.read_csv(os.path.join(HMDB_ROOT, "Data_urine", "selected_metabolites.csv"))
    for id, (freq, fn, nucleus, stype) in best_files.items():
        name = selected_metabolites.loc[selected_metabolites['HMDB.ca ID'] == id, 'name'].iloc[0]
        print(f"{id} ({name}): {fn}, {freq} MHz")

    # Create spectra for quality control and experiments
    mix = create_spectrum(
        path=os.path.join(HMDB_ROOT, "Data_urine", "Urine_samples"),
        spectra_files=QC_FILE,
        labels=["Quality Control"],
        out_folder=os.path.join(OUTPUT_DIR, "spectra_plots")
    )

    mixtures = create_spectrum(
        path=os.path.join(HMDB_ROOT, "Data_urine", "Urine_samples"),
        spectra_files=EXPERIMENTS_FILES,
        labels=["Day 1", "Day 10", "Day 100", "Day 716"],
        out_folder=os.path.join(OUTPUT_DIR, "spectra_plots")
    )

    reference_files = create_spectrum(
        path=os.path.join(OUTPUT_DIR, "hmdb_preprocessed"),
        spectra_files=REFERENCE_FILES,
        labels=[label.split('.')[0] for label in REFERENCE_FILES],
        out_folder=os.path.join(OUTPUT_DIR, "reference_spectra_plots")
    )

    analyze_with_solvers(
        experiment_spectrum=mix[0],
        reference_spectra=reference_files,
        out_folder=os.path.join(OUTPUT_DIR, "solver_results")
    )
    
    compare_kappas(
        experiment_spectrum=mix[0],
        reference_spectra=reference_files,
        kappas_mixture=[0.20, 0.25, 0.30],
        kappas_components=[0.20, 0.22, 0.25],
        out_folder=os.path.join(OUTPUT_DIR, "kappa_comparison"),
        solvers=[pl.LpSolverDefault]
    )

    results = analyze_timecourse(input_folder=os.path.join(HMDB_ROOT, "Data_urine", "Urine_samples"),
                                  reference_spectra=reference_files,
                                  solvent="5001",
                                  kappa_mixture=0.30,
                                  kappa_components=0.20,
                                  solver=pl.LpSolverDefault,
                                  days_range=(0, 255),
                                  out_folder="timecourse_results",
                                  labels=NAMES
    )

    results = analyze_timecourse(input_folder=os.path.join(HMDB_ROOT, "Data_urine", "Urine_samples"),
                                  reference_spectra=reference_files,
                                  solvent="5001",
                                  kappa_mixture=0.30,
                                  kappa_components=0.20,
                                  solver=pl.LpSolverDefault,
                                  days_range=(700, 800),
                                  out_folder="add_timecourse_results",
                                  labels=NAMES
    )

    exp_reference_files = create_spectrum(
        path=os.path.join(OUTPUT_DIR, "hmdb_preprocessed"),
        spectra_files=EXP_REFERENCE_FILES,
        labels=[label.split('.')[0] for label in EXP_REFERENCE_FILES],
        out_folder=os.path.join(OUTPUT_DIR, "experiment_reference_spectra_plots")
    )

    # Analyze mixtures with solvers and compare results
    for el in (mix + mixtures):
        analyze_with_solvers(
            experiment_spectrum=el,
            reference_spectra=reference_files,
            kappa_mixture=0.30,
            kappa_components=0.20,
            solvers=[pl.LpSolverDefault],
            out_folder=os.path.join(OUTPUT_DIR, "original_results"),
            labels=NAMES
        )
        analyze_with_solvers(
            experiment_spectrum=el,
            reference_spectra=exp_reference_files,
            kappa_mixture=0.30,
            kappa_components=0.20,
            solvers=[pl.LpSolverDefault],
            out_folder=os.path.join(OUTPUT_DIR, "experiment_results"),
            labels=EXP_NAMES
        )

    results = analyze_timecourse(input_folder=os.path.join(HMDB_ROOT, "Data_urine", "Urine_samples"),
                                  reference_spectra=exp_reference_files,
                                  solvent="5001",
                                  kappa_mixture=0.30,
                                  kappa_components=0.20,
                                  solver=pl.LpSolverDefault,
                                  days_range=(0, 255),
                                  out_folder="task3_results",
                                  labels=EXP_NAMES
    )

    print("Analysis complete. Results saved in the output directory!")
