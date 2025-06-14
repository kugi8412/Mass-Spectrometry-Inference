import re
import os
import glob
import zipfile
import pandas as pd
import multiprocessing as mp
import xml.etree.ElementTree as ET

from tqdm import tqdm
from typing import List, Tuple, Dict
from collections import defaultdict

### Data preparation and extraction ###
def extract_nested_zips(main_zip_path: str,
                        extract_to: str) -> None:
    """ Extract nested zip files from the main main zip file
    to the specified directory.
    """
    with zipfile.ZipFile(main_zip_path, 'r') as main_zip:
        main_zip.extractall(extract_to)
    
    # All inner HMDB zip files
    hmdb_zips = glob.glob(os.path.join(extract_to, "hmdb_nmr_spectra_*.zip"))
    for hmdb_zip in tqdm(hmdb_zips, desc="Extracting HMDB zips"):
        with zipfile.ZipFile(hmdb_zip, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(extract_to, os.path.splitext(os.path.basename(hmdb_zip))[0]))
        os.remove(hmdb_zip)
    
    # Extract urine data
    urine_zip = os.path.join(extract_to, "Data_urine.zip")
    if os.path.exists(urine_zip):
        with zipfile.ZipFile(urine_zip, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(extract_to, "Data_urine"))
        os.remove(urine_zip)


### Parsing single XML file for spectra extraction ###
def process_single_xml(file_info: str) -> List[str] | None:
    """ Process a single XML file for spectra extraction.
    """
    # Helper function to find elements
    def find_elem(parent, tag_name):
        elem = parent.find(f'.//{ns_prefix}{tag_name}')
        if elem is not None:
            return elem

        return parent.find(f'.//{tag_name}')
    
    filepath, expected_nucleus, output_dir = file_info

    if not filepath.endswith('.xml'):
        return None
        
    try:
        # Parse XML file
        tree = ET.parse(filepath)
        root = tree.getroot()
        if '}' in root.tag:
            ns_uri = root.tag.split('}')[0][1:]
            ns_prefix = '{' + ns_uri + '}'
        else:
            ns_prefix = ''

        nucleus_elem = find_elem(root, 'nucleus')
        if nucleus_elem is None or nucleus_elem.text is None:
            print(f"No nucleus information in {filepath}")
            return None

        actual_nucleus = nucleus_elem.text.strip()

        if actual_nucleus != expected_nucleus:
            print(f"Skipping {filepath}: Expected nucleus {expected_nucleus}, found {actual_nucleus}")
            return None
        
        accession = None
        db_id_elem = find_elem(root, 'database-id')
        if db_id_elem is not None and db_id_elem.text:
            accession = db_id_elem.text.strip()

        if not accession:
            accession_elem = find_elem(root, 'accession')
            if accession_elem is not None and accession_elem.text:
                accession = accession_elem.text.strip()

        if not accession:
            structure_elem = find_elem(root, 'structure-id')
            if structure_elem is not None and structure_elem.text:
                accession = f"HMDB{structure_elem.text.zfill(7)}"
        
        if not accession:
            print(f"No HMDB ID found in {filepath}")
            return None
        
        # Determine spectrum type
        predicted_elem = find_elem(root, 'predicted')
        spectrum_type = "experimental"
        if predicted_elem is not None and predicted_elem.text:
            if predicted_elem.text.strip().lower() == "true":
                spectrum_type = "predicted"
        
        # Get frequency
        freq_elem = find_elem(root, 'frequency')
        freq = "unknown"
        if freq_elem is not None and freq_elem.text:
            freq = freq_elem.text.strip().replace(" ", "_")
        
        # Get spectrum ID
        spectrum_id_elem = find_elem(root, 'id')
        spectrum_id = spectrum_id_elem.text if spectrum_id_elem is not None else os.path.splitext(os.path.basename(filepath))[0]
        
        peaks = []
        peaks_container = find_elem(root, 'nmr-one-d-peaks')
        if peaks_container is None:
            peaks_container = root
            
        # Find all peak elements
        peak_elements = peaks_container.findall(f'.//{ns_prefix}nmr-one-d-peak') or peaks_container.findall('.//nmr-one-d-peak')
        
        for peak in peak_elements:
            chem_shift_elem = peak.find(f'{ns_prefix}chemical-shift') or peak.find('chemical-shift')
            if chem_shift_elem is None:
                chem_shift_elem = peak.find(f'{ns_prefix}chemical_shift') or peak.find('chemical_shift')
                
            peak_pos_elem = peak.find(f'{ns_prefix}peak-position-ppm') or peak.find('peak-position-ppm')
            if peak_pos_elem is None:
                peak_pos_elem = peak.find(f'{ns_prefix}peak_position_ppm') or peak.find('peak_position_ppm')
                
            intensity_elem = peak.find(f'{ns_prefix}intensity') or peak.find('intensity')

            if chem_shift_elem is None or intensity_elem is None:
                continue
                
            try:
                chem_shift = float(chem_shift_elem.text)
                peak_pos = float(peak_pos_elem.text) if peak_pos_elem is not None and peak_pos_elem.text else chem_shift
                intensity = float(intensity_elem.text)
                peaks.append([chem_shift, peak_pos, intensity])
            except (ValueError, TypeError) as e:
                print(f"Error parsing peak in {filepath}: {str(e)}")
                continue
        
        if not peaks:
            print(f"No valid peaks found in {filepath}")
            return None
            
        # Create output dataframe
        df = pd.DataFrame(peaks, columns=['chemical_shift', 'peak_position_ppm', 'intensity'])
        out_name = f"{accession}.1D.{spectrum_id}.{spectrum_type}.{actual_nucleus}.{freq}.csv"
        out_path = os.path.join(output_dir, out_name)
        df.to_csv(out_path, index=False)
        return [out_path]
        
    except ET.ParseError as e:
        print(f"XML parsing error in {filepath}: {str(e)}")
    except Exception as e:
        print(f"Unexpected error processing {filepath}: {str(e)}")
    
    return None


def extract_1D_spectra(root_folder: str,
                       nucleus: str,
                       output_dir: str,
                       num_workers=None) -> pd.DataFrame:
    """
    Extract 1D NMR spectra from HMDB XML files using multiprocessing.
    
    Args:
        root_folder (str): Root folder containing HMDB data subfolders.
        nucleus (str): Nucleus type ('1H'; '13C').
        output_dir (str): Directory to save extracted CSV files.
        num_workers (int): Number of parallel processes.

    Returns:
        pd.DataFrame: DataFrame containing the extracted spectra.
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    os.makedirs(output_dir, exist_ok=True)
    
    xml_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.xml'):
                xml_files.append(os.path.join(dirpath, filename))
    
    file_infos = [(f, nucleus, output_dir) for f in xml_files]
    
    print(f"Processing {len(xml_files)} XML files with {num_workers} workers...")
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_single_xml, file_infos), total=len(file_infos), desc="Extracting spectra"))
    
    # Write csv files
    output_files = [f for sublist in results if sublist for f in sublist]
    print(f"Created {len(output_files)} CSV files")
    return output_files


def preprocess_single_spectrum(file_info: str) -> List[str] | None:
    """ Preprocess a single spectrum file.
    """
    filename, out_folder, sig = file_info
    try:
        df = pd.read_csv(filename)
        
        if 'peak_position_ppm' not in df.columns:
            print(f"Missing columns in {filename}")
            return None

        if sig is not None:
            df['peak_position_ppm'] = df['peak_position_ppm'].round(sig)
        
        os.makedirs(out_folder, exist_ok=True)

        grouped = df.groupby('peak_position_ppm')['intensity'].sum().reset_index()
        base = os.path.basename(filename)
        out_path = os.path.join(out_folder, f"{os.path.splitext(base)[0]}_processed.csv")
        grouped.to_csv(out_path, index=False)
        return out_path
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None


### Preprocessing spectra ###
def preprocess_1D_spectra(filenames: List[str],
                          out_folder: str = "../Preprocessed",
                          sig: int | None = None,
                          num_workers: int | None = None) -> List[str] | None:
    """
    Preprocess extracted spectra using multiprocessing.

    Args:
        filenames (list): List of input CSV files.
        out_folder (str): Output directory.
        sig (int): Significant digits for rounding (optional).
        num_workers (int): Number of parallel processes.
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    file_infos = [(f, out_folder, sig) for f in filenames]
    
    print(f"Preprocessing {len(filenames)} spectra with {num_workers} workers...")
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(preprocess_single_spectrum, file_infos), total=len(file_infos), desc="Preprocessing"))

    processed_files = [f for f in results if f is not None]
    print(f"Created {len(processed_files)} processed spectra")
    return processed_files


def find_best_spectra(selected_metabolites: str,
                    spectra_folder: str) -> None:
    """
    Find the best experimental spectra for selected metabolites.
    
    Args:
        selected_metabolites (str): Path to CSV file with selected metabolites.
        spectra_folder (str): Folder containing the spectra CSV files.
    """
    sel = pd.read_csv(selected_metabolites)
    hmdb_list = sel['HMDB.ca ID'].tolist()
    counts = {"1H": 0, "13C": 0}
    counts_type = {"predicted": 0, "experimental": 0}
    # HMDB0341414.1D.282437.predicted.13C.900_processed.csv
    pattern = re.compile(
    r'^(HMDB\d{7})\.1D\.(\d+)\.(predicted|experimental)\.(1H|13C)\.(\d+(?:\.\d+)?)' +
    r'(?:_MHz)?_processed\.csv$'
    )
    # HMDB_ID -> (frequency:float, filename:str, nucleus, type)
    best = {}

    for fn in os.listdir(spectra_folder):
        m = pattern.match(fn)
        if not m:
            print("Niepasujący plik:", fn)
            continue

        hid, _, stype, nucleus, freq = m.groups()
        freq = float(freq)
        counts[nucleus] += 1
        counts_type[stype] += 1

        if hid in hmdb_list and stype == "experimental" and nucleus == "1H":
            curr = best.get(hid)
            if curr is None or freq > curr[0]:
                best[hid] = (freq, fn, nucleus, stype)

    # Show results
    print("File statistics in the folder:")
    print(f"1H spectra: {counts['1H']}")
    print(f"13C spectra: {counts['13C']}")
    print(f"Experimental: {counts_type['experimental']}")
    print(f"Predicted: {counts_type['predicted']}")
    
    return best

def process_xml_file(file_info: List[str]) -> Dict[str, int]:
    """ Process a single XML file to extract metabolite
    detection and quantification info.
    """
    xml_file, expected_nucleus = file_info
    results = {}
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if '}' in root.tag:
            ns_uri = root.tag.split('}')[0][1:]
            ns_prefix = '{' + ns_uri + '}'
        else:
            ns_prefix = ''

        def find_elem(parent, tag):
            elem = parent.find(f'.//{ns_prefix}{tag}')
            if elem is not None:
                return elem
            return parent.find(f'.//{tag}')

        nucleus_elem = find_elem(root, 'nucleus')
        if nucleus_elem is None or nucleus_elem.text is None:
            return results
        actual_nucleus = nucleus_elem.text.strip()

        if actual_nucleus != expected_nucleus:
            return results

        accession = None
        for tag in ['database-id', 'accession']:
            elem = find_elem(root, tag)
            if elem is not None and elem.text:
                accession = elem.text.strip()
                break
        
        if not accession:
            return results
        
        # Determine spectrum type
        predicted_elem = find_elem(root, 'predicted')
        spectrum_type = "experimental"
        if predicted_elem is not None and predicted_elem.text:
            if predicted_elem.text.strip().lower() == "true":
                spectrum_type = "predicted"

        quantified = False
        conc_elem = find_elem(root, 'sample-concentration')
        if conc_elem is not None and conc_elem.text and conc_elem.text.strip().lower() not in ["", "nil", "null"]:
            try:
                float(conc_elem.text)
                quantified = True
            except ValueError:
                pass
        
        # Return results for this file
        results[accession] = {
            'experimental': spectrum_type == "experimental",
            'quantified': quantified,
            'file': xml_file
        }
        
    except Exception as e:
        pass
    
    return results

def analyze_hmdb_metabolites(xml_dirs: List[str],
                             nucleus='1H',
                             num_workers=None) -> Tuple[int]:
    """
    Analyze all HMDB XML files in folders to determine:
    - Metabolites with experimental spectra
    - Detected and quantified metabolites
    - Detected but not quantified metabolites
    
    Args:
        xml_dirs: List of directories containing HMDB XML files
        nucleus: Nucleus type to consider (default '1H')
        num_workers: Number of parallel processes
        
    Returns:
        Tuple of counts: (total_detected, with_experimental, detected_quantified, detected_not_quantified)
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # Collect all XML files
    xml_files = []
    for xml_dir in xml_dirs:
        for root, _, files in os.walk(xml_dir):
            for file in files:
                if file.endswith('.xml'):
                    xml_files.append(os.path.join(root, file))
    
    print(f"Found {len(xml_files)} XML files for analysis")
    
    tasks = [(f, nucleus) for f in xml_files]
    all_results = defaultdict(list)
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_xml_file, tasks), total=len(tasks), desc="Processing XML files"))

    for result in results:
        for hmdb_id, data in result.items():
            all_results[hmdb_id].append(data)

    metabolites = set(all_results.keys())
    total_detected = len(metabolites)
    with_experimental = 0
    detected_quantified = 0
    detected_not_quantified = 0
    
    for hmdb_id, spectra_list in all_results.items():
        has_experimental = any(s['experimental'] for s in spectra_list)
        has_quantified = any(s['quantified'] for s in spectra_list)
        
        if has_experimental:
            with_experimental += 1
        
        if has_quantified:
            detected_quantified += 1
        elif has_experimental:
            detected_not_quantified += 1
    
    return total_detected, with_experimental, detected_quantified, detected_not_quantified
