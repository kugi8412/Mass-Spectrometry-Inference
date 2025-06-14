import os
import gurobipy as gp
import multiprocessing as mp


SEED = 42
BASE_DIR = os.getcwd()
NUM_WORKERS = mp.cpu_count() - 1  # One worker free for system tasks
HMDB_ROOT = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data")
REFERENCE_FILES = ["HMDB0000023.1D.1037.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000042.1D.1048.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000064.1D.1064.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000087.1D.1077.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000094.1D.1080.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000123.1D.1094.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000142.1D.1107.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000161.1D.1120.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000190.1D.1162.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000211.1D.1200.experimental.1H.600_MHz_processed.csv",
                    "HMDB0000254.1D.1285.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000300.1D.1321.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000462.1D.1380.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000562.1D.1420.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000682.1D.1475.experimental.1H.600_MHz_processed.csv",
                    "HMDB0000714.1D.1494.experimental.1H.600_MHz_processed.csv",
                    "HMDB0000754.1D.1523.experimental.1H.600_MHz_processed.csv",
                    "HMDB0000767.1D.1533.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000812.1D.1553.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000875.1D.1579.experimental.1H.600_MHz_processed.csv",
                    "HMDB0000906.1D.1598.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000925.1D.1605.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000956.1D.1621.experimental.1H.600_MHz_processed.csv",
                    "HMDB0001870.1D.1769.experimental.1H.600_MHz_processed.csv",
                    "HMDB0003911.1D.1985.experimental.1H.600_MHz_processed.csv",
                    "HMDB0011635.1D.2052.experimental.1H.500_MHz_processed.csv"
    ]

QC_FILE = ["1541_QC.csv"]

EXPERIMENTS_FILES = ["1_V5001_D1.csv",
                     "10_V5001_D10.csv",
                     "100_V5001_D100.csv",
                     "255_V5001_D716.csv",
]

NAMES = ["S-3-hydroxyisobutyrate",
        "acetic acid",
        "creatine",
        "dimethylamine",
        "citric acid",
        "glycine",
        "formic acid",
        "alanine",
        "lactic acid",
        "myo-inositol",
        "succinic acid",
        "uracil",
        "allantoin",
        "creatinine",
        "indoxylsulfate",
        "hippuric acid",
        "3-hydroxyisovalerate",
        "pseudouridine",
        "N-acetyl-aspartic acid",
        "trigonelline",
        "trimethylamine",
        "trimethylamine-N-oxide",
        "tartaric acid",
        "benzoic acid",
        "3-aminoisobutyrate",
        "p-cresol sulfate"
]

ENV = gp.Env(params={
        "WLSACCESSID": 'd706cc33-4893-49c0-9aca-ee980c0e6300',
        "WLSSECRET": '1f88265b-151f-4ce2-aed2-cd8d65d4490a',
        "LICENSEID": 2659112,
    }
)

EXP_REFERENCE_FILES = ["HMDB0000023.1D.1037.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000042.1D.1048.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000064.1D.1064.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000087.1D.1077.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000094.1D.1080.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000123.1D.1094.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000142.1D.1107.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000161.1D.1120.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000190.1D.1162.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000211.1D.1200.experimental.1H.600_MHz_processed.csv",
                    "HMDB0000254.1D.1285.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000292.1D.1316.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000294.1D.1317.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000300.1D.1321.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000462.1D.1380.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000562.1D.1420.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000682.1D.1475.experimental.1H.600_MHz_processed.csv",
                    "HMDB0000714.1D.1494.experimental.1H.600_MHz_processed.csv",
                    "HMDB0000754.1D.1523.experimental.1H.600_MHz_processed.csv",
                    "HMDB0000767.1D.1533.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000812.1D.1553.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000875.1D.1579.experimental.1H.600_MHz_processed.csv",
                    "HMDB0000906.1D.1598.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000925.1D.1605.experimental.1H.500_MHz_processed.csv",
                    "HMDB0000956.1D.1621.experimental.1H.600_MHz_processed.csv",
                    "HMDB0001366.1D.1688.experimental.1H.500_MHz_processed.csv",
                    "HMDB0001870.1D.1769.experimental.1H.600_MHz_processed.csv",
                    "HMDB0001991.1D.1830.experimental.1H.600_MHz_processed.csv",
                    "HMDB0003099.1D.1927.experimental.1H.600_MHz_processed.csv",
                    "HMDB0003911.1D.1985.experimental.1H.600_MHz_processed.csv",
                    "HMDB0011635.1D.2052.experimental.1H.500_MHz_processed.csv"
    ]

EXP_NAMES = ["S-3-hydroxyisobutyrate",
        "acetic acid",
        "creatine",
        "dimethylamine",
        "citric acid",
        "glycine",
        "formic acid",
        "alanine",
        "lactic acid",
        "myo-inositol",
        "succinic acid",
        "xanthine"
        "urea",
        "uracil",
        "allantoin",
        "creatinine",
        "indoxylsulfate",
        "hippuric acid",
        "3-hydroxyisovalerate",
        "pseudouridine",
        "N-acetyl-aspartic acid",
        "trigonelline",
        "trimethylamine",
        "trimethylamine-N-oxide",
        "tartaric acid",
        "purine",
        "benzoic acid",
        "7-methylxanthine",
        "1-methyluric acid",
        "3-aminoisobutyrate",
        "p-cresol sulfate"
]
