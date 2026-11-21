# Individual coverage of atrial fibrillation burden intervals

Analysis code for *External validation of individual coverage for atrial fibrillation burden intervals from twenty sampled Holter minutes*. The study estimates the fraction of evaluable Holter time in atrial fibrillation from twenty reviewed minutes and examines how often each interval covers an individual patient’s annotated burden.

The repository contains the sampling and interval algorithms, detector integration, data preparation, evaluation scripts, figure generators and synthetic tests. The version 3 manuscript supplement supplies the frozen minute-level inputs, calibration parameters and reference tables for exact offline replay. The investigators’ original source releases supply the ECG and rhythm annotations.

## Citation and license

Mohamed Kone, Gaoussou Haidara, Chao Hou, Shulin Wang. *Individual coverage of atrial fibrillation burden intervals from twenty sampled Holter minutes* (software version 1.0.0). The four creators are affiliated with the School of Computer Science and Engineering, Hunan University, Changsha, China. Zenodo archives this version under [doi:10.5281/zenodo.24408832](https://doi.org/10.5281/zenodo.24408832). Machine-readable citation metadata are available in [CITATION.cff](CITATION.cff).

The original study code is available under the [MIT License](LICENSE). The ECG databases, annotations and ArNet2 model are distributed by their respective providers under the terms listed in [SOURCES.md](SOURCES.md).

## Run the synthetic tests

Use Python 3.12. The study replay used Linux x86-64 with NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3 and Matplotlib 3.10.8. These tests create their own inputs.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python Study/holter_intervals/test_finite.py
python COSEn/test_detector.py
python Study/holter_intervals/second/test_arnet.py
python Study/holter_pilot/test_reference.py
```

## Reproduce the paper from the supplement

Obtain `Holter_AF_burden_supplement_v3.zip` accompanying the manuscript. Put it beside the repository and run the commands from the repository root. Choose unused output directories outside the checkout. The staging command authenticates the ZIP by SHA-256 and checks the repository’s archived files against the supplement. Once the environment and archive are available, all three commands below run offline on CPU.

```bash
python prepare_replay.py ../Holter_AF_burden_supplement_v3.zip --output ../holter-v3-inputs
python ../holter-v3-inputs/Study/holter_intervals/reproduce.py --output ../holter-primary
python ../holter-v3-inputs/reproduce_revision.py --output ../holter-secondary
```

The primary replay checks all 744 detector/recording minute files, regenerates two calibrated models and compares nine numerical result tables byte for byte. The secondary replay regenerates 18 secondary tables, the four-row editorial lookup and Figure 2. To check the paired Belgian results and redraw both paper figures directly from archived tables:

```bash
python Methods/check_manuscript_numbers.py ../holter-v3-inputs/Study/Holter_intervalles_code_et_resultats.zip
python Figure1/make_figure1.py --results ../holter-v3-inputs/Study/holter_intervals/results --out ../holter-figure1
python Figure2/make_figure2.py --draws ../holter-v3-inputs/Revision/diagnostics/five_cases_draws.csv.gz --out ../holter-figure2
```

## Navigate the code

| Question | Files |
| --- | --- |
| How are sampled minutes selected and interval bounds formed? | `Study/holter_intervals/pilot_estimators.py`, `finite.py`, `study.py`, and `Methods/Interval_methods.txt` |
| How were the protocol, estimators and replay inputs fixed? | `Study/holter_intervals/protocol.md`, `methods_freeze.sha256`, `reproduce.py`, and this release’s `FROZEN_SOURCES.sha256` |
| How are rhythm annotations and detector predictions translated into minute values? | `Study/holter_pilot/`, `COSEn/`, and the `cohorts/`, `second/` and `prepare.py` files under `Study/holter_intervals/` |
| What produces the cutoff, repeat-simulation, miss-pattern and calibration-transfer tables? | `Revision/thresholds/`, `Revision/diagnostics/`, `Revision/transfer/`, and `Editorial/` |
| Where are the figures generated? | `Figure1/make_figure1.py` and `Figure2/make_figure2.py` |

The source inventory in `FROZEN_SOURCES.sha256` covers 45 files from the version 3 supplement and two reference-processing files from the precursor pilot. The four libraries and their versions are pinned in the two requirements files. See [public input sources and reconstruction](SOURCES.md) for the optional raw-input workflow and the terms of the individual resources.
