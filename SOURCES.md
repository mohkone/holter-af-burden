# Input sources and reconstruction

## Source editions

| Resource | Version and attribution | Terms for the upstream resource |
| --- | --- | --- |
| Long Term AF Database | George Moody and Steven Swiryn. Annotations from MEDICALgorithmics, coordinated by Michael Tadeusiak. [LTAFDB 1.0.0](https://doi.org/10.13026/C2QG6Q) | Open Data Commons Attribution License 1.0 |
| SHDB-AF | Kenta Tsutsui, Shany Biton Brimer and Joachim Behar. [SHDB-AF 1.0.1](https://doi.org/10.13026/n6yq-fq90) | Open Data Commons Attribution License 1.0 |
| MIT-BIH Atrial Fibrillation Database | George Moody and Roger Mark. [AFDB 1.0.0](https://doi.org/10.13026/C2MW2D) | Open Data Commons Attribution License 1.0 |
| IRIDIA-AF | Cédric Gilon, Jean-Marie Grégoire, Marianne Mathieu, Stéphane Carlier and Hugues Bersini. [IRIDIA-AF 1.0.1](https://doi.org/10.5281/zenodo.8405941) | CC BY 4.0 for the dataset, as documented in the study provenance |
| ArNet2 | Shany Biton and colleagues, AIM Lab. [Study](https://doi.org/10.1038/s41746-023-00791-1) and [AFtoolkit snapshot](https://github.com/aim-lab/AFtoolkit/tree/aa48a24c34095c8749e977ec5ad8b1442717c81b) | CC BY-NC 4.0 for the upstream model |

The [MIT License](LICENSE) applies to the original study code. The resource-specific terms in the table above govern reuse of third-party data and the ArNet2 model. Obtain patient recordings and annotations from the cited releases. The ArNet2 adapter loads the published model separately and verifies its file digests. The Open Data Commons and ArNet2 notices under `Study/holter_intervals/` document upstream terms. Consult [PhysioNet’s platform citation](https://doi.org/10.1038/s44360-026-00096-z) and the database pages for the source publications. COSEn is the project’s implementation of the [Lake and Moorman method](https://doi.org/10.1152/ajpheart.00561.2010).

## Build primary inputs from public releases

Use a separate working copy for this optional workflow. Acquisition writes source files and recording-level outputs beneath that copy. Internet access is needed to fetch database releases and the ArNet2 model. Source availability and download speed can vary. Use the two separate Python environments below because TensorFlow 2.18.1 uses a different NumPy version from the statistical replay. The following commands assume Bash on Linux from the root of the working copy.

```bash
root="$PWD"
python3.12 -m venv .numerical-env
num="$root/.numerical-env/bin/python"
"$num" -m pip install -r requirements.txt h5py

python3.12 -m venv .model-env
model="$root/.model-env/bin/python"
"$model" -m pip install 'tensorflow-cpu==2.18.1'

pilot="$root/Study/holter_pilot"
study="$root/Study/holter_intervals"
cp COSEn/detector.py COSEn/detector_acquire.py "$pilot/"
"$num" "$pilot/reference.py"
"$num" "$pilot/detector_acquire.py"
"$num" "$pilot/detector.py"
"$num" "$study/cohorts/new_data.py"
"$num" "$study/cohorts/afdb_extra.py"

"$num" "$study/second/fetch_arnet.py"
for cohort in LTA SHDB; do
    "$model" "$study/second/arnet.py" --pilot "$pilot" --cohort "$cohort"
done
for cohort in IRIDIA AFDB; do
    "$model" "$study/second/predict_new.py" --source "$study/cohorts/derived" --cohort "$cohort"
done

mkdir -p "$study/results"
for detector in ArNet2 COSEn; do
    for cohort in LTA SHDB IRIDIA AFDB; do
        "$num" "$study/prepare.py" "$cohort" "$detector"
    done
    "$num" "$study/study.py" train --det "$detector"
    for cohort in SHDB IRIDIA AFDB; do
        "$num" "$study/study.py" evaluate --det "$detector" --cohort "$cohort"
    done
done
"$num" "$study/study.py" aggregate
```

The study protocol and `cohorts/SOURCES.md` document sample eligibility. The original design forms 744 detector/recording input files. The study’s offline replay additionally tests file-level equality against the version 3 supplement. `Study/holter_pilot/reference.py` and its tests come from the archived pilot, `Holter_20_minutes_code_et_resultats.zip`. The detector and acquisition scripts copied into that directory are the files bundled with the version 3 manuscript.
