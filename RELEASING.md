# Archiving a code release

The software citation and the license for the original study code are recorded in [CITATION.cff](CITATION.cff) and [LICENSE](LICENSE). GitHub and Zenodo read the same citation file.

1. Push the source tree to the study’s GitHub repository, with `CITATION.cff` and `LICENSE` at the repository root.
2. Enable the repository in the depositing author’s Zenodo GitHub settings before creating a GitHub release. [Enable a repository](https://help.zenodo.org/docs/github/enable-repository/).
3. Publish the initial GitHub release using tag `v1.0.0`. Zenodo imports the release and assigns a DOI. [Archive a release](https://help.zenodo.org/docs/github/archive-software/github-upload/).
4. Open the Zenodo software record and verify its version, four creators, MIT license and files. Cite the repository URL and the DOI for this specific software version in the manuscript.
