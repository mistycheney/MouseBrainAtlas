# Topics
- [Overview](Overview.md)
- [Data Organization](FileOrganization.md)
- [Environment Setup](InitialSetup.md)
- [Preprocessing](Preprocessing.md)
- [Registration](Registration.md)
- [Detection](Detection.md)
- [Visualization](Visualization.md)
- [Build Atlas](BuildAtlas.md)
- [Train Classifiers](TrainClassifiers.md)


# Software installation

`$ pip install activeatlas`

This will download the scripts and the package containing the reference anatomical model and the trained texture classifiers.

Edit `global_setting.py`. In particular, specify the following variables:
- `DATA_DIR`
- `REPO_DIR`
