This system automatically aligns histological series to a standard atlas by identification of cyto-architectural landmarks. The goal is to enable mapping of cell markers across brains.

Input Data
=========

Input data are sagittal series labeled with markers such as:
- cell type marker, e.g. Choline Acetyltransferase (ChAT)
- neuronal tracers, e.g. pseudo-rabies and delta-G rabies

Tissues must also be counter-stained for Nissl to demonstrate cytoarchitecture. Possible Nissl stains include:
- thionin (brightfield) 
- [Neurotrace Blue](https://www.thermofisher.com/order/catalog/product/N21479) (fluorescent)

Development and validation used ~20 brains, prepared by Partha Mitra's lab in Cold Spring Harbor Laboratory and David Kleinfeld's lab in UCSD. [Complete list of brains](https://docs.google.com/spreadsheets/d/1QHW_hoMVMcKMEqqkzFnrppu8XT92BPdIagpSqQMAJHA/edit?usp=sharing)

Atlas Data
==================



Intermediate Data
==================



Code
==========

[Github Repository](https://github.com/mistycheney/MouseBrainAtlas)

Most of the code is written in Python. IPython notebooks are used for development. Most essential functionalities are then converted to executable scripts.


Reference
==========
[The Active Atlas: Combining 3D Anatomical Models with Texture Detectors](https://arxiv.org/abs/1702.08606), Chen et al., MICCAI 2017
