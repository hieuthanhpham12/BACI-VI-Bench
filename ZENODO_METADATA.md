# Draft Zenodo metadata

## Upload type
Dataset

## Title
BACI-VI-Bench: Benchmark instances for variational inequality and MARL trade-network equilibrium

## Creators
- Pham Thanh Hieu, Posts and Telecommunications Institute of Technology, Vietnam, ORCID: https://orcid.org/0009-0002-2786-2834
- Nguyen Kieu Linh, Posts and Telecommunications Institute of Technology, Vietnam, ORCID: https://orcid.org/0009-0005-6854-1350

## Description
BACI-VI-Bench is a BACI-derived benchmark dataset for variational inequality (VI) and multi-agent reinforcement learning (MARL) experiments on international trade-network equilibrium problems. The source data are CEPII-BACI HS17 V202601 annual bilateral trade-flow files. The benchmark construction pipeline filters positive bilateral flows, maps HS6 products into HS2-based commodity sectors, selects leading exporters and importers, aggregates trade values and quantities into structured tensors, normalizes observed flows, calibrates finite-dimensional trade-network VI operators, and stores solver-ready benchmark instances.

The repository contains metadata files, source-data manifests, Python code, benchmark outputs, and publication-ready figures. After running the provided builder on local CEPII-BACI raw files, the repository contains compressed `.npz` VI instances for years 2017--2024, with default dimension d=500 under m=10 exporters, n=10 importers, K=5 sectors, and L=1 route.

The dataset is intended for benchmarking projection, extragradient, inertial, self-adaptive VI solvers, fixed-point methods, and MARL-based equilibrium algorithms. It does not replace or redistribute the original CEPII-BACI database; rather, it provides a processed benchmark layer derived from BACI records.

## Keywords
International trade; variational inequality; multi-agent reinforcement learning; benchmark dataset; CEPII-BACI; trade network equilibrium; commodity flow; equilibrium computation

## License
Suggested for code: MIT License.  
Suggested for processed data: open data license compatible with CEPII-BACI terms, with attribution to CEPII-BACI and Gaulier & Zignago (2010). Confirm final license choice before upload.

## Related identifiers
- CEPII-BACI source webpage: https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/baci_webpage.html
- Zenodo record DOI (all versions): https://doi.org/10.5281/zenodo.20263232
- Zenodo record DOI (v0.3): https://doi.org/10.5281/zenodo.20266673
- Zenodo record URL: https://zenodo.org/records/20266673
- Data article DOI: [TO BE COMPLETED after Data in Brief publication]
- GitHub repository: https://github.com/hieuthanhpham12/BACI-VI-Bench

## Version
v0.3

## Language
English

