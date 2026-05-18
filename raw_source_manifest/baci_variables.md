# BACI source variables used

Expected CEPII-BACI annual CSV columns:

- `t`: year.
- `k`: product code at the 6-digit Harmonized System level.
- `i`: exporter code.
- `j`: importer code.
- `v`: trade value in thousands of current USD.
- `q`: quantity in tonnes, where available.

The BACI-VI-Bench pipeline filters positive bilateral flows, removes within-country flows, maps HS6 products to HS2-derived sectors, selects top exporters/importers/sectors, and constructs VI benchmark instances.
