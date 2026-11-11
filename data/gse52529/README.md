# GSE52529 human myoblast differentiation

This directory contains the processed single-cell expression matrix and sample metadata for the human skeletal muscle myoblast differentiation time course from Trapnell et al. (2014), GEO series [GSE52529](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE52529).

- `GSE52529_fpkm_matrix.txt.gz` is the GEO per-cell FPKM matrix. Rows are Ensembl gene IDs and columns are single-cell samples. FPKM values are already normalized; the notebook loader applies `log1p`, selects highly variable genes, scales, and computes PCA.
- `GSE52529_cell_metadata.csv` maps matrix columns to GEO sample accessions and records collection time after serum switch and GEO quality flags.
- The comparison loader keeps samples flagged as non-debris, non-control, and exactly one cell in the well (306 of 372 matrix columns). It retains the observed 0, 24, 48, and 72 hour collection times as stage labels.

The dataset was used in the original Monocle pseudotime study. See Trapnell et al., “The dynamics and regulators of cell fate decisions are revealed by pseudotemporal ordering of single cells,” *Nature Biotechnology* (2014), [doi:10.1038/nbt.2859](https://doi.org/10.1038/nbt.2859). Collection time is an observed experimental label; any computed pseudotime remains an inference from expression data.
