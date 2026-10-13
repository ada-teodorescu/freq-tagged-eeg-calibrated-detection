# Raw-reconstruction provenance subset

These selected records were copied from the original raw-replication run `42ae2efcc15c`; no analysis was rerun for this export.

- `config_snapshot.toml` records the preprocessing branches and agreement thresholds.
- `product_comparison.json` and `tables/released_product_agreement.csv` record participant-product agreement and the mapping hypothesis.
- `tables/epoch_agreement.csv` compares reconstructed epochs with the public SET/FDT release.
- `paper_results.json` and `tables/raw_planned_comparisons.csv` record group effects and acoustic associations.

The `lowpass_no_ica` branch closely reproduced the public SET/FDT epochs used by the detector. This agreement does not resolve the discrepancy with archived MATLAB products or establish the undocumented original preprocessing history. The stored participant order is a hypothesis; optimal assignment is diagnostic only. This is a reporting subset, not a complete raw-data replication export.
