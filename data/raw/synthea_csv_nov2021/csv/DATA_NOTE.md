# Data: 200 synthetic patients

Synthetic data only (Synthea November 2021 CSV sample). No real patients.
200 living patients in 16 physician panels. IDs were relabelled: patients ptA to ptGR, visits enaaa onward, payers 101 onward.
  id_map.csv maps each new ID to the original Synthea ID. physician_panels.csv lists each patient's primary physician.
  Built with make_200_patient_subset.py, relabel_ids.py and check_subset.py.
  
