import wfdb
import matplotlib.pyplot as plt

# Replace '100' with the record name you want to view
record = wfdb.rdrecord('100', pn_dir='mitdb/')

# Read the annotations for the same record
annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb/')

# Print some of the record's metadata
print(record._dict_)

# Plot the first signal
wfdb.plot_wfdb(record=record, annotation=annotation, title='Record 100 from MIT-BIH Arrhythmia Database')

plt.show()