# plot_csp.py
import pandas as pd
import matplotlib.pyplot as plt
import sys
csv_file = sys.argv[1] if len(sys.argv)>1 else "csp_metrics.csv"
df = pd.read_csv(csv_file)
df['time_ms'] = df['time_ms'].astype(float)
plt.figure(figsize=(6,4))
plt.bar(df['mode'], df['time_ms'])
plt.title("CSP runtime by Method")
plt.ylabel("Time (ms)")
plt.savefig("csp_time.png", bbox_inches='tight')
plt.show()
