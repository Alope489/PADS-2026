#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.ticker import MaxNLocator
import argparse

link_bw = 25

parser = argparse.ArgumentParser(description='Plot CODES terminal(flow) outputs for ModSim25 experiments.')
parser.add_argument('terminal_file', help='The path and name of the terminal output from CODES.')
parser.add_argument('--output', '-o', help='The path and name of the output file', default='multiflow.png')

# Parse arguments
args = parser.parse_args()
terminalfile = args.terminal_file
outfile = 'multiflow.out'
if args.output:
    outfile = args.output

# Read Terminal Data
tdf = pd.read_table(terminalfile, sep=' ')

# Drop unused columns and sort
cols_to_drop = [col for col in tdf.columns if col.startswith('qos-')]
cols_to_drop = [col for col in tdf.columns if col.startswith('vc')]
cols_to_drop.append('Unnamed: 0')
cols_to_drop.append('qos-level')
cols_to_drop.append('downstream-credits')
tdf = tdf.drop(cols_to_drop, axis=1)

# Remove empty records with no bandwidth
tdf = tdf[tdf['bw-consumed'] != 0]

# Convert values
tdf['time-stamp'] = tdf['time-stamp'].apply(lambda x: x / 1000) # Convert ns to us
tdf['bw-consumed'] = tdf['bw-consumed'].apply(lambda x: (x * link_bw)/100) # Convert % to bw

tdf = tdf.sort_values(['time-stamp','term-id'])
print(tdf['term-id'].unique())

# Setup flow-id:term-id mappings
flowmapping = {0: 14, 1: 15, 2: 1}


# Setup flows for figure
flows = {}

for flow in sorted(flowmapping.keys()):
    termid = flowmapping[flow]
    x = tdf[tdf['term-id'] == termid]['time-stamp']
    y = tdf[tdf['term-id'] == termid]['bw-consumed']
    if len(x) > 0:
        flows[flow] = {'x': x, 'y': y}


# Setup figure based on num flows
num_flows = len(flows.keys())
fig, axs = plt.subplots(1, num_flows)

# Plot flows
for i, flow in enumerate(sorted(flows.keys())):
    axs[i].set_title(f"Flow {flow}", fontsize=10)
    axs[i].plot(flows[flow]['x'], flows[flow]['y'], marker='.', fillstyle='none', color='y')

# Format figure
for ax in axs.flat:
    ax.set (xlabel='Time (us)', ylabel='Flow Injection Rate (GB/s)')
    ax.set_ylim(0, 35)
    #ax.legend()
    ax.label_outer()
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

fig.set_figheight(3.5)
fig.set_figwidth( 4 * num_flows)

plt.savefig(outfile, bbox_inches='tight')
