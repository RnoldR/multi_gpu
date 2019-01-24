#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 19:28:08 2019

@author: arnold
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv', index_col=0)

print(df)

#df_grouped = df.groupedby('GPU\'s')

# My Code to Plot in Three Panels
df_subset0 = df[df['GPU\'s'] == 1.0]
df_subset1 = df[df['GPU\'s'] == 2.0]

print(df_subset0)
print(df_subset1)

barwidth = 0.25
x = [str(bs) for bs in df_subset0['Batch size']]
y01 = df_subset0['Acc']
y02 = df_subset0['Val. Acc']
y03 = df_subset0['Time']
y11 = df_subset1['Acc']
y12 = df_subset1['Val. Acc']
y13 = df_subset1['Time']
# {maybe insert a line here to summarize df_subset somehow interesting?}

r1 = np.arange(len(y01))
r2 = [x + barwidth for x in r1]
r3 = [x + barwidth for x in r2]
r4 = [x + barwidth for x in r3]

fig, ax1 = plt.subplots()
ax1 = plt.axes()
ax1.bar(r1, y01, color='g', width=barwidth, edgecolor='white', label='1 GPU')
ax1.bar(r2, y11, color='b', width=barwidth, edgecolor='white', label='2 GPU\'s')
#plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='var3')
ax1.set_xlabel('Batch size', fontweight='bold')
ax1.set_ylim(0, 1)
plt.xticks([r + barwidth for r in range(len(y01))], x)

ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy for 1 and 2 GPU\'s after 15 epochs')
ax1.legend()
plt.savefig('results-accuracy.png')
plt.show()

ax2 = plt.axes()
ax2.bar(r1, y02, color='g', width=barwidth, edgecolor='white', label='1 GPU')
ax2.bar(r2, y12, color='b', width=barwidth, edgecolor='white', label='2 GPU\'s')
ax2.set_xlabel('Batch size', fontweight='bold')
plt.xticks([r + barwidth for r in range(len(y01))], x)

ax2.set_ylabel('Validation accuracy')
ax2.set_title('Validation accuracy for 1 and 2 GPU\'s after 15 epochs')
ax2.set_ylim(0, 1)
ax2.legend()
plt.savefig('results-time.png')
plt.show()

ax3 = plt.axes()
ax3.bar(r1, y03, color='g', width=barwidth, edgecolor='white', label='1 GPU')
ax3.bar(r2, y13, color='b', width=barwidth, edgecolor='white', label='2 GPU\'s')
ax3.set_xlabel('Batch size', fontweight='bold')
plt.xticks([r + barwidth for r in range(len(y01))], x)

ax3.set_ylabel('Time (s)')
ax3.set_title('Time in seconds for 1 and 2 GPU\'s after 15 epochs')
ax3.legend()
plt.savefig('results-val-accuracy.png')
plt.show()
