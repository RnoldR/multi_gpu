#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 19:28:08 2019

@author: arnold
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ok.csv', index_col=0)

print(df)

#df_grouped = df.groupedby('GPU\'s')

# My Code to Plot in Three Panels
df_subset0 = df[df['GPU\'s'] == 1.0]
df_subset1 = df[df['GPU\'s'] == 2.0]

print(df_subset0)
print(df_subset1)

x = df_subset0['Batch size']
y01 = df_subset0['Acc']
y02 = df_subset0['Val. Acc']
y03 = df_subset0['Time']
y11 = df_subset1['Acc']
y12 = df_subset1['Val. Acc']
y13 = df_subset1['Time']
# {maybe insert a line here to summarize df_subset somehow interesting?}

fig, ax1 = plt.subplots()
ax1 = plt.axes()
#ax1.plot(x, y01, linestyle=':', color='g', label='Accuracy, 1 GPU')
ax1.plot(x, y02, linestyle='-', color='g', label='Val Acc, 1 GPU')
#ax1.plot(x, y11, linestyle=':', color='b', label='Accuracy, 2 GPU\'s')
ax1.plot(x, y12, linestyle='-', color='b', label='Val Acc, 2 GPU\'s')
ax1.set_ylim(0, 1)
ax1.set_xlabel('Batch size')
ax1.set_ylabel('Accuracy')
ax1.set_title('Results for 1 and 2 GPU\'s after 5 epochs')
legend = ax1.legend(loc='upper left')


ax2 = ax1.twinx()
ax2.plot(x, y03, color='y', label='Time, 1 GPU')
ax2.plot(x, y13, color='r', label='Time, 2 GPU\'s')
ax2.set_ylabel('Time (s)')
legend = ax2.legend(loc='upper right')

plt.savefig('results.png')

plt.show()
