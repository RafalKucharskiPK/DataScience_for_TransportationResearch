#!/usr/bin/env python
# -*- coding: utf8 -*-

import pandas as pd
import matplotlib.pyplot as plt

from simpledbf import Dbf5

dbf = Dbf5("./data/EE1160.dbf", codec ='cp1250')

df = dbf.to_dataframe()
df = df.ffill()

print(df.columns)

print(df.head())

print(df.describe())

df[df.columns[2]].plot()

plt.show()




