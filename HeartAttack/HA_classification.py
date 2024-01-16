# A guide to classification methods in heart attack analysis
import os
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
pd.set_option("display.max_rows", None)
from sklearn import preprocessing
import matplotlib
matplotlib.style.use('ggplot')
from sklearn.preprocessing import LabelEncoder

# Read dataset
ht = pd.read_csv(~/ML/HeartAttack/heart.csv)
ht.head()
