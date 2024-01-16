# dataset source: https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq

# import neccessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# import datasets
data = pd.read_csv("~/ML/GeneRNA-seq/data.csv")
labels = pd.read_csv("~/ML/GeneRNA-seq/labels.csv")



