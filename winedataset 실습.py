from sklearn import datasets
import numpy as np
import pandas as pd

wine = datasets.load_wine()

features = wine['data']
feature_names = wine['feature_names']
labels = wine['target']

df = pd.DataFrame(wine['data'], columns=wine['feature_names'])
df['target'] = wine['target']
df.head()