import pandas as pd
"""
load data
"""

feature_dict = {i:label for i,label in zip(
                range(4),
                  ('sepal length in cm',
                  'sepal width in cm',
                  'petal length in cm',
                  'petal width in cm', ))}

df = pd.io.parsers.read_csv(
    filepath_or_buffer='ionosphere/ionosphere.data',
    header=None,
    sep=',',
    )

df.columns = range(34) + ['class label']
df.dropna(how="all", inplace=True) # to drop the empty line at file-end
df.tail()

X = df[range(34)].values
y = df['class label'].values

print X[1,:]
print y[1:20]