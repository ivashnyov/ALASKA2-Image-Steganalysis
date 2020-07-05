from glob import glob
import random
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = []

for label, kind in enumerate(['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']):
    for path in glob('data/Cover/*.jpg'):
        dataset.append({
            'kind': kind,
            'image_name': path.split('/')[-1],
            'label': label
        })

random.shuffle(dataset)
dataset = pd.DataFrame(dataset)

train_df, val_df = train_test_split(dataset, test_size=0.2, stratify=dataset["label"], random_state=69)

train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)