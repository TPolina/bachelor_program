#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

tf.random.set_seed(42)


class1 = []
class2 = []

with open('ukr_words', 'r') as f:
    for token in f:
        class1.append(token[:-1].lower())

with open('anglicisms', 'r') as f:
    for token in f:
        class2.append(token[:-1].lower())

class1 = sorted(class1)
class1 = class1[37:]  # skip non-ukrainian words


train_xr = []
train_y = []

test_xr = []
test_y = []


def trtest(cl, l):
    for i, token in enumerate(cl):
        if i%10 == 0:
            test_xr.append(token)
            test_y.append(l)
        else:
            train_xr.append(token)
            train_y.append(l)


trtest(class1, 0)
trtest(class2, 1)

test_y = np.array(test_y)
train_y = np.array(train_y)


mx_len = max(len(w) for w in class1 + class2)

letters = set(l for word in class1 for l in word)
letters = {l:i for i, l in enumerate(letters)}

n_in = len(letters) + 1


def encode(data):
    out = np.ones((len(data), mx_len)) * n_in
    for i, w in enumerate(data):
        vec = np.zeros((len(w), n_in))
        for j, l in enumerate(w):
            if l in letters:
                out[i, j] = letters[l]
            else:
                out[i, j] = n_in-1
    return out


def decode(seq):
    dletters = {letters[k]: k for k in letters}
    return "".join(dletters[l] for l in seq if l < n_in-1)


train_x = encode(train_xr)
test_x = encode(test_xr)


BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_x, train_y = shuffle(train_x, train_y, random_state=42)


################################################################

# Define neural network
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=n_in+1,
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy', 'Recall', 'AUC'])

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(BATCH_SIZE)


history = model.fit(train_dataset, epochs=200,
                    validation_data=test_dataset,
                    validation_steps=20)


pred = model.predict(test_x).flatten()
res_ukr = pred < 0.5
res_en = pred >= 0.5


def get_metrics(data, y):
    tp = ((data == y) & (y == 1)).sum()
    tn = ((data == y) & (y == 0)).sum()
    fp = ((data != y) & (y == 1)).sum()
    fn = ((data != y) & (y == 0)).sum()
    total = len(y)
    accuracy = (tp+tn) / total
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * precision * recall / (precision + recall)
    return accuracy, recall, f_score


print('Anglicisms metrics: ', get_metrics(res_en, test_y))
print('Ukrainian words metrics: ', get_metrics(res_ukr, 1 - test_y))
