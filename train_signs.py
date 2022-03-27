import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


data = []
labels = []
classes = 43
cur_path = os.getcwd()

for a_class in range(classes):
    path = os.path.join(cur_path, 'data1', 'train', str(a_class))
    images = os.listdir(path)

    print(a_class)

    for a in images:
        image = cv2.imread(path + '\\' + a)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (30, 30))
        data.append(image)
        labels.append(a_class)


data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=2006)

y_train = tf.compat.v1.keras.utils.to_categorical(y_train, 43)
y_test = tf.compat.v1.keras.utils.to_categorical(y_test, 43)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(rate=0.25))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(rate=0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(43, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=64, epochs=15, validation_data=(X_test, y_test))
model.save('models/sign_recognition.h5')

hist_df = pd.DataFrame(history.history)

with open('models/model_history.json', mode='w') as f:
    hist_df.to_json(f)
