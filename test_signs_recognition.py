import os
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

cwd = os.getcwd()

try:
    model = keras.models.load_model('models\\sign_recognition.h5')
except OSError:
    raise OSError('Please run train_signs.py or upload sign_recognition.h5 to models folder')


if not os.path.isfile(os.path.join(cwd, 'models', 'model_history.json')):
    raise OSError('Please run train_signs.py or upload model_history.json to models folder')

df = pd.read_json('models\\model_history.json')


plt.figure(0)
plt.plot(df['accuracy'], label='training accuracy')
plt.plot(df['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('stats\\accuracy.jpg')

plt.figure(1)
plt.plot(df['loss'], label='training loss')
plt.plot(df['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('stats\\loss.jpg')


y_test = pd.read_csv(os.path.join(cwd, 'data1', 'Test.csv'))
y_test['Path'] = y_test['Path'].apply(lambda x: x.split('/')[1])

y_values = y_test["ClassId"].values
paths = y_test["Path"].values

data = []
for path in paths:
    image = cv2.imread(os.path.join(cwd, 'data1', 'Test', path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (30, 30))
    data.append(image)

X_test = np.array(data)
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

with open('accuracy.txt', 'w') as f:
    f.write(str(accuracy_score(y_values, y_pred)))

labels = {
    'Speed limit (20km/h)',
    'Speed limit (30km/h)',
    'Speed limit (50km/h)',
    'Speed limit (60km/h)',
    'Speed limit (70km/h)',
    'Speed limit (80km/h)',
    'End of speed limit (80km/h)',
    'Speed limit (100km/h)',
    'Speed limit (120km/h)',
    'No passing',
    'No passing veh over 3.5 tons',
    'Right-of-way at intersection',
    'Priority road',
    'Yield',
    'Stop',
    'No vehicles',
    'Veh > 3.5 tons prohibited',
    'No entry',
    'General caution',
    'Dangerous curve left',
    'Dangerous curve right',
    'Double curve',
    'Bumpy road',
    'Slippery road',
    'Road narrows on the right',
    'Road work',
    'Traffic signals',
    'Pedestrians',
    'Children crossing',
    'Bicycles crossing',
    'Beware of ice/snow',
    'Wild animals crossing',
    'End speed + passing limits',
    'Turn right ahead',
    'Turn left ahead',
    'Ahead only',
    'Go straight or right',
    'Go straight or left',
    'Keep right',
    'Keep left',
    'Roundabout mandatory',
    'End of no passing',
    'End no passing veh > 3.5 tons'
}

fig, ax = plt.subplots(figsize=(15, 15))
cm = confusion_matrix(y_values, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
display.plot(xticks_rotation='vertical', ax=ax)
plt.savefig('stats\\confusion_matrix.jpg')

