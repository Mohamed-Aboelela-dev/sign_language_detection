import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Loading data
data = pd.read_csv('data/sign_data.csv', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(63,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(8, activation='softmax')
])

# Preparing the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Evaluating the model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Saving the model
model.save('model.h5')
print("Model Saved!")