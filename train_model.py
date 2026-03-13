import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

def build_model(input_shape=(130,13,1)):

    model = models.Sequential()

    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((3,3)))

    model.add(layers.Conv2D(32,(3,3),activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((3,3)))

    model.add(layers.Conv2D(32,(2,2),activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(10,activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


model = build_model()

model.summary()