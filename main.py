import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Reshape, LSTM, MultiHeadAttention, LayerNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

dataset_dir = os.path.join(path, "Dataset_BUSI_with_GT")
benign_dir = os.path.join(dataset_dir, "benign")
malignant_dir = os.path.join(dataset_dir, "malignant")
normal_dir = os.path.join(dataset_dir, "normal")

print("Contents of benign directory:", len(os.listdir(benign_dir)))
print("Contents of malignant directory:", len(os.listdir(malignant_dir)))
print("Contents of normal directory:", len(os.listdir(normal_dir)))

img_height, img_width = 224, 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

def cnn_block(input_tensor):
    x = Conv2D(32, (3, 3), activation='relu')(input_tensor)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    return x

def vit_block(input_tensor, num_heads=8, ff_dim=512):
    x = Reshape((-1, input_tensor.shape[-1]))(input_tensor)
    x = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dense(ff_dim, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(input_tensor.shape[-1])(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

def lstm_block(input_tensor):
    x = Reshape((-1, input_tensor.shape[-1]))(input_tensor)
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.5)(x)
    x = LSTM(64)(x)
    x = Dropout(0.5)(x)
    return x

input_tensor = Input(shape=(img_height, img_width, 3))
cnn_output = cnn_block(input_tensor)
vit_output = vit_block(cnn_output)
lstm_output = lstm_block(vit_output)

x = Dense(512, activation='relu')(lstm_output)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output_tensor = Dense(3, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=output_tensor)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

epochs = 30
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)

loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

model.save('advanced_breast_cancer_classification_model.h5')

loaded_model = load_model('advanced_breast_cancer_classification_model.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = loaded_model.predict(img_array)
    class_names = ['benign', 'malignant', 'normal']
    return class_names[np.argmax(prediction)]
