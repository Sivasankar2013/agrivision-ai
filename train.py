import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import json

train_dir = "dataset/Training"
test_dir = "dataset/Test"

IMG_SIZE = (224,224)
BATCH_SIZE = 32
EPOCHS = 30

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

num_classes = len(train_generator.class_indices)

with open("class_indices.json","w") as f:
    json.dump(train_generator.class_indices,f)

model = models.Sequential([
layers.Conv2D(32,(3,3),activation="relu",input_shape=(224,224,3)),
layers.MaxPooling2D(2,2),

layers.Conv2D(64,(3,3),activation="relu"),
layers.MaxPooling2D(2,2),

layers.Conv2D(128,(3,3),activation="relu"),
layers.MaxPooling2D(2,2),

layers.Conv2D(256,(3,3),activation="relu"),
layers.MaxPooling2D(2,2),

layers.Flatten(),
layers.Dense(512,activation="relu"),
layers.Dropout(0.5),

layers.Dense(num_classes,activation="softmax")
])

model.compile(
optimizer="adam",
loss="categorical_crossentropy",
metrics=["accuracy"]
)

model.fit(
train_generator,
epochs=EPOCHS,
validation_data=test_generator
)

model.save("agrivision_model.keras")

print("Training completed")
