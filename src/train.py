import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

batch_size = 32
num_epochs = 20


def asl_demo():
    # create image generator
    # this includes splitting the data for train/test as well as some data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,
        # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # don't randomly flip images
        vertical_flip=False,  # don't randomly flip images
        validation_split=0.2)  # set validation split

    train_generator = train_datagen.flow_from_directory(
        "./data/asl_alphabet_train",
        target_size=(28, 28),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        subset='training')  # set as training data

    validation_generator = train_datagen.flow_from_directory(
        "./data/asl_alphabet_train",
        target_size=(28, 28),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        subset='validation')  # set as validation data

    # Training The Model

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5,
                                                min_lr=0.00001)

    model = tf.keras.models.Sequential([
        Conv2D(75, (3, 3), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(75, (3, 3), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2), strides=2, padding='same'),
        Conv2D(50, (3, 3), strides=1, padding='same', activation='relu'),
        Dropout(0.2),
        BatchNormalization(),
        MaxPooling2D((2, 2), strides=2, padding='same'),
        Conv2D(25, (3, 3), strides=1, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2), strides=2, padding='same'),
        Flatten(),
        Dense(units=512, activation='relu'),
        Dropout(0.3),
        Dense(units=29, activation='softmax'),
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=num_epochs,
        callbacks=[learning_rate_reduction])

    model.save('asl_model')

    eval_history = model.evaluate_generator(validation_generator,
                                            steps=validation_generator.samples // batch_size)

    print(eval_history)

    # Analysis after Model Training
    epochs = [i for i in range(20)]
    fig, ax = plt.subplots(1, 2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    fig.set_size_inches(16, 9)

    ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
    ax[0].plot(epochs, val_acc, 'ro-', label='Testing Accuracy')
    ax[0].set_title('Training & Validation Accuracy')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(epochs, train_loss, 'g-o', label='Training Loss')
    ax[1].plot(epochs, val_loss, 'r-o', label='Testing Loss')
    ax[1].set_title('Testing Accuracy & Loss')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    plt.show()

    confusion_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=False,
        vertical_flip=False)

    confusion_generator = confusion_datagen.flow_from_directory(
        "./data/asl_alphabet_test",
        target_size=(28, 28),
        batch_size=batch_size,
        class_mode='binary')

    predictions = model.predict_generator(confusion_generator, steps=confusion_generator.samples // batch_size)
    for i in range(len(predictions)):
        if predictions[i] >= 9:
            predictions[i] += 1
    predictions[:5]


if __name__ == "__main__":
    asl_demo()
