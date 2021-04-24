import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import Xception


class MyModel:
    def __init__(self):
        self.loss = 'binary_crossentropy'
        self.optimizer = 'adam'
        self.metric = ['accuracy']
        self.callbacks = [EarlyStopping(patience=10, min_delta=0.01, verbose=1, monitor='val_accuracy', mode='max'),
                          ModelCheckpoint(filepath='Model/best_model.h5', monitor='val_accuracy', verbose=1,
                                          save_best_only=True)]

    # Method to load and fine tube the predefined model
    def load_and_finetune_model(self, img_size, nb_channels, num_trainable_layers):
        input_shape = Input(shape=(img_size, img_size, nb_channels))
        pretrained = Xception(include_top=False, weights='imagenet', input_tensor=input_shape)
        output = pretrained.layers[-1].output
        output = Flatten()(output)
        pretrained = Model(pretrained.input, output)
        # print(pretrained.summary())

        last_layer = pretrained.layers[-1].output
        x = Dropout(0.5, name='dp1')(last_layer)
        x = Dense(2048, activation='relu', name='fc1')(x)
        output = Dense(1, activation='sigmoid', name='output')(x)
        model = Model(input_shape, output)
        for layer in model.layers[:-num_trainable_layers]:
            layer.trainable = False
        # print(model.summary())
        return model

    def train_model(self, my_model, train_generator, val_generator, epochs):
        my_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metric)
        history = my_model.fit(train_generator,
                               epochs=epochs,
                               validation_data=val_generator,
                               verbose=1,
                               callbacks=self.callbacks)
        return history

    def plot_history(self, history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def load_and_predict(self, model_path, test_generator):
        predictions = None
        best_model = load_model(model_path)
        if best_model:
            predictions = (best_model.predict(test_generator, verbose=1) > 0.5).astype("int64")
        else:
            print("Cannot find best model at path: Model/")
        return predictions

    def evaluate_model(self, predictions, labels):
        cm = confusion_matrix(labels, predictions.ravel())
        cr = classification_report(labels, predictions.ravel())
        return cm, cr