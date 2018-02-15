from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# rescale [0,255] --> [0,1]
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# one-hot encode the labels
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Define Model
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# Summarize Model
model.summary()

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# Train the model
checkpointer = ModelCheckpoint(filepath="mnist.model.best.hdf5", verbose=1, save_best_only=True)
hist = model.fit(X_train, y_train, batch_size=128, epochs=10, 
            validation_split=0.2, callbacks=[checkpointer],
            verbose=1, shuffle=True)
model.load_weights("mnist.model.best.hdf5")

score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]
print('Test Accuracy; %.4f%%' % accuracy)