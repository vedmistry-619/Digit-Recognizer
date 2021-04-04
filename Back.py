import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

(X_train,y_train),(X_test,y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)
input_shape = (28,28,1)

y_train = tensorflow.keras.utils.to_categorical(y_train,num_classes=None)
y_test = tensorflow.keras.utils.to_categorical(y_test,num_classes=None)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

batch_size = 128
num_classes = 10
epochs = 15

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape = input_shape))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))

model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,optimizer=tensorflow.keras.optimizers.Adadelta(),metrics=['accuracy'])

hist = model.fit(X_train,y_train,batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test,y_test))
model.save('mnist.h5')
