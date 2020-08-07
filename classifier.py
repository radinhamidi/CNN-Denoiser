'''

Radin Hamidi Rad
500979422

'''

import glob
from util import *
from keras.models import model_from_json, Model
from keras.layers import Input, Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# from keras import backend as kb
# from keras import optimizers
# from keras.callbacks import TensorBoard

# print(kb.tensorflow_backend._get_available_gpus())
np.random.seed(7)

# load data
train_x = np.load('./dataset/train_x_50.npy')
test_x = np.load('./dataset/test_x_50.npy')
train_y = np.load('./dataset/train_y.npy')
test_y = np.load('./dataset/test_y.npy')
labels = np.loadtxt('./dataset/class_names.txt', dtype=np.str)

# transform labels
train_y = np.reshape(train_y, (-1, 1))
test_y = np.reshape(test_y, (-1, 1))
train_onehotencoder = OneHotEncoder()
train_onehotencoder.fit(train_y)
train_y = train_onehotencoder.transform(train_y).todense()
test_onehotencoder = OneHotEncoder()
test_onehotencoder.fit(test_y)
test_y = test_onehotencoder.transform(test_y).todense()

# load encoder model
model_names = []
for path in glob.glob('./models/encoder_*.json'):
    model_names.append(path)
print('Please enter you model number form list below:')
for i, path in enumerate(model_names):
    print('{}. {}'.format(i, path))
model_number = int(input('?'))
encoder_model_name = model_names[model_number].replace('.json', '')
encoder_weight_name = encoder_model_name.replace('models', 'weights')

json_file = open('{}.json'.format(encoder_model_name), 'r')
loaded_model_json = json_file.read()
json_file.close()
encoder = model_from_json(loaded_model_json)
# load weights into new model
encoder.load_weights("{}.h5".format(encoder_weight_name))
print("Loaded: {} model from disk".format(encoder_model_name.replace('./models\\', '')))

# featurize images
encoder.compile(optimizer='adadelta', loss='mean_squared_error')
train_x = train_x.astype('float32') / 255.
test_x = test_x.astype('float32') / 255.
train_encoded = encoder.predict(train_x)
test_encoded = encoder.predict(test_x)
print('Images converted to features via encoder.')

x_train, x_val, y_train, y_val = train_test_split(train_encoded, train_y, test_size=0.15)
x_test = test_encoded
y_test = test_y
# x_train = x_train.astype('float32') / 255.
# x_val = x_val.astype('float32') / 255.
# x_test = test_encoded.astype('float32') / 255.

# create model
hidden_dim = 1000  # hidden layer neurons
input_img_features = Input(shape=(x_train.shape[1],))
hidden_layer = Dense(hidden_dim, activation='sigmoid', name='Hidden_Layer',
                     kernel_initializer='random_uniform', bias_initializer='ones')(input_img_features)
classes = Dense(y_train.shape[1], activation='sigmoid', name='Output_Layer',
                kernel_initializer='random_uniform', bias_initializer='ones')(hidden_layer)
classifier = Model(input_img_features, classes, name='Classifier')
# Compile model
classifier.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# Fit the model
print('Model training phase started.')
batch_sizes = [1000, 200, 50]
epochs      = [300, 30, 4]
for epoch, batch_size in zip(epochs, batch_sizes):
    print('*'*30)
    print('*'*30)
    print('Starting batch size {} for {} times.'.format(batch_size, epoch))
    history = classifier.fit(x_train, y_train,
                             epochs=epoch,
                             batch_size=batch_size,
                             shuffle=True,
                             verbose=2,
                             validation_data=(x_val, y_val))

# evaluate the model
scores = classifier.evaluate(x_test, y_test, verbose=0)
accuracy = scores[1] * 100
print("%s: %.2f%%" % (classifier.metrics_names[1], accuracy))

# confusion matrix
y_predict = classifier.predict(x_test)
y_predict = [labels[np.argmax(i)] for i in y_predict]
y_true = [labels[np.argmax(i)] for i in y_test]
cm = confusion_matrix(y_true, y_predict, labels)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(0)
plot_confusion_matrix(cm, class_names=labels, title='Confusion matrix - Test Accuracy = {0:.2f}%'.format(accuracy), normalize=True)
print('*' * 30)
print('Accuracy : {0:.3f}'.format(accuracy))

save_model_q = input('Save the models? (y/n)')
if save_model_q == 'y':
    classifier_model_json = classifier.to_json()
    classifier_name = input('Please enter classifier model name:')
    with open('./models/{}.json'.format(classifier_name), "w") as json_file:
        json_file.write(classifier_model_json)
    encoder.save_weights("./weights/{}.h5".format(classifier_name))
    print('Model Saved.')
    plt.savefig('./figures/{}.png'.format(classifier_name))
    print('Confusion Matrix Saved.')

# summarize history for accuracy
plt.figure(1)
plt.grid()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
# plt.show()
# summarize history for loss
plt.figure(2)
plt.grid()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()
