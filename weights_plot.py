'''

Radin Hamidi Rad
500979422

'''

import numpy as np
import matplotlib.pyplot as plt
import glob
import time
from keras.models import model_from_json
from keras.layers import Input, Dense
from keras.models import Model
from keras.losses import mean_squared_error

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

weights = []
first_time = True
for l in encoder.layers:
    if first_time:
        first_time = False
    else:
        w = l.get_weights()
        w = np.asarray(w)
        w = w[0]
        weights.append(w)
        print(w.shape)
        plt.imshow(w)
plt.show()
