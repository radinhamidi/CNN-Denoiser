'''

Radin Hamidi Rad
500979422

'''

import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.cluster import KMeans
from keras.models import model_from_json
# from keras import backend as kb

# print(kb.tensorflow_backend._get_available_gpus())
seed = 7
np.random.seed(seed)

# load data
data = np.load('./dataset/unlabeled_x_50.npy')
# data = np.load('./dataset/train_x_50.npy')
labels = np.loadtxt('./dataset/class_names.txt', dtype=np.str)
print('Data loaded.')

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

encoder_json_file = open('{}.json'.format(encoder_model_name), 'r')
loaded_encoder_json = encoder_json_file.read()
encoder_json_file.close()
encoder = model_from_json(loaded_encoder_json)

# load weights into new model
encoder.load_weights("{}.h5".format(encoder_weight_name))
print("Loaded: {} model from disk".format(encoder_model_name.replace('./models\\', '')))

# compile models
encoder.compile(optimizer='adadelta', loss='mean_squared_error')

# featurize images
data = data.astype('float32') / 255.
data_encoded = encoder.predict(data)
print('Images converted to features via encoder.')

# KMeans
k = 10
load_kmeans_q = input('Load precomputed kmeans? (y/n)')
if load_kmeans_q == 'y':
    kmeans_labels = np.load('./models/kmeans_labels.npy')
    kmeans_centroids = np.load('./models/kmeans_centroids.npy')
else:
    kmeans = KMeans(n_clusters=k, random_state=seed).fit(data_encoded)
    kmeans_labels = kmeans.labels_
    kmeans_centroids = kmeans.cluster_centers_
    np.save('./models/kmeans_labels.npy', kmeans_labels)
    np.save('./models/kmeans_centroids.npy', kmeans_centroids)

ids = []
clusters = []
for i in range(k):
    ids.append(np.argwhere(kmeans_labels == i))
for j in ids:
    clusters.append(data[j])

n = 10  # how many digits we will display
plt.figure(figsize=(100, 4))
plt.title('KMeans Clusters samples')
for i in range(k):
    for j in range(n):
        # display sample j of a specified cluster i
        ax = plt.subplot(k, n, i*k + j + 1)
        plt.imshow(clusters[i][j].reshape(50, 50, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.xlabel('Samples')
plt.ylabel('Clusters')
plt.show()

kmeans_fig_save_q = input('Save the figure? (y/n)')
if kmeans_fig_save_q == 'y':
    plt.savefig('./figures/kmeans_{}.png'.format(encoder_model_name.replace('./models\\', '')))
    print('*'*30)
    print('KMeans samples per cluster Saved.')
