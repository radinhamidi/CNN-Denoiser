'''

Radin Hamidi Rad
500979422

'''

import numpy as np
import matplotlib.pyplot as plt
import glob
from keras.models import model_from_json

# from keras import backend as kb

# print(kb.tensorflow_backend._get_available_gpus())
seed = 7
np.random.seed(seed)

# load data
data = np.load('./dataset/unlabeled_x_50.npy')
# data = np.load('./dataset/train_x_50.npy')
centroids = np.load('./models/kmeans_centroids.npy')
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

# KNN
k = 11
clusters_labels = []
distances = []
for d_e in data_encoded:
    min_dis = np.inf
    c_id_min = None
    for c_id, c in enumerate(centroids):
        dis = np.linalg.norm(d_e - c)
        if dis < min_dis:
            min_dis = dis
            c_id_min = c_id
    clusters_labels.append(c_id_min)
    distances.append(min_dis)
clusters_labels = np.asarray(clusters_labels)
distances = np.asarray(distances)
print('Distances calculated.')

ids = []
clusters_distances = []
for i in range(k):
    ids.append(np.argwhere(clusters_labels == i))
for j in ids:
    sorted_dis = np.argpartition(distances[j].T, k)
    clusters_distances.append(j[sorted_dis][0])

clusters_distances = np.asarray(clusters_distances)
chosen_pics = []
for chosen in clusters_distances:
    chosen_pics.append(data[chosen])

num = list(input('Picture id:'))

n = 10  # how many digits we will display
plt.figure(figsize=(100, 4))
plt.title('KMeans Clusters samples')
for i in range(k):
    for j in range(n):
        # display sample j of a specified cluster i
        ax = plt.subplot(k, n, i * k + j + 1)
        plt.imshow(chosen_pics[i][j].reshape(50, 50, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.xlabel('Samples')
plt.ylabel('Clusters')

plt.figure()
plt.imshow(data[num].reshape(50,50,3))
plt.show()

kmeans_fig_save_q = input('Save the figure? (y/n)')
if kmeans_fig_save_q == 'y':
    plt.savefig('./figures/knn_{}.png'.format(encoder_model_name.replace('./models\\', '')))
    print('*' * 30)
    print('KNN samples per cluster Saved.')
