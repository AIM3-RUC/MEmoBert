import os
import os.path as osp
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt

def load_data():
    root = '/data7/MEmoBert/emobert/exp/face_model/densenet100_adam0.001_0.0/log'
    ft = np.load(osp.join(root, 'val_features.npy'))
    label = np.load(osp.join(root, 'val_label.npy'))
    return ft, label

def plot():
    save_path = 'pics/tsne_val.png'
    colors = ['gray', 'red', 'blue', 'deepskyblue', 'green', 'springgreen', 'brown', 'darkorange']
    ft, label = load_data()
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    y = ts.fit_transform(ft)
    for category in range(8):
        cat_data = y[np.where(label==category)]
        plt.scatter(cat_data[:, 0], cat_data[:, 1], s=1, c=colors[category], cmap=plt.cm.Spectral)
    plt.legend(['neu', 'hap', 'sur', 'sad', 'ang', 'dis', 'fea', 'con'])
    plt.savefig(save_path)

if __name__ == '__main__':
    ft, label = load_data()
    print(ft.shape, label.shape)
    plot()