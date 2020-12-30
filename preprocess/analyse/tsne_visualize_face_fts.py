'''
对比 MEmoBert 出来的特征 和 本身 Denseface 的特征的效果
'''
from code.downstream.run_baseline import main
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

def load_data(feature_path, label_path):
    latent = np.load(feature_path)
    label = np.load(label_path)
    return latent, label

def plot(latent, label, save_path):
    '''
    latent: shape= (N, D)
    label: shape= (N,)
    '''
    print("Finish loading data: ")
    print("Latent:", latent.shape)
    print("Label:", label.shape)
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    y = ts.fit_transform(latent)
    # init graph
    plt.figure(figsize=(5, 5))
    # define colors
    colors_label = ['red', 'blue', 'green', 'gold']
    cat0_data = y[np.where(label==0)]
    cat1_data = y[np.where(label==1)]
    cat2_data = y[np.where(label==2)]
    cat3_data = y[np.where(label==3)]
    plt.scatter(cat0_data[:, 0], cat0_data[:, 1], s=1, c=colors_label[0], cmap=plt.cm.Spectral)
    plt.scatter(cat1_data[:, 0], cat1_data[:, 1], s=1, c=colors_label[1], cmap=plt.cm.Spectral)
    plt.scatter(cat2_data[:, 0], cat2_data[:, 1], s=1, c=colors_label[2], cmap=plt.cm.Spectral)
    plt.scatter(cat3_data[:, 0], cat3_data[:, 1], s=1, c=colors_label[3], cmap=plt.cm.Spectral)
    plt.legend(['Ang', 'Hap', 'Neu', 'Sad'], fontsize=12, loc='lower right', markerscale=4)
    plt.savefig(save_path)

if __name__ == '__main__':
    save_path = './tsne_emobert_cls.jpg'
    feature_path = ''
    label_path = ''
    latent, label = load_data(feature_path, label_path)
    main(latent, label, save_path)
    