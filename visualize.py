import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
# Use matplotlib backend 'Agg' to write to file
# or 'TkAgg' to visualize from console
matplotlib.use('Agg')
# matplotlib.use('TkAgg')

def visualize(embed, labels):
    labelset = set(labels.tolist())
    
    # For visualization to file
    # Start
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)

    for label in labelset:
        indices = np.where(label == labels)
        ax.scatter(embed[indices, 0], embed[indices, 1], label=label, s=20)
    ax.legend()
    fig.savefig('embed.png', format='png', dpi=600, bbox_inches='tight')
    # End
    
    # For visualization from console
    # Start
    # ax_min = np.min(embed,0)
    # ax_max = np.max(embed,0)
    
    # plt.figure()
    # ax = plt.subplot(111)

    # for label in labelset:
    #     indices = np.where(label == labels)
    #     ax.scatter(embed[indices, 0], embed[indices, 1], label=label, s=20)
    # ax.legend()
    # plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
    # plt.title('Image Similarity')
    # plt.show()
    # End

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
label_test = mnist.test.labels

embed = np.fromfile('embed.txt', dtype=np.float32)
embed = embed.reshape([-1, 2])

visualize(embed, label_test)