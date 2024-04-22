import numpy as np
import matplotlib.pyplot as plt


point_cloud = np.load('./dataset/point_clouds/pc-00080.npy')
bounding_boxes = np.load('./dataset/bounding_boxes/bb-00080.npy')

ax = plt.figure().add_subplot(111, projection='3d')

#ax.set_xlim([0, 25])
#ax.set_ylim([0, 25])
#ax.set_zlim([0, 25])

#for pc in point_cloud:
#    ax.scatter(xs=pc[0], ys=pc[1], zs=pc[2], color='blue', s=0.5)

for pedestrian in bounding_boxes:
    for bb in pedestrian:
        ax.scatter(xs=bb[0], ys=bb[1], zs=bb[2], color='red', s=0.5)
    break


plt.show()