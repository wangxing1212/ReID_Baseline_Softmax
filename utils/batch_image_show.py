import numpy as np
import matplotlib
import matplotlib.pyplot as plt
def batch_image_show(images_batch, classes, n):
    images = images_batch[n].numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = std * images + mean
    images = np.clip(images, 0, 1)
    plt.imshow(images)
    plt.title('Person ID is '+ str(classes[n]))
    plt.pause(0.001)