import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

data = plt.imread("mnist_images.png")

print("data shape", data.shape)

nd = data

nd = nd.round(0)

nd = nd.astype("uint8")

for x in range(10):
    a = np.array(nd[x])
    a = np.reshape(a, (28, 28))
    a = a*255
    data = Image.fromarray(a)
    data.save(f'./images/example{x}.png')

open("mnist_images_uint8_bw_", "wb").write(nd.tobytes())