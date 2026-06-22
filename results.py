import os 
from pathlib import Path 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image, display

path=Path(__file__).resolve().parent
img_dir=path/'runs/detect/train-2'

for d in os.listdir(img_dir):
    if d.endswith(('.jpg','.png')):
        file_path=os.path.join(img_dir,d)
        img=mpimg.imread(file_path)
        display(Image(filename=file_path))
        plt.imshow(img)
        plt.show()