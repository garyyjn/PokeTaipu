import numpy as np
import os
from PIL import Image

data_directory = 'DataCleaning\\saved_data'

def load_image_name_type(data_directory = data_directory):
    image = np.load(os.path.join(data_directory, 'pokemonImage.npy'))
    name = np.load(os.path.join(data_directory, 'pokemonName.npy'))
    type = np.load(os.path.join(data_directory, 'pokemonType.npy'))
    name = name.astype('<S3')
    return image, name, type

if __name__ == '__main__':
    image, name, type = load_image_name_type()
    print(image[0].shape)
    print(name[0])
    print(type[0])

    '''
    print(image.shape)
    print(name.shape)
    print(type.shape)
    for i, n, t, count in zip(image,name,type, range(name.size)):
        if count == 200:
            break
        print(n)
        print(t)
        print(i.shape)
    '''