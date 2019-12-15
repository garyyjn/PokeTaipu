import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import os
import glob

output_dim = (225, 225)

def jpgtoarray(path):
    original_image = Image.open(path)
    resizedImage = ImageOps.fit(original_image, output_dim, Image.ANTIALIAS)
    resizedImage.save("exmapleResize.jpg")
    I = np.asarray(resizedImage)
    return I

pokedex_type = pd.read_csv('pokedex_type_onehot.csv', delimiter=' ')
pokedex_type = pokedex_type.to_numpy()

def typeLabel(pokemon_name):
    for i in range(pokedex_type.shape[0]):
        if pokemon_name == pokedex_type[i, 1]:
            one_hot_label = pokedex_type[i, 2:]
            return one_hot_label

data_directory = 'dataset'
save_directory = '15ImagePerPokemon'

sample_per_pokemon = 15
sample_num = sample_per_pokemon * 151 # 10 per pokemon

photo_array = np.ones((sample_num, 3, output_dim[0], output_dim[1]), dtype=np.uint8)
type_array = np.zeros((sample_num, 18),dtype=np.int)
pokemon_name_array = np.ones((sample_num),dtype= '<S3')

counter = 0

for pokemon_name in os.listdir(data_directory):
    pokemonFolder = os.path.join(data_directory, pokemon_name)
    pokemon_type = typeLabel(pokemon_name)
    current_pokemon_counter = 0
    for filename in os.listdir(pokemonFolder):
        if current_pokemon_counter == sample_per_pokemon:
            break
        if not filename.endswith('jpg'):
            continue
        print(pokemon_name, filename)
        imagepath = os.path.join(pokemonFolder, filename)
        image_array = jpgtoarray(imagepath)
        image_array = image_array.T
        if image_array.shape == (3, output_dim[0], output_dim[1]):
            good = 1
        elif image_array.shape ==  output_dim:
            new_image_array = np.ones((3,output_dim[0], output_dim[1]))
            new_image_array[0,:,:] = image_array.copy()
            new_image_array[1,:,:] = image_array.copy()
            new_image_array[2,:,:] = image_array.copy()
            image_array = new_image_array
        else:
            print("Weird dims from image! ".format((image_array.shape)))

        print(image_array.shape)
        photo_array[counter,:,:,:] = image_array
        type_array[counter,:] = pokemon_type
        pokemon_name_array[counter] = pokemon_name
        print(image_array.shape)
        print(image_array.dtype)
        print(photo_array[counter].shape)
        current_pokemon_counter += 1
        counter += 1

np.save(os.path.join(save_directory,'pokemonImage.npy'), photo_array)
np.save(os.path.join(save_directory,'pokemonType.npy'),type_array)
np.save(os.path.join(save_directory,'pokemonName.npy'),pokemon_name_array)





