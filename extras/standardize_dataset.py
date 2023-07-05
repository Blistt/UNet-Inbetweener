import os
import cv2
import numpy as np

"""Read Dataset and get most common shape"""
def get_dataset_shapes(data_dir):
    triplet_paths = [os.path.join(data_dir, p) for p in os.listdir(data_dir)]
    shapes = []
    for triplet_path in triplet_paths:
        for img_path in os.listdir(triplet_path):
            img = cv2.imread(os.path.join(triplet_path, img_path), cv2.IMREAD_GRAYSCALE)
            shapes.append(img.shape)

    # Gets list of unique shapes and their counts
    unique_shapes, counts = np.unique(shapes, axis=0, return_counts=True)
    standard_shape = unique_shapes[np.argmax(counts)] # Most common shape

    # Prints out the shapes and their counts
    print('Dataset has the following shapes')
    for i, shape in enumerate(unique_shapes):
        print(shape, ':', counts[i])
    
    # converts to tuple
    return tuple(standard_shape)


# Standardize dataset to given shape
def reshape_dataset(data_dir, standard_shape):
    triplet_paths = [os.path.join(data_dir, p) for p in os.listdir(data_dir)]

    # Makes new directory for standardized dataset in the same directory as the original dataset
    new_data_dir = data_dir[:-1] + '_standardized/'
    if not os.path.exists(new_data_dir):
        os.makedirs(new_data_dir)
    
    # Resizes all images to the most common shape and saves them in the new directory
    new_triplet_paths = [os.path.join(new_data_dir, p) for p in os.listdir(data_dir)]
    for i, original_triplet_path in enumerate(triplet_paths):
        # Makes new dir for standardized triplet
        if not os.path.exists(new_triplet_paths[i]):
            os.makedirs(new_triplet_paths[i])
        # Resizes all images in triplet not matching provided size and saves them in the new dir
        for img_path in os.listdir(original_triplet_path):
            img = cv2.imread(os.path.join(original_triplet_path, img_path), cv2.IMREAD_GRAYSCALE)
            if img.shape != standard_shape:
                # cv2.resize takes (cols, rows)
                img = cv2.resize(img, (standard_shape[1], standard_shape[0]) , interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(new_triplet_paths[i], img_path), img)


if __name__ == '__main__':
    # Choose shape (rows, cols) OR standardize to most common shape in dataset (None)
    shape = None

    if shape is None:
        standard_shape = get_dataset_shapes('mini_test_triplets/')
        reshape_dataset('mini_test_triplets/', standard_shape)

    elif isinstance(shape, tuple):
        reshape_dataset('mini_test_triplets/', shape) 
    else:
        ("Invalid shape. Provide a valid shape in the form (# of rows, # of cols)")

    print('------------------AFTER STANDARDIZATION------------------')
    get_dataset_shapes('mini_test_triplets_standardized/')