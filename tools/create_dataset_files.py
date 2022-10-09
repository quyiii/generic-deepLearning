import os
import numpy as np
import h5py
import json
import torch
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample

def create_dataset_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder, max_len, resize):
    """
    Creates input files (.h5 which is a huge file) for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    :param resize: image resize size, such as 256
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}
    # read json with imagepath captions ...
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)
    
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        image_name = img['filename']
        for c in img['sentences']:
            # update the frequency of each words
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                # add image caption
                captions.append(c['tokens'])
        if len(captions) == 0:
            raise RuntimeError('image->{} do not have caption'.format(image_name))

        # get image_path
        path = os.path.join(image_folder, img['filepath'], image_name) if dataset == 'coco' else os.path.join(image_folder, image_name)

        split = img['split']
        if split in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif split in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif split in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)
        else:
            raise RuntimeError("split for image->{} is wrong type: {}".format(image_name, split))
    
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    # add unkown sign (words less frequent or do not appear)
    word_map['<unk>'] = len(word_map) + 1
    # add start sign
    word_map['<start>'] = len(word_map) + 1
    # add end sign
    word_map['<end>'] = len(word_map) + 1
    # add pad sign
    word_map['<pad>'] = 0

    # create root name for all output files
    root_filename = dataset + '_' + str(captions_per_image) + \
                    '_caps_per_img_' + str(min_word_freq) + '_min_word_freq'

    with open(os.path.join(output_folder, 'WORDMAP_' + root_filename + '.json'), 'w') as j:
        # write word_map to file as json
        json.dump(word_map, j)
    
    # set random seed
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:
        # a means write at end
        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + root_filename + '.hdf5'), 'a') as h:
            # sample number
            h.attrs['captions_per_image'] = captions_per_image
            # create dataset inside .hdf5 to store image
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("Reading {} images and captions, storing to file".format(split))

            enc_captions = []
            caplens = []

            # store image, sample captions and encode
            for i, path in enumerate(tqdm(impaths)):
                # sample captions_per_image captions for each image
                if len(imcaps[i]) < captions_per_image:
                    # random.choice: choose a random element in sequence
                    # for _ in range(n): just for circulate n times 
                    # use choice and for _ to let len(imcaps) == captions_per_image
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                assert len(captions) == captions_per_image

                # store image
                img = imread(path)
                if len(img.shape) == 2:
                    # h,w -> h,w,c
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255
                # save image to .hdf5
                images[i] = img

                # encode images
                for j, c in enumerate(captions):
                    # encode caption
                    # dict.get(key, replace) find key's value in dict, if not find output replace
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c + \
                             word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
                    
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)
            
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            with open(os.path.join(output_folder, split + '_CAPTIONS_' + root_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)
            
            with open(os.path.join(output_folder, split + '_CAPLENS_' + root_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


 
if __name__ == "__main__":
    imread("xxx")