import os
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt


# Function for generating training data
#
# Image pairs are created with a binary label signifying match (1) or mismatch (0)
#
# Pairs are randomly extracted from multiple directories of images, each with their own ground truth in MOT format
#
# Augmentation is optional and can be tuned as required
#
# Assumes input data as follows:
#
# - MOT
#     - train
#         - name
#             - gt
#                 - gt.txt
#             - img1
#                 - 000001.jpg
#                 - etc


def get_ids(instances):
    ids = [i[1] for i in instances]
    ids = [i for i in ids if i is not -1]
    return list(set(ids))


def get_rand_instance(instance):
    return random.choice(instance)


def get_crop(instance, image_dir, use):
    # Get image
    img = get_image(instance, image_dir, use)

    row, col, ch = img.shape

    # Extract bounding box
    x1 = int(instance[2])
    y1 = int(instance[3])
    x2 = int(instance[4] + x1)
    y2 = int(instance[5] + y1)

    # Try to deal with negative values
    if x1 < 0:
        x1 = 0
    if x2 > col:
        x2 = col
    if y1 < 0:
        y1 = 0
    if y2 > row:
        y2 = row

    return img[y1:y2, x1:x2]


def get_image(instance, image_dir, use):
    image_num = instance[0]  # Extract image number from gt instance
    if not use == 'test':
        image_num = str(image_num).zfill(6)
    else:
        image_num = 'timelapse_' + str(image_num).zfill(5)
    image_path = os.path.join(image_dir, image_num+'.jpg')
    img = cv2.imread(image_path)
    return img


def rand_resize(image, lower=0.5, upper=1.5):
    w, h, d = image.shape
    f = random.uniform(lower, upper)
    return cv2.resize(image, (int(f*h), int(f*w)))


def gaussian_blur(images, verbose=False, test=False):
    y = random.choice([1, 3, 5, 7])
    x = random.choice([1, 3, 5, 7])
    if test:
        y = 15
        x = 15
    if verbose:
        print('Applying gaussian blur: {} {}'.format(y, x))
    output = [cv2.GaussianBlur(image, (y, x), 0) for image in images]
    return output


def median_blur(images, verbose=False, test=False):
    y = random.choice([1, 3, 5, 7])
    if test:
        y = 11
    if verbose:
        print('Applying median blur: {}'.format(y))
    output = [cv2.medianBlur(image, y) for image in images]
    return output


def rand_blur(images, verbose=False):
    c = random.randint(0, 1)
    if c == 0:
        images = gaussian_blur(images, verbose)
    if c == 1:
        images = median_blur(images, verbose)
    return images


def gaussian_noise(images, verbose=False, test=False):
    if verbose:
        print('Applying gaussian noise')
    row, col, ch = images[0].shape
    mean = 0
    var = random.choice([20, 30, 40, 50])
    if test:
        var = 50
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    output = [image + gauss for image in images]
    output = [np.clip(out, 0, 255) for out in output]
    output = [np.array(out, dtype=np.uint8) for out in output]
    return output


def salt_pepper_noise(images, verbose=False, test=False):
    if verbose:
        print('Applying s&p noise')
    s_vs_p = 0.5
    amount = random.choice([2, 3, 4, 5])
    if test:
        amount = 5
    amount = amount/1000
    # Salt mode
    num_salt = np.ceil(amount * images[0].size * s_vs_p)
    noise_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in images[0].shape]
    # Pepper mode
    num_pepper = np.ceil(amount * images[0].size * (1. - s_vs_p))
    noise_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in images[0].shape]
    outputs = []
    for image in images:
        output = np.copy(image)
        # Salt mode
        output[noise_salt] = 255
        # Pepper mode
        output[noise_pepper] = 0
        output = np.clip(output, 1, 254)
        output = np.array(output, dtype=np.uint8)
        outputs.append(output)
    return outputs


def poisson_noise(images, verbose=False, test=False):
    noise = len(np.unique(images[0]))
    noise = 2 ** np.ceil(np.log2(noise))
    val = random.choice([1, 2])
    if test:
        val = 2
    if verbose:
        print('Applying poisson noise of {}'.format(val))
    outputs = []
    for image in images:
        output = np.random.poisson(image * noise) / (float(noise) * val)
        output = np.clip(output, 1, 254)
        output = np.array(output, dtype=np.uint8)
        outputs.append(output)
    return outputs


def speckle_noise(images, verbose=False, test=False):
    if verbose:
        print('Applying speckle noise')
    val = random.choice([10, 20])
    if test:
        val = 10
    row, col, ch = images[0].shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)
    outputs = []
    for image in images:
        output = image + (image * gauss / val)
        output = np.clip(output, 1, 254)
        output = np.array(output, dtype=np.uint8)
        outputs.append(output)
    return outputs


def rand_noise(images, verbose=False):
    c = random.randint(0, 2)
    if c == 0:
        images = gaussian_noise(images, verbose)
    if c == 1:
        images = salt_pepper_noise(images, verbose)
    if c == 2:
        images = speckle_noise(images, verbose)
    # if c == 3:
    #     images = poisson_noise(images, verbose)
    return images


def rand_rotation(images, verbose=False, test=False):
    outputs = np.copy(images)
    rows, cols = outputs[0].shape[:2]
    r = random.randint(5, 15)
    if test:
        r = 15
    r = r * random.choice([-1, 1])
    m = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), r, 1)
    if verbose:
        print('Rotation:{} '.format(r))
    return [cv2.warpAffine(output, m, (cols, rows)) for output in outputs]


def rand_crop(images, verbose, min_crop=0.05, max_crop=0.25, test=False):

    crop_h = random.uniform(min_crop, max_crop)
    crop_w = random.uniform(min_crop, max_crop)

    if test:
        crop_h = .25
        crop_w = .25

    crop_h1 = random.uniform(0, crop_h)
    crop_w1 = random.uniform(0, crop_w)

    h, w = images[0].shape[:2]

    # crop height
    y1 = int(crop_h1 * h)
    y2 = int((1 - crop_h + crop_h1) * h)

    # crop width
    x1 = int(crop_w1 * w)
    x2 = int((1 - crop_w + crop_w1) * w)

    if verbose:
        print('Cropping lhs:{} rhs:{} top:{} bottom:{}'.format(x1, w-x2, y1, h-y2))

    outputs = [image[y1:y2, x1:x2] for image in images]

    # Return to original size
    outputs = [cv2.resize(output, (w, h)) for output in outputs]

    return outputs


def rand_obstruction(images, verbose, min_obs=0.2, max_obs=0.5, test=False):
    outputs = np.copy(images)

    h, w = outputs[0].shape[:2]

    obs_h = random.uniform(min_obs, max_obs)
    obs_w = random.uniform(min_obs, max_obs)

    if test:
        obs_h = max_obs
        obs_w = max_obs

    obs_y = random.uniform(0, 1 - obs_h)
    obs_x = random.uniform(0, 1 - obs_w)

    # obs height
    y1 = int(obs_y * h)
    y2 = int((obs_h + obs_y) * h)

    # obs width
    x1 = int(obs_x * w)
    x2 = int((obs_w + obs_x) * w)

    if verbose:
        print('Obstructing h:{} w:{} y:{} x:{}'.format(y2-y1, x2-x1, y1, x1))

    colour = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

    for output in outputs:
        output[y1:y2, x1:x2] = colour

    return outputs


def rand_hue(images, verbose=False, test=False):
    outputs = []
    dh = random.randint(5, 20)
    if test:
        dh = 30
    c = random.choice([1, 0])

    for image in images:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # convert it to hsv
        h, s, v = cv2.split(hsv)

        if c:
            lim = 255 - dh
            h[h > lim] = 255
            h[h <= lim] += dh

        if not c:
            lim = dh
            h[h < lim] = 0
            h[h >= lim] -= dh

        final_hsv = cv2.merge((h, s, v))
        if verbose:
            print('Altering hue: {}'.format(dh))
        output = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        outputs.append(output)
    return outputs


def rand_saturation(images, verbose=False, test=False):
    outputs = []
    ds = random.randint(5, 30)
    if test:
        ds = 30
    c = random.choice([1, 0])

    for image in images:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # convert it to hsv
        h, s, v = cv2.split(hsv)

        if c:
            lim = 255 - ds
            s[s > lim] = 255
            s[s <= lim] += ds

        if not c:
            lim = ds
            s[s < lim] = 0
            s[s >= lim] -= ds

        final_hsv = cv2.merge((h, s, v))
        if verbose:
            print('Altering saturation: {}'.format(ds))
        output = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        outputs.append(output)
    return outputs


def rand_brightness(images, verbose=False, test=False):
    outputs = []
    dv = random.randint(1, 75)
    if test:
        dv = 75
    c = random.choice([1, 0])

    for image in images:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # convert it to hsv
        h, s, v = cv2.split(hsv)

        if c:
            lim = 255 - dv
            v[v > lim] = 255
            v[v <= lim] += dv

        if not c:
            lim = dv
            v[v < lim] = 0
            v[v >= lim] -= dv

        final_hsv = cv2.merge((h, s, v))
        if verbose:
            print('Altering brightness: {}'.format(dv))
        output = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        outputs.append(output)
    return outputs


def random_augmentation(images, p, verbose=False):
    if len(images) > 1:
        c = np.random.choice(a=[1, 0], size=8, p=[p, 1-p])
    else:
        c = np.random.choice(a=[1, 0], size=5, p=[p, 1-p])
        c = np.append(c, [0, 0, 0])

    if c[0]:
        try:
            images = rand_blur(images, verbose)
        except:
            if verbose:
                print("Random blur failed")
            pass
    if c[1]:
        try:
            images = rand_noise(images, verbose)
        except:
            if verbose:
                print("Random noise failed")
            pass
    if c[2]:
        try:
            images = rand_rotation(images, verbose)
        except:
            if verbose:
                print("Random rotation failed")
            pass
    if c[3]:
        try:
            images = rand_crop(images, verbose)
        except:
            if verbose:
                print("Random crop failed")
            pass
    if c[4]:
        try:
            images = rand_obstruction(images, verbose)
        except:
            if verbose:
                print("Random obstruction failed")
            pass
    if c[5]:
        try:
            images = rand_hue(images, verbose)
        except:
            if verbose:
                print("Random hue failed")
            pass
    if c[6]:
        try:
            images = rand_saturation(images, verbose)
        except:
            if verbose:
                print("Random saturation failed")
            pass
    if c[7]:
        try:
            images = rand_brightness(images, verbose)
        except:
            if verbose:
                print("Random brightness failed")
            pass
    return images


def get_mismatch(instances, image_dir, use, verbose=False, image_verbose=False):
    # Get a random id from available ids in instance list
    ids = get_ids(instances)
    ped1_id, ped2_id = random.sample(ids, 2)

    # Filter for ids
    instances1 = [i for i in instances if i[1] == ped1_id]
    instances2 = [i for i in instances if i[1] == ped2_id]

    # Get a random instance of ped1
    instance1 = random.choice(instances1)
    roi1 = get_crop(instance1, image_dir, use)

    # Get a random instance of ped2
    instance2 = random.choice(instances2)
    roi2 = get_crop(instance2, image_dir, use)

    if all(roi1.shape) and all(roi2.shape):

        if verbose:
            print('Mismatch from {}, ids: {} & {} from frames {} & {}'.
                  format(image_dir, ped1_id, ped2_id, instance1[0], instance2[0]))

        if image_verbose:
            f, ax = plt.subplots(1, 2, figsize=(15, 10))
            ax[0].imshow(cv2.cvtColor(roi1, cv2.COLOR_BGR2RGB))
            ax[1].imshow(cv2.cvtColor(roi2, cv2.COLOR_BGR2RGB))
            plt.show()

    else:
        roi1, roi2 = get_match(instances, image_dir, verbose, image_verbose)

    return [roi1, roi2]


def get_match(instances, image_dir, use, verbose=False, image_verbose=False):

    unique = False
    while not unique:
        # Get a random id from available ids in instance list
        ped_id = random.choice(get_ids(instances))
        # Filter for id
        instances_match = [i for i in instances if i[1] == ped_id]

        # Get a random instance of ped
        instance1 = random.choice(instances_match)

        for t in range(3):
            # Get a random instance of ped
            instance2 = random.choice(instances_match)
            if not instance2[0] == instance1[0]:
                unique = True
                roi1 = get_crop(instance1, image_dir, use)
                roi2 = get_crop(instance2, image_dir, use)
                break

    if all(roi1.shape) and all(roi2.shape):

        if verbose:
            print('Match from {}, id: {} from frames {} & {}'.format(image_dir, ped_id, instance1[0], instance2[0]))

        if image_verbose:
            f, ax = plt.subplots(1, 2, figsize=(15, 10))
            ax[0].imshow(cv2.cvtColor(roi1, cv2.COLOR_BGR2RGB))
            ax[1].imshow(cv2.cvtColor(roi2, cv2.COLOR_BGR2RGB))
            plt.show()

    else:
        roi1, roi2 = get_match(instances, image_dir, verbose, image_verbose)

    return [roi1, roi2]


def filter_instances(_line):

    return (_line[4] * _line[5] > 10000) and (_line[4] > 50) and (_line[5] > 50) and ((_line[5] / _line[4]) > 0.1)


def filter_gt(gt_in):
    size = 10000
    dim = 40
    ratio = 0.2

    if gt_in[4] * gt_in[5] > size and (gt_in[4] > dim) and (gt_in[5] > dim) and ((gt_in[5] / gt_in[4]) > ratio) and ((gt_in[4] / gt_in[5]) > ratio):
        return True
    else:
        return False


def make_resize_square(img, size):

    in_size = img.shape[:2]
    ratio = float(size[0]) / max(in_size)
    out_size = tuple([int(x * ratio) for x in in_size])
    img = cv2.resize(img, (out_size[1], out_size[0]))

    delta_w = size[0] - out_size[1]
    delta_h = size[0] - out_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def get_weights(target_directory, verbose=False):

    if target_directory == 'test':
        return None, None

    index = []
    match_info = []
    missmatch_info = []

    dir_location = os.path.join('MOT', target_directory)

    if verbose:
        print('\nBuilding sampling weights for {}:'.format(dir_location))

    dir_names = os.listdir(dir_location)
    dir_names = [n for n in dir_names if n[0] is not '.']  # Filter out hidden files

    for i, dir_name in enumerate(sorted(dir_names)):
        dir_path = os.path.join(dir_location, dir_name)
        gt_dir = os.path.join(dir_path, 'gt', 'gt.txt')
        gt_name = os.path.basename(dir_path)

        gt = []
        with open(gt_dir) as f:
            for line in f:
                gt.append([int(n) for n in line.strip().split(',')[:-1]])  # Loose last entry which is not an integer

        # Filter gt for certain classes
        classes = [-1, 1, 2, 13]
        gt_class = [line for line in gt if line[7] in classes]

        # Filter for crop size
        gt_class = [line for line in gt_class if filter_gt(line)]

        # Check how many Ids are available
        ids = [c[1] for c in gt_class]

        _num_instances = len(gt_class)
        _num_objects = len(set(ids))

        if _num_objects:
            _av_frames = _num_instances / _num_objects
            _match_info = _num_objects * _av_frames * (_av_frames - 1) * 0.5
            _missmatch_info = _num_objects * (_num_objects - 1) * 0.5

            index.append(gt_name)
            match_info.append(int(_match_info))
            missmatch_info.append(int(_missmatch_info))

            if verbose:
                print('>>> Directory {}/{}: {}: match information weighting = {}, missmatch information weighting = {}'.
                      format(i, len(dir_names), gt_name, _match_info, _missmatch_info))

    # Build dictionaries
    match_dict = {key: value for (key, value) in zip(index, match_info)}
    missmatch_dict = {key: value for (key, value) in zip(index, missmatch_info)}

    return match_dict, missmatch_dict


def get_instances(use, weights):

    # Find directories
    dir_location = os.path.join('MOT', use)
    dir_names = os.listdir(dir_location)
    dir_names = [n for n in dir_names if n[0] is not '.']  # Filter out hidden files

    if use == 'test':
        dir_names = ['test3', 'test4']

    if weights:
        weights_list = [int(weights.get(str(d), 0)) for d in dir_names]
    else:
        weights_list = None

    # Choose random directory
    dir_name = random.choices(dir_names, weights=weights_list)

    dir_path = os.path.join(dir_location, dir_name[0])
    gt_dir = os.path.join(dir_path, 'gt/gt.txt')
    image_dir = os.path.join(dir_path, 'img1')

    gt = []

    if not use == 'test':
        with open(gt_dir) as f:
            for line in f:
                gt.append([int(n) for n in line.strip().split(',')[:-1]])  # Ignore last entry which is not an integer

        # Filter gt for certain classes
        classes = [1, 2, -1]  # Person and person on vehicle
        gt_class = [line for line in gt if line[7] in classes]

    else:
        with open(gt_dir) as f:
            for line in f:
                gt_line = [int(float(n)) for n in line.strip().split('\t')]
                gt_line[2] = gt_line[2] * 2  # Adjust for previous resize
                gt_line[3] = gt_line[3] * 2
                gt_line[4] = gt_line[4] * 2
                gt_line[5] = gt_line[5] * 2
                gt.append(gt_line)
        gt_class = gt

    # Filter for crop size
    gt_class = [line for line in gt_class if filter_gt(line)]

    return gt_class, image_dir


def generator(use='train',
              image_size=(197, 197, 3),
              aug_mismatches=True,
              border=False,
              mp=0.5,
              ap=0.2,
              batch_size=256,
              image_verbose=False,
              verbose=False):

    # calculate information weights
    match_weights, miss_weights = get_weights(use, verbose)

    # Loop through batch size
    while True:

        samples1 = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))
        samples2 = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))
        tags = np.zeros((batch_size,))

        for i in range(batch_size):

            # Choose type, can either be match (1) or not (0)
            tag = np.random.choice([0, 1], p=[mp, 1-mp])  # Change proportion of matches to mismatches

            if tag:
                weights = match_weights
            else:
                weights = miss_weights

            # Choose random ground truth location,
            instances, image_dir = get_instances(use, weights)

            string = 'Mismatch'
            if tag:
                string = 'Match'

            if verbose:
                print('\n------------------------------------------------------'
                      '\n\tImage {}'
                      '\n> {}'.
                      format(string, image_dir))

            # Get images
            if tag:
                sample1, sample2 = get_match(instances, image_dir, use, image_verbose=image_verbose, verbose=verbose)
            else:
                sample1, sample2 = get_mismatch(instances, image_dir, use, image_verbose=image_verbose, verbose=verbose)

            # Augment training data
            if use == 'train' or use == 'validate':
                # Pairwise augmentation
                if verbose:
                    print('Applying symmetric random augmentation to images:')
                samples = [sample1, sample2]
                samples = random_augmentation(samples, ap, verbose)

                # Asymmetric augmentation
                sample1 = [samples[0]]
                sample2 = [samples[1]]

                if use == 'train':

                    if tag or ((not tag) and aug_mismatches):
                        if verbose:
                            print('Applying random augmentation to image 1:')
                        sample1 = random_augmentation(sample1, ap, verbose)
                        if verbose:
                            print('Applying random augmentation to image 2:')
                        sample2 = random_augmentation(sample2, ap, verbose)

                # Unpack from list
                sample1 = sample1[0]
                sample2 = sample2[0]

            # Image manipulations
            if border:
                sample1 = make_resize_square(sample1, image_size)
                sample2 = make_resize_square(sample2, image_size)
            else:
                sample1 = cv2.resize(sample1, (image_size[0], image_size[1]))
                sample2 = cv2.resize(sample2, (image_size[0], image_size[1]))

            if image_verbose:
                f, ax = plt.subplots(1, 2, figsize=(15, 10))
                ax[0].imshow(cv2.cvtColor(sample1, cv2.COLOR_BGR2RGB))
                ax[1].imshow(cv2.cvtColor(sample2, cv2.COLOR_BGR2RGB))
                plt.show()

            # Rescale image channels by /255
            sample1 = sample1.astype('float') / 255
            sample2 = sample2.astype('float') / 255
            tag = np.array(tag).astype('float')

            # Add to batch
            samples1[i] = sample1
            samples2[i] = sample2
            tags[i] = tag

            output = [np.array(samples1), np.array(samples2)], np.array(tags)

        if verbose:
            print('Batch produced with {}/{} matches'.format(int(sum(tags)), batch_size))

        yield output
