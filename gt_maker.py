import argparse
import os
import errno
import cv2

# Function for creating multiple object tracking training data from single object training data
#
# Groups multiple sources of images and ground truth data into single ground truth
#
# Assumes input data as follows:
#
# - dir
#     - dir1
#          - *.txt
#          - img
#             - 0001.jpg
#             - etc
#     - dir2
#          - *.txt
#          - img
#             - 0001.jpg
#             - etc
#     - etc
#
# Gives output data as follows:
#
# - MOT
#     - train
#         - name
#             - gt
#                 - gt.txt
#             - img1
#                 - 000001.jpg
#                 - etc

# /Volumes/FRED SHONE > stick path

scale = 2  # Double image size


def adjust(frame_count, object_count, line_in, scaler=2):
    """
    Re-formats ground truth data to MOT format.
    :param rame_count: Integer, input frame number for new format.
    :param object_count: Integer, input object id for new format.
    :param line_in: Input ground truth, either tab or comma delineated.
    :param scaler: Integer, option to re-scale ground truth detection location (default 2).
    :return: Returns new ground truth line in MOT format.
    """
    new_line = str(frame_count) + ',' + str(object_count)
    if ',' in line_in:
        line_split = line_in.split(',')
    else:
        line_split = line_in.split()

    for item in line_split:
        new_line += (',' + str(int(item)*scaler))
    for x in range(4):
        new_line += ',-1'
    return new_line + '\n'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='location of raw ground truth')
    parser.add_argument('name', type=str, help='name of new directory')
    parser.add_argument('--save', type=str, default='MOT/train', help='destination directory')

    args = parser.parse_args()

    base_load_path = args.dir
    base_path = os.path.join(args.save, args.name)
    image_dir_path = os.path.join(base_path, 'img1')
    gt_path = os.path.join(base_path, 'gt')

    print('Loading from: {}'
          '\nSaving gt to: {}'
          '\nSaving images to: {}\n'.
          format(base_load_path, gt_path, image_dir_path))

    try:
        os.makedirs(image_dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    try:
        os.makedirs(gt_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    object_counter = 1  # Give new object ID for each directory
    exit_frame_counter = 1

    directories = os.listdir(base_load_path)
    directories = [d for d in directories if d[0] is not '.']

    for i, directory in enumerate(directories):

        gt_load_path = os.path.join(base_load_path, directory, 'groundtruth_rect.txt')
        print('> Loading from {}'.format(directory))

        with open(gt_load_path) as read_file, open(gt_path + '/gt.txt', 'a') as write_file:
            lines = read_file.readlines()
            frame_counter = 1
            for line in lines:
                write_file.writelines(adjust(exit_frame_counter, object_counter, line, scaler=scale))

                # load image
                image_num = str(frame_counter).zfill(4)
                img_load_path = os.path.join(base_load_path, directory, 'img', image_num + '.jpg')
                img = cv2.imread(img_load_path)
                h, w = img.shape[:2]
                img = cv2.resize(img, (scale*w, scale*h))
                # calculate new image name
                new_image_num = str(exit_frame_counter).zfill(6)
                new_image_path = os.path.join(image_dir_path, new_image_num + '.jpg')
                # save image
                cv2.imwrite(new_image_path, img)

                # update counters
                frame_counter += 1
                exit_frame_counter += 1
        object_counter += 1
        print('\n> Completed {}: {}/{}\n'.format(directory, i+1, len(directories)))

    print('\n> Completed All\n')