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
# - training (run in this directory)
#     - image_02
#           - 0000
#               - 000000.png
#               - etc
#           - etc
#     - label_02
#          - 0000.txt
#          - etc
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

# MOT data = <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
# /Volumes/FRED SHONE > stick path

type_dict = {'Car': False,
             'Van': False,
             'Truck': False,
             'Pedestrian': True,
             'Person_sitting': True,
             'Cyclist': True,
             'Tram': False,
             'Misc': False,
             'DontCare': False}

truncation_dict = {0: True,
                   1: True,
                   2: False}

occlusion_dict = {0: True,
                  1: True,
                  2: False,
                  3: False}


def split_line(line_in):

    if ',' in line_in:
        line_split = line_in.split(',')
    else:
        line_split = line_in.split(' ')

    return line_split


def adjust(line_split):

    frame = int(line_split[0])
    object_id = int(line_split[1])
    type_str = str(line_split[2])
    truncation = int(line_split[3])
    occlusion = int(line_split[4])
    left = int(float(line_split[6]))
    top = int(float(line_split[7]))
    right = int(float(line_split[8]))
    bottom = int(float(line_split[9]))

    type_switch = type_dict.get(type_str)
    trunc_switch = truncation_dict.get(truncation)
    occ_switch = occlusion_dict.get(occlusion)

    if type_switch and trunc_switch and occ_switch:

        width = right - left
        height = bottom - top

        new_data = [frame, object_id, left, top, width, height, '-1', '-1', '-1']
        line_str = ','.join([str(d) for d in new_data])

        return line_str + '\n'

    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='location of raw ground truth')
    parser.add_argument('--save', type=str, default='MOT/train', help='destination directory')

    args = parser.parse_args()

    base_load_path = args.dir
    base_path = args.save

    base_gt_path = os.path.join(base_load_path, 'label_02')
    base_image_path = os.path.join(base_load_path, 'image_02')

    gts = os.listdir(base_gt_path)
    gts = [d for d in gts if d[0] is not '.']

    for i, gt in enumerate(gts):

        load_gt_path = os.path.join(base_gt_path, gt)

        directory_name = gt[:4]
        new_directory_name = 'KITTI_' + directory_name

        image_dir_path = os.path.join(base_path, new_directory_name, 'img1')
        gt_path = os.path.join(base_path, new_directory_name, 'gt')

        print('Loading from: {}'
              '\nSaving gt to: {}'
              '\nSaving images to: {}\n'.
              format(load_gt_path, gt_path, image_dir_path))

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

        with open(load_gt_path) as read_file, open(gt_path + '/gt.txt', 'a') as write_file:
            lines = read_file.readlines()
            for line in lines:

                split_line = [line.split(' ')]

                new_line = adjust(split_line[0])

                if new_line:

                    write_file.writelines(new_line)

                    # load image
                    image_num = str(split_line[0][0]).zfill(6)
                    img_load_path = os.path.join(base_image_path, gt[:4], image_num + '.png')

                    img = cv2.imread(img_load_path)

                    # calculate new image name
                    new_image_path = os.path.join(image_dir_path, image_num + '.jpg')
                    # save image
                    cv2.imwrite(new_image_path, img)

        print('\n> Completed {}: {}/{}\n'.format(new_directory_name, i+1, len(gts)))

    print('\n> Completed All\n')