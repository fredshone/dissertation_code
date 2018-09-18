"""Tool for visualising detection ground truth data so that MOT tracking ground truth can be labelled."""

import numpy as np
import os
import argparse
import cv2
from object_detection.utils import label_map_util

history_list = ['current', 'previous1', 'previous2']

colours = [(255, 255, 255),
           (100, 50, 200),
           (255, 100, 0),
           (0, 255, 100),
           (100, 0, 255),
           (100, 100, 200),
           (100, 200, 100),
           (200, 100, 50),
           (200, 200, 100),
           (255, 100, 200)
           ]


def get_colour(index):
    get = int(index) % len(colours)
    return colours[get]


def get_name():
    i = 0
    while True:
        i = i+1 % 100
        yield names[i]


def ask_user(question):
    check = str(input(question + " (Y/N): ")).lower().strip()
    try:
        if check[0] == 'y':
            return True
        elif check[0] == 'n':
            return False
        else:
            print('Invalid Input')
            return ask_user(question)
    except Exception as error:
        print("Please enter valid inputs")
        print(error)
        return ask_user(question)


# Load label map
PATH_TO_LABELS = os.path.join('object_detection', 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 100
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
                                                            max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

font = cv2.FONT_HERSHEY_SIMPLEX

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('det', type=str, help='location of test directory detections')
    parser.add_argument('--scale', type=float, default=0.5, help='display scale')
    args = parser.parse_args()

    gt_dir = os.path.dirname(args.det)
    test_dir = os.path.dirname(gt_dir)
    img_dir = os.path.join(test_dir, 'img1')
    print('\n> Accessing test directory: {}'.format(test_dir))
    gt_raw = np.load(args.det)
    print('\t Loaded {} lines of detection'.format(len(gt_raw)))

    # Add in temp ids
    for i, line in enumerate(gt_raw):
        line[1] = i

    gt_name = os.path.basename(args.det)[:-3] + 'csv'
    print('\n> Saving tracking gt as: {}'.format(gt_name))
    np.savetxt(os.path.join(gt_dir, gt_name), gt_raw, delimiter=",")

    # Group gt by frame
    gt_grouped = {}
    gt_group = []
    frame_list = []
    for gt in gt_raw:
        frame = int(gt[0])
        if frame in frame_list:
            gt_group.append(gt)
        else:
            if gt_group:
                gt_grouped[old_frame] = gt_group

            gt_group = [gt]
            frame_list.append(frame)
        old_frame = frame
    gt_grouped[old_frame] = gt_group

    # Loop through groups, load images before
    for frame, group in gt_grouped.items():
        print('\n-------------------------------------------------------'
              '\nGround Truth for frame {} with {} instances'.format(frame, len(group)))
        image_num = frame
        images = []
        # Load test image
        image_path = os.path.join(img_dir, 'timelapse_' + str(frame).zfill(5) + '.jpg')
        image = cv2.imread(image_path)
        h, w, c = image.shape
        for history in history_list:
            print('\n-------------------------'
                  '\n{}: frame {}'.format(history, image_num))
            image_path = os.path.join(img_dir, 'timelapse_' + str(image_num).zfill(5) + '.jpg')

            try:
                image = cv2.imread(image_path)
                h, w, c = image.shape
                image = cv2.resize(image, (int(w * args.scale), int(h * args.scale)))

                cv2.putText(image, 'Frame: ' + str(image_num), (10, 25), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

                hist_group = gt_grouped.get(image_num, False)
                if hist_group:
                    frames = []
                    for gt in hist_group:
                        frame, instance, left, top, width, height, conf = gt[0:7]
                        frames.append(int(frame))

                        top_left = [left, top]
                        top_right = [left + width, top]
                        bottom_right = [left + width, top + height]
                        bottom_left = [left, top + height]
                        text_centre = (int(left) + 2, int(top + height) - 2)

                        pts = np.array([top_left, top_right, bottom_right, bottom_left], np.int32)
                        pts = pts.reshape((-1, 1, 2))

                        cv2.polylines(image, [pts], True, get_colour(instance))
                        cv2.putText(image, str(int(instance)), text_centre, font, 0.6, get_colour(instance), 1, cv2.LINE_AA)

                    print('Frames plotted: {}'.format(str(frames)))

            except:
                print('{} not available, creating blank image w:{} h:{}'.format(history, w, h))
                image = np.zeros((w, h, 1), np.uint8)
                pass

            images.append(image)
            image_num -= 1

        h0, w0, c0 = images[0].shape
        h1, w1, c1 = images[1].shape
        h2, w2, c2 = images[2].shape

        # create empty matrix
        vis = np.zeros((h0 + h1 + h2, max(w0, w1, w2), 3), np.uint8)

        # combine 2 images
        vis[:h0, :w0, :c0] = images[0]
        vis[h0:h0+h1, :w1, :c1] = images[1]
        vis[h0+h1:h0+h1+h2, :w2, :c2] = images[2]
        h, w, c = vis.shape
        vis = cv2.resize(vis, (int(w/1.5), int(h/1.5)))

        cv2.imshow('all', vis)
        cv2.moveWindow("all", 0, 0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        continue








