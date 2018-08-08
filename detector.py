#General
import os
from datetime import datetime
from datetime import timedelta
import cv2
import numpy as np
import argparse

# Object Detection
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import tracker_utils

# Setup
start_time = '08:00'  # Assumes camera turned on at 8am every day
frame_step = 3  # Seconds
detection_threshold = 0.5

stick_location = '/Volumes/FRED SHONE'  # Memory Stick
drive_location = '/Volumes/Elements'  # CASA Drive
test0 = 'test_inputs/test0'
test1 = 'test_inputs/test1'
test2 = 'test_inputs/test2'
test3 = 'test_inputs/test3'

default = test2
default_model = 'ssd_inception_v2_coco'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--test', action='store_true', help='Initiate testing mode')
    parser.add_argument('-M', '--model', default=default_model, help='Choose detection model')
    parser.add_argument('--dir', default=default, help='Frame directory')
    parser.add_argument('--sensitivity', default=0.0001, help='Sensitivity of motion trigger')
    parser.add_argument('-D', '--display', action='store_true', help='Display video of detections')
    parser.add_argument('-V', '--verbose', action='store_true')
    args = parser.parse_args()

    print('Accessing images from {}'.format(args.dir))
    print('Trigger sensitivity set to {}'.format(args.sensitivity))
    print('Object detection with {}'.format(args.model))

# set set up detection model

model_zoo = {'sd_mobilenet_v1_coco': 'ssd_mobilenet_v1_coco_2018_01_28',
             'sd_mobilenet_v2_coco': 'ssd_mobilenet_v2_coco_2018_03_29',
             'ssdlite_mobilenet_v2_coco': 'ssdlite_mobilenet_v2_coco_2018_05_09',
             'ssd_inception_v2_coco': 'ssd_inception_v2_coco_2018_01_28',
             'faster_rcnn_inception_v2_coco': 'faster_rcnn_inception_v2_coco_2018_01_28',
             'faster_rcnn_resnet50_coco': 'faster_rcnn_resnet50_coco_2018_01_28',
             'faster_rcnn_resnet50_lowproposals_coco': 'faster_rcnn_resnet50_lowproposals_coco_2018_01_28',
             'rfcn_resnet101_coco': 'rfcn_resnet101_coco_2018_01_28',
             'faster_rcnn_resnet101_coco': 'faster_rcnn_resnet101_coco_2018_01_28',
             'faster_rcnn_resnet101_lowproposals_coco': 'faster_rcnn_resnet101_lowproposals_coco_2018_01_28',
             'mask_rcnn_inception_v2_coco': 'mask_rcnn_inception_v2_coco_2018_01_28'}

##################################################
# Prepare paths
MODEL_NAME = model_zoo.get(args.model)
PATH_TO_LABELS = os.path.join('object_detection', 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 10

# Load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tracker_utils.load_model(MODEL_NAME)

##################################################
# Set up time and date
base_path = os.path.basename(args.dir)
start_datetime = datetime.strptime(base_path+' '+start_time, '%m-%d-%Y %H:%M')
print("Set up for regular directory mode:\n"
      "Start date id {}\n"
      "Start time is {}".format(base_path, start_time))

##################################################
# Make list of frames (later make list of directory days and frames within)
images = sorted([image for image in os.listdir(args.dir) if not image.startswith('.')])
image_count = len(images)
print('Found {}: {} images'.format(args.dir, image_count))
print('\t>>>Frame rate assumed to be {} seconds per frame'.format(frame_step))

# Initiate loop
background = None  # Initialise with no background
time_stamp_dict = {}  # Dictionary for timestamps
result_array = np.empty((0, 10))  # Results array
detection_times = []  # Detection timer array

with detection_graph.as_default():
    with tf.Session() as sess:
        print("Commencing loop:")
        for image in images:

            # Extract frame number from image path
            image_index = [int(s) for s in list(image) if s.isdigit()]
            image_index = ''.join(str(x) for x in image_index)
            image_index = int(image_index)

            # Calculate time stamp of frame
            time_delta = timedelta(seconds=(frame_step * image_index))
            time_stamp = start_datetime + time_delta
            time_stamp_dict[image_index] = time_stamp  # Add to dictionary

            image_path = os.path.join(args.dir, image)
            img = cv2.imread(image_path)
            img = cv2.resize(img, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)  # Resize to 25% area

            # Trigger ##############################################
            thumb = tracker_utils.trigger_image(img)

            if tracker_utils.motion_trigger(args.sensitivity, thumb, background):

                timer_start = datetime.now()

                # If trigger then run object detection
                detections_dict = tracker_utils.run_object_detection(sess, img)

                if args.test:
                    timer = datetime.now() - timer_start
                    detection_times.append(timer)

                if args.verbose:
                    print('Motion detection at frame {}'.format(image_index))

                if args.display:
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        img,
                        detections_dict['detection_boxes'],
                        detections_dict['detection_classes'],
                        detections_dict['detection_scores'],
                        category_index,
                        instance_masks=detections_dict.get('detection_masks'),
                        use_normalized_coordinates=True,
                        line_thickness=4,
                        min_score_thresh=detection_threshold)
                    cv2.imshow('frame', img)
                    cv2.waitKey(1)

                # Collect results in correct format for MOT
                if detections_dict:
                    num_detections = detections_dict['num_detections']
                    if num_detections:
                        detection_classes = detections_dict['detection_classes']
                        detection_scores = detections_dict['detection_scores']
                        mask = (detection_classes == 1) & (detection_scores > detection_threshold)
                        detection_boxes = detections_dict['detection_boxes'][mask]
                        detection_scores = detection_scores[mask]

                        height, width = img.shape[:2]
                        for detection, confidence in zip(detection_boxes, detection_scores):
                            y_min, x_min, y_max, x_max = detection
                            bb_top = y_min * height
                            bb_left = x_min * width
                            bb_height = (y_max * height) - bb_top
                            bb_width = (x_max * width) - bb_left
                            result = image_index, -1, bb_left, bb_top, bb_width, bb_height, confidence, -1, -1, -1
                            result_array = np.append(result_array, [result], axis=0)

            background = thumb  # Set background for next frame

        # Save results
        string = str(args.dir) + '_' + args.model + '.npy'
        print('Saving {} detections as {}'.format(len(result_array), string))
        np.save(string, result_array)
        print('Completed')
        if args.display:
            cv2.destroyAllWindows()

        if args.test:
            print("Test results:")

            if detection_times:
                detections = len(detection_times)
                total = 0.
                for time in detection_times:
                    total += time.microseconds / 1000000
                    total += time.seconds
                av_time = total / detections
                print('\t>>> {} detections\n'
                      '\t>>> {} seconds per detection'.format(detections, av_time))






