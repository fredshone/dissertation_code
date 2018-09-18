"""
Module to run tests on Object Detection models.
Saves detection and performance results to given directory.
Tests all models in model zoo. This includes downloading model and weights. At last testing, some model weights could
not be accessed - these models are skipped and no results given.
"""
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
test = 'test_inputs'
test0 = 'test_inputs/test0'
test1 = 'test_inputs/test1'
test2 = 'test_inputs/test2'
test3 = 'test_inputs/test3'

default_test = test

model_zoo = {'ssd_mobilenet_v1_coco': 'ssd_mobilenet_v1_coco_2018_01_28',
             'ssd_mobilenet_v1_0.75_depth_coco': 'ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03',
             'ssd_mobilenet_v1_quantized_coco': 'ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_03',
             'ssd_mobilenet_v1_0.75_depth_quantized_coco': 'ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_03',
             'ssd_mobilenet_v1_ppn_coco': 'ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03',
             'ssd_mobilenet_v1_fpn_coco': 'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
             'ssd_resnet_50_fpn_coco': 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
             'sd_mobilenet_v2_coco': 'ssd_mobilenet_v2_coco_2018_03_29',
             'ssdlite_mobilenet_v2_coco': 'ssdlite_mobilenet_v2_coco_2018_05_09',
             'ssd_inception_v2_coco': 'ssd_inception_v2_coco_2018_01_28',
             'faster_rcnn_inception_v2_coco': 'faster_rcnn_inception_v2_coco_2018_01_28',
             'faster_rcnn_resnet50_coco': 'faster_rcnn_resnet50_coco_2018_01_28',
             'faster_rcnn_resnet50_lowproposals_coco': 'faster_rcnn_resnet50_lowproposals_coco_2018_01_28',
             'rfcn_resnet101_coco': 'rfcn_resnet101_coco_2018_01_28.tar.gz',
             'faster_rcnn_resnet101_coco': 'faster_rcnn_resnet101_coco_2018_01_28',
             'faster_rcnn_resnet101_lowproposals_coco': 'faster_rcnn_resnet101_lowproposals_coco_2018_01_28',
             'mask_rcnn_inception_v2_coco': 'mask_rcnn_inception_v2_coco_2018_01_28'}


def test(test_name):
    """
    Main function for running tests on model zoo.
    :param test_name: String, relative directory path.
    :return: Nothing returned, results saved to test results directory.
    """
    ##################################################
    # Set up directory
    test_path = os.path.join(args.dir, test_name)
    day_dir = os.listdir(test_path)
    day_dir = [d for d in day_dir if not d.startswith('.')][0]
    full_path = os.path.join(test_path, day_dir)

    print('\n---------------------------------------------------------'
          '\n\tCommencing testing at {}...'.format(full_path))

    ##################################################
    # Set up time and date
    base_path = os.path.basename(full_path)
    start_datetime = datetime.strptime(base_path + ' ' + start_time, '%d-%m-%Y %H:%M')
    print("> Start date is {}"
          "\n> Start time is {}".format(base_path, start_time))

    ##################################################
    # Make list of frames
    images = sorted([image for image in os.listdir(full_path) if not image.startswith('.')])
    image_count = len(images)
    print('> Found {} images'.format(image_count))
    print('\t>>>Frame rate assumed to be {} seconds per frame'.format(frame_step))

    ##################################################
    # Prepare model loop
    model_list = []
    test_results = []
    performances_dict = {}

    for model_name, model_path in model_zoo.items():
        try:
            ##################################################
            # Prepare loop
            MODEL_NAME = model_zoo.get(model_name)

            detection_graph = tracker_utils.load_model(MODEL_NAME)

            print('Object detection with {}'.format(model_name))

            background = None  # Initialise with no background
            time_stamp_dict = {}  # Dictionary for timestamps
            result_array = np.empty((0, 10))  # Results array
            detection_times = []  # Detection timer array

            with detection_graph.as_default():
                with tf.Session() as sess:
                    print("Commencing model test...")

                    for image in images:
                        if args.verbose:
                            print('Image: {}'.format(image))

                        # Extract frame number from image path
                        image_index = [int(s) for s in list(image) if s.isdigit()]
                        image_index = ''.join(str(x) for x in image_index)
                        image_index = int(image_index)

                        # Calculate time stamp of frame
                        time_delta = timedelta(seconds=(frame_step * image_index))
                        time_stamp = start_datetime + time_delta
                        time_stamp_dict[image_index] = time_stamp  # Add to dictionary

                        image_path = os.path.join(full_path, image)
                        img = cv2.imread(image_path)
                        img = cv2.resize(img, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)  # Resize to 25% area

                        # Trigger ##############################################
                        thumb = tracker_utils.trigger_image(img)

                        if tracker_utils.motion_trigger(args.sensitivity, thumb, background):

                            timer_start = datetime.now()

                            # If trigger then run object detection
                            detections_dict = tracker_utils.run_object_detection(sess, img)

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
                    string = test_dir + '_' + model_name + '.npy'
                    print('\n>>> Saving {} detections as {}'.format(len(result_array), string))
                    np.save(os.path.join(_results_path, string), result_array)

                    if args.display:
                        cv2.destroyAllWindows()

                    print("Test results:")

                    if detection_times:
                        triggers = len(detection_times)
                        detections = len(result_array)
                        total = 0.
                        for time in detection_times:
                            total += time.microseconds / 1000000
                            total += time.seconds
                        av_time = total / triggers
                        print('\t>>> {} triggers'
                              '\n\t>>> {} objects detected'
                              '\n\t>>> {} seconds per trigger'.format(triggers, detections, av_time))

                        performance_dict = {'triggers': triggers, 'av_time': av_time}
                        performances_dict[model_name] = performance_dict

                        test_results.append([detections, av_time])  # TODO clean up dict vs lists for performance results

                        model_list.append(MODEL_NAME)

        except Exception as e:
            print(e)
            print("\n\t>>> Failed to use model: {} <<<\n".format(model_name))
            continue

    string = 'performance_' + test_dir + '.npy'
    print('\n>>> Saving detection performance as {}'.format(string))
    np.save(os.path.join(_results_path, string), performances_dict)

    print("-------------------------------------\n\t>>>RESULTS:\n")
    col_width = 50
    for model, (detections, av_time) in zip(model_list, test_results):
        print(model.ljust(col_width) + ('Detections: ' + str(detections)).ljust(col_width) + (
            'Seconds: ' + str(av_time)).ljust(col_width))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=default_test, help='Frame directory')
    parser.add_argument('--save', default='.', help='Set location of results')
    parser.add_argument('--sensitivity', default=0.0001, help='Sensitivity of motion trigger')
    parser.add_argument('-D', '--display', action='store_true', help='Display video of detections')
    parser.add_argument('-V', '--verbose', action='store_true')

    args = parser.parse_args()

    # Set up results dir
    results_path = os.path.join(args.save, 'test_results_' + str(datetime.now().strftime("%Y-%m-%d")))
    _results_path = results_path
    i = 0

    while os.path.exists(_results_path):
        i += 1
        _results_path = results_path + '_' + str(i)

    os.mkdir(_results_path)
    print('\n---------------------------------------------------------')
    print('\nSaving results to {}'.format(_results_path))
    print('Accessing images from {}'.format(args.dir))
    print('Trigger sensitivity set to {}'.format(args.sensitivity))

    # get test directories

    test_directories = os.listdir(args.dir)
    test_directories = sorted([d for d in os.listdir(args.dir) if not d.startswith('.')])[3:]
    print('\nExtracting data from:')
    for test_dir in test_directories:
        print('\t> {}'.format(test_dir))

    ##################################################
    # Prepare paths
    print('\nPreparing detection paths...')
    PATH_TO_LABELS = os.path.join('object_detection', 'data', 'mscoco_label_map.pbtxt')
    NUM_CLASSES = 10

    # Load label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    for test_dir in test_directories:

        test(test_dir)







