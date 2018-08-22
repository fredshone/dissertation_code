import argparse
import os
import urllib.request
import re
from collections import OrderedDict
from datetime import datetime
from pytube import YouTube
import numpy as np

import cv2

# Object Detection
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import tracker_utils

# Object Association
import association_utils


detection_model_zoo = {'ssd_mobilenet_v1_coco': 'ssd_mobilenet_v1_coco_2018_01_28',
             'ssd_mobilenet_v1_0.75_depth_coco': 'ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03',
             'ssd_mobilenet_v1_quantized_coco': 'ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_03',
             'ssd_mobilenet_v1_0.75_depth_quantized_coco': 'ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_03',
             'ssd_mobilenet_v1_ppn_coco': 'ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03',
             'ssd_mobilenet_v1_fpn_coco': 'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
             'ssd_resnet_50_fpn_coco': 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
             'ssd_mobilenet_v2_coco': 'ssd_mobilenet_v2_coco_2018_03_29',
             'ssdlite_mobilenet_v2_coco': 'ssdlite_mobilenet_v2_coco_2018_05_09',
             'ssd_inception_v2_coco': 'ssd_inception_v2_coco_2018_01_28',
             'faster_rcnn_inception_v2_coco': 'faster_rcnn_inception_v2_coco_2018_01_28',
             'faster_rcnn_resnet50_coco': 'faster_rcnn_resnet50_coco_2018_01_28',
             'faster_rcnn_resnet50_lowproposals_coco': 'faster_rcnn_resnet50_lowproposals_coco_2018_01_28',
             'rfcn_resnet101_coco': 'rfcn_resnet101_coco_2018_01_28',
             'faster_rcnn_resnet101_coco': 'faster_rcnn_resnet101_coco_2018_01_28',
             'faster_rcnn_resnet101_lowproposals_coco': 'faster_rcnn_resnet101_lowproposals_coco_2018_01_28',
             'mask_rcnn_inception_v2_coco': 'mask_rcnn_inception_v2_coco_2018_01_28'}

default_detection_model = 'ssd_mobilenet_v2_coco'

association_model_zoo = {'pixel_feature_matching': 'pixel_feature_matching.hdf5',
                         'complex_feature_matching': 'complex_feature_matching.hdf5',
                         'transfer_feature_matching': None}

default_association_model = 'complex_feature_matching'


def get_args():
    default_playlist = "https://www.youtube.com/watch?v=f5V-cH-9udI&list=UUDDgDE-EMc-tSyp7xy4Pk9w"
    # default_start = datetime.strptime('2015-10-28 00:00:00', '%Y-%m-%d %H:%M:%S').date()
    default_start = '2015-10-28'

    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_model', default=default_detection_model, help="Choose detection model")
    parser.add_argument('--association_model', default=default_association_model, help="Choose association model")
    parser.add_argument('--dir', help='Directory path')
    parser.add_argument('--url', default=default_playlist, help='Video playlist')
    parser.add_argument('--save', default='results', help='Set location of results')
    parser.add_argument('--start', default=default_start, help='Video start date')
    parser.add_argument('--sensitivity', default=0.0001, help='Sensitivity of motion trigger')
    parser.add_argument('-D', '--display', action='store_true', help='Display video of detections')
    parser.add_argument('-V', '--verbose', action='store_true')
    args = parser.parse_args()

    # Set up results dir
    models_path = '_' + args.detection_model + '__' + args.association_model
    if args.dir:
        string = '_' + os.path.basename(args.dir) + '_'
    else:
        string = '_'
    _results_path = os.path.join(args.save, 'tracker' + string + str(datetime.now().strftime("%Y-%m-%d")) + models_path)
    results_path = _results_path
    i = 0
    while os.path.exists(results_path):
        i += 1
        results_path = _results_path + '_' + str(i)
    os.mkdir(results_path)

    print('\n---------------------------------------------------------')
    if args.dir:
        print('Accessing images from {}'.format(args.dir))
    elif args.url:
        print('Accessing images from {}'.format(args.url))
    if args.detection_model:
        print('Detection model: {}'.format(args.detection_model))
    else:
        print('Detection model: {}'.format(default_detection_model))
    if args.association_model:
        print('Association model: {}'.format(args.association_model))
    else:
        print('Association model: {}'.format(default_association_model))
    print('Saving results to {}'.format(_results_path))

    return args, results_path


def yield_directory():
    # Set up directory
    img_dir = os.path.join(args.dir, 'img1')
    img_names = os.listdir(img_dir)
    img_names = sorted([d for d in img_names if not d.startswith('.')])
    ret = True
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)

        # Extract frame number from image path
        image_index = [int(s) for s in list(img_name) if s.isdigit()]
        image_index = ''.join(str(x) for x in image_index)
        image_index = int(image_index)

        yield ret, image_index, img


def get_youtube_paths(playlist_url):

    final_url = []
    eq = playlist_url.rfind('=') + 1
    cPL = playlist_url[eq:]
    yTUBE = urllib.request.urlopen(playlist_url).read()
    sTUBE = str(yTUBE)
    tmp_mat = re.compile(r'watch\?v=\S+?list=' + cPL)
    mat = re.findall(tmp_mat, sTUBE)
    for PL in mat:
        yPL = str(PL)
        if '&' in yPL:
            yPL_amp = yPL.index('&')
        final_url.append('http://www.youtube.com/' + yPL[:yPL_amp])

    all_url = list(OrderedDict.fromkeys(final_url))

    video_dict = {}

    for url in all_url:
        yt = YouTube(url)
        title = yt.title
        date_string = re.findall("(\d+\-\d+\-\d+)", title)
        date = datetime.strptime(date_string[0], "%d-%m-%Y")
        video_dict[date] = yt.streams.first().url

    video_dates = sorted(list(video_dict.keys()))

    if args.verbose:
        for date in video_dates:
            print(date)
            print(video_dict[date])

    return video_dates, video_dict


def yield_playlist(date):
    img_index = -1
    url = url_dict.get(date, False)
    if args.verbose:
        print('>>>Reading video from:')
        print(url)
    if url:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print('\n>>> Failed to read {}'.format(date))
            yield None
        else:
            ret = True
            if args.verbose:
                print('\n>>> Reading frames from {}'.format(date))
            while ret:
                ret, frame = cap.read()
                img_index += 1
                if args.verbose:
                    print(img_index)
                yield ret, img_index, frame

            cap.release()


def loop_through_frames(date, results_path):
    print('\n\tDate: {}'.format(date))
    ##################################################
    # Prepare loop
    new_instance_id = 0
    background = None  # Initialise with no background
    empty = np.empty((0, 10))  # Results array
    result_array = empty
    detection_times = []  # Detection timer array
    detection_history = {-1: empty, -2: empty, -3: empty}
    img_history = {-1: [], -2: [], -3: []}

    if args.dir:
        frames_gen = yield_directory()
        print('\nAccessing frames from local directory: {}'.format(args.dir))
    elif args.url:
        frames_gen = yield_playlist(date)
        print('\nAccessing frames from Youtube playlist for : {}'.format(date))

    if frames_gen:
        for returned, img_index, frame in frames_gen:

            try:
                img = cv2.resize(frame, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)  # Resize to 25% area
                img_instances = []
                frame_results_array = empty

                # Trigger ##############################################
                thumb = tracker_utils.trigger_image(img)

                if tracker_utils.motion_trigger(args.sensitivity, thumb, background):

                    timer_start = datetime.now()

                    # If trigger then run object detection
                    detections_dict = tracker_utils.run_object_detection(sess, img)

                    if args.verbose:
                        print('Motion detection at frame {}'.format(img_index))

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
                        if not num_detections:
                            if args.verbose:
                                print('No detections')
                        if num_detections:
                            detection_classes = detections_dict['detection_classes']
                            detection_scores = detections_dict['detection_scores']
                            mask = (detection_classes == 1) & (detection_scores > detection_threshold)
                            detection_boxes = detections_dict['detection_boxes'][mask]
                            detection_scores = detection_scores[mask]
                            if args.verbose:
                                print('{} detections'.format(len(detection_scores)))

                            height, width = img.shape[:2]
                            for counter, (detection, confidence) in enumerate(zip(detection_boxes, detection_scores)):
                                if args.verbose:
                                    print('\n\tDetection: {}'.format(counter + 1))
                                y_min, x_min, y_max, x_max = detection
                                bb_top = int(y_min * height)
                                bb_left = int(x_min * width)
                                bb_height = int((y_max * height) - bb_top)
                                bb_width = int((x_max * width) - bb_left)

                                ##################################################
                                #  Check for matches
                                # retrieve instance image
                                img_instance = img[bb_top:(bb_top + bb_height), bb_left:(bb_left + bb_width)]
                                # convert to numpy array for inference
                                img_instance = cv2.resize(img_instance, (candidate_image_size[0], candidate_image_size[1]))
                                img_instance = img_instance.astype('float') / 255
                                img_instance = np.expand_dims(img_instance, axis=0)
                                img_instances.append(img_instance)

                                set_new_id = True  # Default to adding new id unless match can be found below

                                # loop through histories
                                history_order = [-1, -2, -3]
                                for key in history_order:
                                    hist_detections = detection_history.get(key)
                                    hist_detections_img = img_history.get(key)
                                    if not len(hist_detections):
                                        if args.verbose:
                                            print('No detections found in history {}, continue'.format(key))
                                        continue

                                    # loop through detections in history
                                    if args.verbose:
                                        print('Found {} detections and {} images in detection history {}'.
                                              format(len(hist_detections), len(hist_detections_img), key))

                                    match_likelihoods = []
                                    for d, img_candidate in enumerate(hist_detections_img):

                                        # make list of matches and confidence
                                        likelihood = association_model.predict(img_instance, img_candidate)  # TODO
                                        match_likelihoods.append(likelihood)

                                    # choose highest confidence > 0.5
                                    if max(match_likelihoods) >= association_threshold:
                                        set_new_id = False
                                        if args.verbose:
                                            print('Match found in history:')
                                            print(hist_detections.astype(int))
                                        max_likelihood = -1
                                        for l_index, likelihood in enumerate(match_likelihoods):
                                            if args.verbose:
                                                print(likelihood)
                                            if likelihood > max_likelihood:
                                                max_likelihood = likelihood
                                                match_index = l_index
                                                match_id = hist_detections[match_index][1]

                                        # Remove from histories
                                        detection_history[key] = np.delete(hist_detections, match_index, 0)
                                        hist_detections_img.pop(match_index)
                                        img_history[key] = hist_detections_img
                                        if args.verbose:
                                            print('Match found, id:{} from history: {}'.format(match_id, key))

                                        break

                                if set_new_id:
                                    match_id = new_instance_id
                                    new_instance_id += 1
                                    if args.verbose:
                                        print('No match found, adding new id:{}'.format(match_id))

                                result = img_index, match_id, bb_left, bb_top, bb_width, bb_height, confidence, -1, -1, -1
                                frame_results_array = np.append(frame_results_array, [result], axis=0)
                                result_array = np.append(result_array, [result], axis=0)

                    timer = datetime.now() - timer_start
                    detection_times.append(timer)

                background = thumb  # Set background for next frame

                # Update history
                detection_history[-3] = detection_history[-2]
                detection_history[-2] = detection_history[-1]
                if len(frame_results_array):
                    detection_history[-1] = frame_results_array
                    if args.verbose:
                        print('\nAdding {} detections to history'.format(len(frame_results_array)))
                else:
                    detection_history[-1] = empty

                img_history[-3] = img_history[-2]
                img_history[-2] = img_history[-1]
                if len(frame_results_array):
                    img_history[-1] = img_instances
                    if args.verbose:
                        print('\nAdding {} images to history'.format(len(img_instances)))
                else:
                    img_history[-1] = []

            except:
                break

        # Save results
        results_name = str(date.date()) + '.npy'
        print('\n>>> Saving {} detections as {}'.format(len(result_array), results_name))
        np.save(os.path.join(results_path, results_name), result_array)
        print('...saved to {}'.format(os.path.join(results_path, results_name)))
        # Save results txt
        results_name = str(date.date()) + '.txt'
        print('\n>>> Saving {} detections as {}'.format(len(result_array), results_name))
        np.savetxt(os.path.join(results_path, results_name), result_array, fmt='%i', delimiter=",")
        print('...saved to {}'.format(os.path.join(results_path, results_name)))

        if args.display:
            cv2.destroyAllWindows()

        print("\n\tTest results:")

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

            string = 'performance_' + str(date.date()) + '.npy'
            print('\n>>> Saving detection performance as {}'.format(string))
            np.save(os.path.join(results_path, string), performance_dict)
            print('...saved to {}'.format(os.path.join(results_path, string)))


if __name__ == "__main__":
    args, results_path = get_args()

    ##################################################
    # Prepare tool
    start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
    start_time = '08:00'  # Assumes camera turned on at 8am every day
    start_time = datetime.strptime(start_time, '%H:%M').time()
    frame_step = 3  # Seconds
    detection_threshold = 0.5
    association_threshold = 0.5

    print('\nPreparing tool variables:'
          '\n\tStart date: {}'
          '\n\tStart time: {}'
          '\n\tFrame step: {}'
          '\n\tTrigger sensitivity: {}'
          '\n\tDetection threshold: {}'
          '\n\tAssociation threshold: {}'.
          format(start_date,
                 start_time,
                 frame_step,
                 args.sensitivity,
                 detection_threshold,
                 association_threshold))

    ##################################################
    # Prepare detection
    print('\nPreparing detection model')
    DETECTION_MODEL_NAME = detection_model_zoo.get(args.detection_model, False)

    if not DETECTION_MODEL_NAME:
        raise Exception('Unrecognised detection model, choose:'
                        '\n{}'.format(detection_model_zoo))

    print('\nPreparing detection paths...')
    PATH_TO_LABELS = os.path.join('object_detection', 'data', 'mscoco_label_map.pbtxt')
    NUM_CLASSES = 10

    # Load label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    ##################################################
    # Prepare association
    print('\nPreparing association model')
    ASSOCIATION_MODEL_NAME = association_model_zoo.get(args.association_model, False)
    if not ASSOCIATION_MODEL_NAME:
        raise Exception('Unrecognised association model, choose:'
                        '\n{}'.format(association_model_zoo))

    association_model_path = os.path.join('object_association', ASSOCIATION_MODEL_NAME)

    candidate_image_size = (224, 224, 3)

    association_model = association_utils.ObjectAssociationModel(association_model_path)
    print('Association model loaded from {}'.format(association_model_path))

    ##################################################
    # Prepare image paths
    print('\nPreparing video paths')

    if args.dir:
        dates = [datetime.strptime('2015-01-03 00:00:00', '%Y-%m-%d %H:%M:%S')]
    elif args.url:
        dates, url_dict = get_youtube_paths(args.url)
    else:
        raise Exception('No image or video path supplied')

    ##################################################

    detection_graph = tracker_utils.load_model(DETECTION_MODEL_NAME)
    print('\nCommencing operation:')
    for date in dates:
        print(date)
    with detection_graph.as_default():
        with tf.Session() as sess:
            # Loop through days
            for date in dates:
                if date.date() >= start_date:
                    loop_through_frames(date, results_path)
