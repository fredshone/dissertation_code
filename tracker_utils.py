"""General utility functions for tracking"""
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import numpy as np
import cv2
from object_detection.utils import ops as utils_ops


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# Load Frozen model into memory
def load_model(model_name, location='object_detection'):
    """
    Loads frozen model into memory. If model not found then will try to download model.
    :param model_name: String, model name.
    :param location: Sting, relative directory path to frozen models.
    :return: Returns frozen model graph.
    """
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    ckpt_path = os.path.join(location, model_name, 'frozen_inference_graph.pb')

    # Download if necessary
    if not os.path.isfile(ckpt_path):
        print('>>> Cannot find frozen model')
        download_model(model_name, location)

    print('Loading frozen model: {}'.format(ckpt_path))
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(ckpt_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print('Done')
    return detection_graph


# Download Model
def download_model(model_name, location='object_detection'):
    """
    Downloads model. Requires internet connection.
    :param model_name: String, model name.
    :param location: Sting, relative directory path to frozen models.
    :return: Nothing. Mode downloaded to directory.
    """
    model_file = model_name + '.tar.gz'
    download_base = 'http://download.tensorflow.org/models/object_detection'
    download_path = os.path.join(download_base, model_file)
    model_path = os.path.join(location, model_file)

    # Download if necessary
    if not os.path.isfile(model_path):
        print('Downloading: {}'.format(model_file))
        opener = urllib.request.URLopener()
        opener.retrieve(download_path, model_path)

        print('Extracting frozen inference graph from: {}'.format(model_path))
        tar_file = tarfile.open(model_path)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.path.join(os.getcwd(), location))


def trigger_image(img):
    """
    Reduces image size to 1% and processes to B&W.
    :param img: Input.
    :return: Image.
    """
    # Reduce image size to 1%
    thumb = cv2.resize(img, None, fx=.1, fy=.1, interpolation=cv2.INTER_AREA)
    thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)

    return thumb


def motion_trigger(sensitivity, thumb, background):
    """
    Boolean trigger for image motion. Calculates pixel difference between frame and previous frame then returns
    True if difference is above given sensitivity value.
    :param sensitivity: Integer. threshold for image difference trigger.
    :param thumb: Image, current frame.
    :param background: Image, previous frame.
    :return: Boolean. True for trigger.
    """
    if background is None:  # Initialise background for first pass
        background = thumb

    delta = cv2.absdiff(background, thumb)
    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
    pixel_diff = cv2.sumElems(thresh)[0]

    if pixel_diff > thumb.size * sensitivity:
        return True
    else:
        return False


# Object Detection
def run_object_detection(session, image):
    """
    Main function for running object detection on images.
    :param session: Tensorflow Session.
    :param image: Image.
    :return: Returns dictionary of detection results.
    """
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}

    for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Re frame is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = session.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

    return output_dict

