# coding=utf-8
 
from __future__ import print_function
import os
import sys
import time

import cv2
import numpy as np
import weave

#if necessary, use custom caffe build
#sys.path.insert(0, 'build/install/python')
sys.path.insert(0, '/home/ndr/Downloads/caffe/install_new/python/')
import caffe


def id2bgr(im):
 
    w, h = im.shape
    color_image = np.empty((w, h, 3), dtype=np.uint8)
     
    weave.inline(code, ["im", "color_image"])
    return color_image

def trainID2labelID(im):
    """
    A fast conversion from train id to label id.
    :param im: 2d array with shape (w,h) with recognized object train IDs as pixel values
    :return: greyscale_image: pixel value is label id
    The greyscale values are the same as in CityScapes dataset:
    github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    """
    w, h = im.shape
    greyscale_image = np.empty((w, h), dtype=np.uint8)
    code = """
    unsigned char label_ids[19] = {7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33};
   
    int impos=0;
    int retpos=0;
    for(int j=0; j<Nim[0]; j++) {
        for (int i=0; i<Nim[1]; i++) {
            unsigned char d=im[impos++];
            greyscale_image[retpos++] = label_ids[d];
        }
    }
    """
    weave.inline(code, ["im", "greyscale_image"])
    return greyscale_image

def fast_mean_subtraction_bgr(im):
    """
    Fast mean subtraction
    :param im: input image
    :return: image with subtracted mean values of ImageNet dataset
    """
    code = """
    float mean_r = 123;
    float mean_g = 117;
    float mean_b = 104;
    int retpos=0;
    for(int j=0; j<Nim[0]; j++) {
        for (int i=0; i<Nim[1]; i++) {
            im[retpos++] -=  mean_b;
            im[retpos++] -=  mean_g;
            im[retpos++] -=  mean_r;
        }
    }
    """
    weave.inline(code, ["im"])
    return im


def feed_and_run(input_frame):
    """
    Format input data and run object recognition 
    :param input_frame: image data from file
    :return: forward_time, segmentation_result
    """
    start = time.time()
    input_frame = np.array(input_frame, dtype=np.float32)
    input_frame = fast_mean_subtraction_bgr(input_frame)
    input_frame = input_frame.transpose((2, 0, 1))
    net.blobs['data'].data[...] = input_frame
    print("Data input took {} ms.".format(round((time.time() - start) * 1000)))

    start = time.time()
    net.forward()
    forward_time = round((time.time() - start) * 1000)
    print("Net.forward() took {} ms.".format(forward_time))

    start = time.time()

  
    start = time.time()
    greyscale_segmentation_result = trainID2labelID(result_with_train_ids)
    print("Conversion from train ID to label ID took {} ms.".format(round((time.time() - start) * 1000)))

    start = time.time()
    segmentation_result = id2bgr(result_with_train_ids)
    print("Conversion from object class ID to color took {} ms.".format(round((time.time() - start) * 1000)))

    return forward_time, segmentation_result, greyscale_segmentation_result


if __name__ == "__main__":

    # input_dir, results_dir should be the same as in set_env.sh
    input_dir   =  os.environ['CITYSCAPES_DATASET_IN']
    #input_dir = '/home/ndr/work/Datasets/cityscapes/images/leftImg8bit/val'

    results_dir =  os.environ['CITYSCAPES_RESULTS']
    #results_dir = '/home/ndr/work/Datasets/cityscapes/results/val'

    input_images = []
    for dataset_dir, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".png"):
                input_images.append(os.path.join(dataset_dir, file))

    #------------------------ Change main parameters here ---------------

    model_weights = './models/weights.caffemodel'

    # original model
    # model_description = './models/deploy_model_with_upsampling_without_argmax.prototxt'
    # model_has_argmax = False

    # improved 1
    model_description = './models/deploy_model_with_upsampling_with_argmax.prototxt'
    model_has_argmax = True

    # improved 2
    # model_description = './models/deploy_model_without_upsampling_without_argmax.prototxt'
    # model_has_argmax = False

    # improved 3
    # model_description = './models/deploy_model_without_upsampling_with_argmax.prototxt'
    # model_has_argmax = True

    show_gui = True
    save_image_result = True
    createVideoFromResults = True

    input_w = 2048
    input_h = 1024

    greyscale_image_results_folder = results_dir
    color_image_results_folder = './results_in_color/'

    if not os.path.exists(greyscale_image_results_folder):
        os.makedirs(greyscale_image_results_folder)

    if not os.path.exists(color_image_results_folder):
        os.makedirs(color_image_results_folder)


    #--------------------------------------------------------------------
 
    writer = None
    if createVideoFromResults:
        fps = 30
        codec = 'mp4v'
        videoFileName = 'result_val.mkv'
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(videoFileName, fourcc, fps, (input_w, input_h))

    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(model_description, 1, weights=model_weights)

    result_out_upscaled = np.empty((input_h, input_w, 3), dtype=np.uint8)
    # transparency of the overlaid object segments
    alpha = 0.7
    blended_result = np.empty((input_h, input_w, 3), dtype=np.uint8)

    if show_gui:
        cv2.namedWindow("Classification results")

    num_images_processed = 0
    for image in input_images:     # main loop

        initial_time = time.time()
        start = time.time()
        frame = cv2.imread(image)

        print("Input image file reading time: {} ms.".format(round((time.time() - start) * 1000)))

        core_forward_time, recognition_result, greyscale_recognition_result = feed_and_run(frame)

        start = time.time()
        #result_out_upscaled = cv2.resize(recognition_result, (input_w, input_h), interpolation=cv2.INTER_NEAREST)
        result_out_upscaled = recognition_result

        print("Resize time: {} ms.".format(round((time.time() - start) * 1000)))

        start = time.time()
        cv2.addWeighted(result_out_upscaled, alpha, frame, 1.0 - alpha, 0.0, blended_result)
        print("Overlay detection results: {} ms.".format(round((time.time() - start) * 1000)))

        start = time.time()

        if show_gui:
            cv2.imshow("Classification results", blended_result)

        print("cv2.imshow time: {} ms.".format(round((time.time() - start) * 1000)))

        if save_image_result:
            start = time.time()
            cv2.imwrite(color_image_results_folder + os.path.basename(image), blended_result)
            print("cv2.imwrite time for color image result: {} ms.".format(round((time.time() - start) * 1000)))

            start = time.time()
            cv2.imwrite(greyscale_image_results_folder + os.path.basename(image), greyscale_recognition_result)
            print("cv2.imwrite time for greyscale image result: {} ms.".format(round((time.time() - start) * 1000)))

        if createVideoFromResults:
            start = time.time()
            writer.write(blended_result)
            print("Add frame to video file: {} ms.".format(round((time.time() - start) * 1000)))

        key = cv2.waitKey(1)
        if key == 27:  # exit on ESC
            break

        num_images_processed += 1

        print("Total time with data i/o and image pre/post postprocessing - {} ms.".format(
            round((time.time() - initial_time) * 1000)))
        print("---------> Finished processing image #{}, {}, net.forward() time: {} ms.".format(num_images_processed,
                                                                                         os.path.basename(image),
                                                                                         core_forward_time))
    if createVideoFromResults:
        writer.release()

    if show_gui:
        cv2.destroyWindow("Classification results")

 

