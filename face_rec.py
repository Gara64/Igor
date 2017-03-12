#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

import logging
import cv2
import time
import sys
import json
import random
sys.path.append("lib")

from facerec.model import PredictableModel
from facerec.feature import Fisherfaces
from facerec.distance import EuclideanDistance
from facerec.classifier import NearestNeighbor
from facerec.validation import KFoldCrossValidation
from facerec.serialization import save_model, load_model
# for face detection (you can also use OpenCV2 directly):
from facedet.detector import CascadedDetector
# helpers
from helper.common import *
from helper.video import *

# actions
from actions.actions import *


MAX_DISTANCE = 900          # Minimal euclidean distance to recognize someone
DELAY_RECO_SAME_PERSON = 5  # Delay (seconds) between 2 same person recognition
SAVE_PIC = False            # Boolean to save or not the image at the recognition

# Subclasses the PredictableModel to store some more information, so we don't
# need to pass the dataset on each program call...
class ExtendedPredictableModel(PredictableModel):

    def __init__(self, feature, classifier, image_size, subject_names):
        PredictableModel.__init__(self, feature=feature, classifier=classifier)
        self.image_size = image_size
        self.subject_names = subject_names


# Get the PrediictableModel, used to learn a model
def get_model(image_size, subject_names):
    # Define the Fisherfaces Method as Feature Extraction method:
    feature = Fisherfaces()
    # Define a 1-NN classifier with Euclidean Distance:
    classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)

    return ExtendedPredictableModel(feature=feature, classifier=classifier, image_size=image_size, subject_names=subject_names)



class App(object):

    # Dict to associate the models names with detection timestamps
    lastFaceDetect = {}

    def __init__(self, model, camera_id, cascade_filename, people, sentences, photoPath):
        self.model = model
        self.detector = CascadedDetector(cascade_fn=cascade_filename, minNeighbors=5, scaleFactor=1.4)
        self.cam = create_capture(camera_id)

        buildSentences(sentences)
        buildPeople(people)

        # TODO : implements photo save at detection


    # Get a face from the image
    def getFace(self, img, r):
        # (1) Get face, (2) Convert to grayscale & (3) resize to image_size:
        x0,y0,x1,y1 = r
        face = img[y0:y1, x0:x1]
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, self.model.image_size, interpolation = cv2.INTER_CUBIC)

        return face



    # Returns true if the predicted face is recognized
    def recognize(self, predict):
        indexName = predict[0]
        distance = predict[1]['distances']

        print "Distance : " + str(distance)

        # The lower the distance, the better the prediction
        if distance < MAX_DISTANCE:
            print  "Prediction : " + self.model.subject_names[indexName]

            # Get the face name from the model
            faceName = self.model.subject_names[indexName]
            curTime = time.mktime(time.localtime())

            # Check the last detect for this face was after the minimal delay
            if faceName in self.lastFaceDetect:
                delay = curTime - self.lastFaceDetect[faceName]
                if  delay <  DELAY_RECO_SAME_PERSON:
                    return [None, None, None]
            else:
                delay = 0
            self.lastFaceDetect[faceName] = curTime

            return [faceName, curTime, delay]

        else:
            return [None, None, None]


    # Run the face detection
    def run(self):
        cpt = 0
        while True:
            ret, frame = self.cam.read()
            width = frame.shape[1] / 2
            height = frame.shape[0] / 2

            # Resize the frame to half the original size for speeding up the detection process:
            img = cv2.resize(frame, (width, height), interpolation = cv2.INTER_CUBIC)
            imgout = img.copy()

            # For each frame with a detected face
            for i,r in enumerate(self.detector.detect(img)):

                # Get the face from image
                face = self.getFace(img, r)

                # Get a model prediction from the face
                predict = self.model.predict(face)

                # Let's recognize the face
                [faceName, curTime, delay] = self.recognize(predict)

                # Perform action if face is recognized
                # TODO move this in lib/actions
                if faceName:
                    actionAfterRecognition(faceName, self.lastFaceDetect, curTime, delay)

                if SAVE_PIC:
                    cv2.imwrite(self.photoPath + "/img-" + str(i) + ".jpg", img)
                    cv2.imwrite(self.photoPath + "/face-" + str(i) + ".jpg", face)
                    print "Image saved"


if __name__ == '__main__':
    from optparse import OptionParser
    # model.pkl is a trained PredictableModel, which is used to make predictions.
    # A model can be created by running create_model.py
    usage = "usage: %prog [options] model_filename"

    # Add options for training, resizing, validation and setting the camera id:
    parser = OptionParser(usage=usage)

    parser.add_option("-r", "--resize", action="store", type="string", dest="size", default="100x100",
        help="Resizes the given dataset to a given size in format [width]x[height] (default: 100x100).")

    parser.add_option("-i", "--id", action="store", dest="camera_id", type="int", default=0,
        help="Sets the Camera Id to be used (default: 0).")

    parser.add_option("-c", "--cascade", action="store", dest="cascade_filename", default="haarcascade_frontalface_alt2.xml",
        help="Sets the path to the Haar Cascade used for the face detection part (default: haarcascade_frontalface_alt2.xml).")

    parser.add_option("-p", "--photo", action="store", dest="photo_path",
        help="Store the picture taken for each detection.")

    # Parse arguments:
    (options, args) = parser.parse_args()

    # Show the options to the user:
    if len(args) == 0:
        parser.print_help()
        sys.exit()

    model_filename = args[0]

    # Check if the given (or default) cascade file exists:
    if not os.path.exists(options.cascade_filename):
        print "[Error] No Cascade File found at '%s'." % options.cascade_filename
        sys.exit()

    # We are resizing the images to a fixed size, as this is neccessary for some of
    # the algorithms, some algorithms like LBPH don't have this requirement. To
    # prevent problems from popping up, we resize them with a default value if none
    # was given:
    try:
        image_size = (int(options.size.split("x")[0]), int(options.size.split("x")[1]))
    except:
        print "[Error] Unable to parse the given image size '%s'. Please pass it in the format [width]x[height]!" % options.size
        sys.exit()

    print "Loading the model..."
    model = load_model(model_filename)

    # Check the model is an ExtendedPredictableModel
    if not isinstance(model, ExtendedPredictableModel):
        print "[Error] The given model is not of type '%s'." % "ExtendedPredictableModel"
        sys.exit()

    people = {}
    sentences = {}

    # Read the config files
    with open("config/people.json") as f:
        people = json.load(f)
        people = people["people"]
    f.closed
    with open("config/sentences.json") as f:
        sentences = json.load(f)
        sentences = sentences["sentences"]
    f.closed

    # Actve the photo shooting
    if options.photo_path:
        print "path : " + options.photo_path
        SAVE_PIC = True

    # Start the Application based on the given model
    print "Starting application..."
    App(model=model,
        camera_id=options.camera_id,
        cascade_filename=options.cascade_filename,
        people=people,
        sentences=sentences,
        photoPath=options.photo_path
    ).run()
