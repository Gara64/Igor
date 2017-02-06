#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

import logging
import cv2
import time
import sys
sys.path.append("lib")

# facerec imports
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
# text to speech
from voice.text2speech import textToSpeechGoogle


MAX_DISTANCE = 850          # Minimal euclidean distance to recognize someone
DELAY_RECO_SAME_PERSON = 5  # Delay (seconds) between 2 same person recognition


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

    # Dict to store faces detection timestamps
    lastFaceDetect = {}

    def __init__(self, model, camera_id, cascade_filename):
        self.model = model
        self.detector = CascadedDetector(cascade_fn=cascade_filename, minNeighbors=5, scaleFactor=1.4)
        self.cam = create_capture(camera_id)

    # Get a face from the image
    def getFace(self, img, r):
        # (1) Get face, (2) Convert to grayscale & (3) resize to image_size:
        x0,y0,x1,y1 = r
        face = img[y0:y1, x0:x1]
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, self.model.image_size, interpolation = cv2.INTER_CUBIC)

        return face

    # Gender convertion for speech after recognition
    def convertGender(faceName, text):
        gender = 0 if faceName is "Pauline" else 1

        if text is "maitre":
            if gender is 0: text = "maitresse"
        elif text is "travailleur":
            if gender is 0: text = "travailleuse"

        return text

    # Action after recognition
    def actionAfterRecognition(self, faceName, curTime, delay):
        # TODO : the basic hello is boring. make an algo to make it smarter
        # eg say hello once in the day
        # say random sentence / surname, depending on the time, weather, ..
        # Idea : voice mail -> drop messages to say when SO is back a home

        # NOTE : the speech orthograph is sometimes awful: this is because the
        # the Google translate script does not support the accents very well

        delayHours = delay / 3600
        hourOfDay = time.localtime(curTime).tm_hour
        dayOfWeek =  time.localtime(curTime).tm_wday

        # The SO was detected few seconds before
        SO = "Paul" if faceName is "Pauline" else "Pauline"
        if SO in self.lastFaceDetect:
            lastSODetection = curTime - self.lastFaceDetect[SO]
            if lastSODetection < 10:
                text = "Vous egalement, " + self.convertGender('maitre') + " " + faceName

        # It's been a long time since the last detect
        elif delayHours > 40:
            text = "Vous revoila + " + faceName + " ! Vous m'avez manquai"

        # Why are you still there ?
        elif delay > 0 and delay < 10:
            text = "Pourquoi restez-vous là, " + self.convertGender('maitre') + " ?"

        # Man, it's pretty late and we're during the week
        elif hourOfDay > 21 and delayHours > 10 and dayOfWeek < 4:
            text = "Bienvenue chez vous " + faceName + "o , brave " + self.convertGender('travailleur')

        # Still late but hey, it's week-end!
        elif dayOfWeek > 3:
            text = "Bonsoir " + faceName + " et bonne nuit, je suppose"

        # Back to basics
        else:
            text = "Bonjour " + faceName

        textToSpeechGoogle(text)


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

            for i,r in enumerate(self.detector.detect(img)):

                # Get the face from image
                face = self.getFace(img, r)

                # Get a model prediction from the face
                predict = self.model.predict(face)

                # Let's recognize the face
                [faceName, curTime, delay] = self.recognize(predict)

                # Perform action if face is recognized
                if faceName:
                    self.actionAfterRecognition(faceName, curTime, delay)


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

    # Start the Application based on the given model
    print "Starting application..."
    App(model=model,
        camera_id=options.camera_id,
        cascade_filename=options.cascade_filename).run()
