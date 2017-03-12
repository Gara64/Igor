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
from voice.text2speech import textToSpeechPico


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
    # Dict to associate the models names with people info (e.g. nicknames, gender)
    peopleInfo = {}
    # Dict to associate action detection type with speeches info
    speechesAfterDetect = {}

    photoPath = ""


    def __init__(self, model, camera_id, cascade_filename, people, sentences, photoPath):
        self.model = model
        self.detector = CascadedDetector(cascade_fn=cascade_filename, minNeighbors=5, scaleFactor=1.4)
        self.cam = create_capture(camera_id)

        self.buildSentences(sentences)
        self.buildPeople(people)

        self.photoPath = photoPath
        # TODO : implements photo save at detection


    # Get a face from the image
    def getFace(self, img, r):
        # (1) Get face, (2) Convert to grayscale & (3) resize to image_size:
        x0,y0,x1,y1 = r
        face = img[y0:y1, x0:x1]
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, self.model.image_size, interpolation = cv2.INTER_CUBIC)

        return face


    # Build the sentences structures
    def buildSentences(self, sentences):
        for sentence in sentences:
            # Action after detection speeches
            if "after" in sentence["when"]:
                # An "after type" can be self, other, or any
                afterType = sentence["when"]["after"]

                # The any type just indicates the default speeches
                if afterType == "any":
                    self.speechesAfterDetect[afterType] = {}
                    self.speechesAfterDetect[afterType]["speeches"] = sentence["speeches"]

                # All the others type contains info stored for detection
                else:
                    if afterType not in self.speechesAfterDetect:
                        self.speechesAfterDetect[afterType] = []

                    afterInfo = {}
                    # The actions can be triggered with a min or max delay
                    if "min_delay" in sentence["when"]:
                        minDelay = sentence["when"]["min_delay"]
                        afterInfo["min_delay"] = minDelay
                    if "max_delay" in sentence["when"]:
                        maxDelay = sentence["when"]["max_delay"]
                        afterInfo["max_delay"] = maxDelay

                    afterInfo["speeches"] = sentence["speeches"]
                    self.speechesAfterDetect[afterType].append(afterInfo)


    # Build the people names structure
    def buildPeople(self, people):
        for p in people:
            self.peopleInfo[p["model_name"]] = (
                p["speech_name"],
                p["nicknames"],
                p["gender"])


    # Gender conversion for speech after face recognition
    def gendirify(self, text, faceName):
        if "{" in text and "}" in text:
            text = text.replace("{", "").replace("}","")
            tokens = text.split(",")
            gender = self.peopleInfo[faceName][2]
            if gender == "female":
                return tokens[1]
            else:
                return tokens[0]
        else:
            return text


    # Util method to pick a random text speech in the array
    def pickSpeech(self, speeches, faceName):
        speech = speeches[random.randint(0, len(speeches) - 1)]
        return self.gendirify(speech, faceName)


    # Util method to pick a speech or nickname name based on the model name
    def pickName(self, faceName):
        people = self.peopleInfo[faceName]
        nickNames = people[1]
        # TODO : test this condition
        if nickNames is not None and len(nickNames) > 0:
            return nickNames[random.randint(0, len(nickNames) - 1)]
        else:
            return people[0]


    # Returns speeches that respect the last detect delay
    def getSpeeches(self, lastDetectDelay, speechesInfo):
        hasSpeech = False
        hasMinDelay = False
        hasMaxDelay = False

        # Lookup all the speeches info and check the min/max delay
        for speechInfo in speechesInfo:
            if "min_delay" in speechInfo:
                hasMinDelay = True
                if lastDetectDelay >= speechInfo["min_delay"]:
                    hasSpeech = True
            if "max_delay" in speechInfo:
                hasMaxDelay = True
                if lastDetectDelay <= speechInfo["max_delay"]:
                    # Additional check in case of both min and max delays
                    if hasMinDelay and not hasSpeech:
                        hasSpeech = False
                    else:
                        hasSpeech = True
                else:
                    hasSpeech = False

            # Returns the speeches if all the delays conditions are met
            if hasSpeech:
                return speechInfo["speeches"]
            else:
                return None


    # Return text if someone else was recently detected
    def getAfterDetectionText(self, faceName, curTime, lastDetectDelay):
        # Iterate over all the previous detection
        for name, delay in self.lastFaceDetect.iteritems():

            # Check a recent other detection
            if name != faceName:
                lastDetectdelay = curTime - int(delay)
                speechesInfo = self.speechesAfterDetect["other"]
            # Check a recent self detection
            else:
                speechesInfo = self.speechesAfterDetect["self"]

            print "last detect : " + str(lastDetectDelay)

            # Get speeches based on the last detect delay
            speeches =  self.getSpeeches(lastDetectDelay, speechesInfo)
            if speeches is not None:
                return self.pickSpeech(speeches, faceName) + " " + self.pickName(faceName)
            else:
                return None


    # Get a default text
    def getDefaultText(self, faceName):
        speeches = self.speechesAfterDetect["any"]["speeches"]
        return self.pickSpeech(speeches, faceName) + ", " + self.pickName(faceName)


    # Action after recognition
    # Algo :
    # * Detection d'une autre personne recemment
    # * détection de la meme personne plusieurs fois de suite
    # * detection après un long délai
    # * détection en fct de l'heure / jour
    # * détection par défaut
    # TODO : the basic hello is boring. make an algo to make it smarter
    # eg say hello once in the day
    # say random sentence / surname, depending on the time, weather, ..
    # Idea : voice mail -> drop messages to say when SO is back a home
    def actionAfterRecognition(self, faceName, curTime, delay):

        delayHours = delay / 3600
        hourOfDay = time.localtime(curTime).tm_hour
        dayOfWeek =  time.localtime(curTime).tm_wday

        # # It's been a long time since the last detect
        # elif delayHours > 40:
        #     text = "Vous revoila + " + faceName + " ! Vous m'avez manquai"
        #
        # # Why are you still there ?
        # elif delay > 0 and delay < 10:
        #     text = "Pourquoi restez-vous là, " + self.convertGender(faceName, 'maitre') + " ?"
        #
        # # Man, it's pretty late and we're during the week
        # elif hourOfDay > 21 and delayHours > 10 and dayOfWeek < 4:
        #     text = "Bienvenue chez vous " + faceName + "o , brave " + self.convertGender(faceName, 'travailleur')
        #
        # # Still late but hey, it's week-end!
        # elif dayOfWeek > 3:
        #     text = "Bonsoir " + faceName + " et bonne nuit, je suppose"
        #
        # # Back to basics
        # else:
        #     text = "Bonjour, " + faceName


        text = self.getAfterDetectionText(faceName, curTime, delay)
        if text is None:
            text = self.getDefaultText(faceName)

        print "text to speech : " + text
        textToSpeechPico(text)


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
                if faceName:
                    self.actionAfterRecognition(faceName, curTime, delay)

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
