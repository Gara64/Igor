#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import sys
import random
sys.path.append("lib")

from voice.text2speech import textToSpeechPico

# Dict to associate action detection type with speeches info
speechesAfterDetect = {}
# Dict to associate the models names with people info (e.g. nicknames, gender)
peopleInfo = {}

# Build the sentences structures
def buildSentences(sentences):
    for sentence in sentences:
        # Action after detection speeches
        if "after" in sentence["when"]:
            # An "after type" can be self, other, or any
            afterType = sentence["when"]["after"]

            # The any type just indicates the default speeches
            if afterType == "any":
                speechesAfterDetect[afterType] = {}
                speechesAfterDetect[afterType]["speeches"] = sentence["speeches"]

            # All the others type contains info stored for detection
            else:
                if afterType not in speechesAfterDetect:
                    speechesAfterDetect[afterType] = []

                afterInfo = {}
                # The actions can be triggered with a min or max delay
                if "min_delay" in sentence["when"]:
                    minDelay = sentence["when"]["min_delay"]
                    afterInfo["min_delay"] = minDelay
                if "max_delay" in sentence["when"]:
                    maxDelay = sentence["when"]["max_delay"]
                    afterInfo["max_delay"] = maxDelay

                afterInfo["speeches"] = sentence["speeches"]
                speechesAfterDetect[afterType].append(afterInfo)


# Build the people names structure
def buildPeople(people):
    for p in people:
        peopleInfo[p["model_name"]] = (
            p["speech_name"],
            p["nicknames"],
            p["gender"])


# Gender conversion for speech after face recognition
def gendirify(text, faceName):
    if "{" in text and "}" in text:
        text = text.replace("{", "").replace("}","")
        tokens = text.split(",")
        gender = peopleInfo[faceName][2]
        if gender == "female":
            return tokens[1]
        else:
            return tokens[0]
    else:
        return text


# Util method to pick a random text speech in the array
def pickSpeech(speeches, faceName):
    speech = speeches[random.randint(0, len(speeches) - 1)]
    return gendirify(speech, faceName)


# Util method to pick a speech or nickname name based on the model name
def pickName(faceName):
    people = peopleInfo[faceName]
    nickNames = people[1]
    # TODO : test this condition
    if nickNames is not None and len(nickNames) > 0:
        return nickNames[random.randint(0, len(nickNames) - 1)]
    else:
        return people[0]


# Returns speeches that respect the last detect delay
def getSpeeches(lastDetectDelay, speechesInfo):
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
def getAfterDetectionText(faceName, lastFaceDetect, curTime, lastDetectDelay):
    # Iterate over all the previous detection
    for name, delay in lastFaceDetect.iteritems():

        # Check a recent other detection
        if name != faceName:
            lastDetectdelay = curTime - int(delay)
            speechesInfo = speechesAfterDetect["other"]
        # Check a recent self detection
        else:
            speechesInfo = speechesAfterDetect["self"]

        print "last detect : " + str(lastDetectDelay)

        # Get speeches based on the last detect delay
        speeches =  getSpeeches(lastDetectDelay, speechesInfo)
        if speeches is not None:
            return pickSpeech(speeches, faceName) + " " + pickName(faceName)
        else:
            return None


# Get a default text
def getDefaultText(faceName):
    speeches = speechesAfterDetect["any"]["speeches"]
    return pickSpeech(speeches, faceName) + ", " + pickName(faceName)


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
def actionAfterRecognition(faceName, lastFaceDetect, curTime, delay):

    delayHours = delay / 3600
    hourOfDay = time.localtime(curTime).tm_hour
    dayOfWeek =  time.localtime(curTime).tm_wday

    # # It's been a long time since the last detect
    # elif delayHours > 40:
    #     text = "Vous revoila + " + faceName + " ! Vous m'avez manquai"
    #
    # # Why are you still there ?
    # elif delay > 0 and delay < 10:
    #     text = "Pourquoi restez-vous là, " + convertGender(faceName, 'maitre') + " ?"
    #
    # # Man, it's pretty late and we're during the week
    # elif hourOfDay > 21 and delayHours > 10 and dayOfWeek < 4:
    #     text = "Bienvenue chez vous " + faceName + "o , brave " + convertGender(faceName, 'travailleur')
    #
    # # Still late but hey, it's week-end!
    # elif dayOfWeek > 3:
    #     text = "Bonsoir " + faceName + " et bonne nuit, je suppose"
    #
    # # Back to basics
    # else:
    #     text = "Bonjour, " + faceName


    text = getAfterDetectionText(faceName, lastFaceDetect, curTime, delay)
    if text is None:
        text = getDefaultText(faceName)

    print "text to speech : " + text
    textToSpeechPico(text)
