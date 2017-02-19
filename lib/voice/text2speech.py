#!usr/bin/python
# -*- coding: utf8 -*-

import sys
import os
import urllib
import urllib2
import subprocess

reload(sys)
sys.setdefaultencoding('utf8')



class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


# Uses the Google translate API and mpg321 to turn a text into voice
# Prerequisites : mpg321
def textToSpeechGoogle(text):
    # Make text url compliant
    urlText = urllib2.quote(text.encode('utf8'))

    # Call google API and play the stream
    url = "http://translate.google.com/translate_tts?tl=fr&client=tw-ob&q="+urlText
    subprocess.call(["mpg321", url])

# Uses the Google Pico TTS. Require to compile the bin and aplay
# See https://github.com/ch3ll0v3k/picoPi2
def textToSpeechPico(text):
    with cd("lib/voice/pico/tts"):
        pico = subprocess.Popen(("./picoPi2Std", text), stdout=subprocess.PIPE)
        subprocess.call(["aplay", "-f", "S16_LE", "-r", "16000"], stdin=pico.stdout)
        pico.wait()
