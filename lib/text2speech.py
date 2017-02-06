#!usr/bin/python
# -*- coding: utf8 -*-

import sys
import urllib
import urllib2
from subprocess import call

reload(sys)  
sys.setdefaultencoding('utf8')

# This script uses the Google translate API and mpg321 to turn a text into voice
# Prerequisites : mpg321

def textToSpeechGoogle(text):
    # Make text url compliant
    urlText = urllib2.quote(text.encode('utf8'))

    # Call google API and play the stream
    url = "http://translate.google.com/translate_tts?tl=fr&client=tw-ob&q="+urlText
    call(["mpg321", url])
