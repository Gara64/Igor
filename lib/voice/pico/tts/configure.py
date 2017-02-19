#!/usr/bin/python
# -*- coding: utf-8 -*-
##############################################################################################
import os
import time
import sys
import subprocess

##############################################################################################
class ConfirugePicoPi(object):
    
    # ===========================================================================
    def __init__(self, argv):
        
        # --------------------------------------------------------------
        self._VERSION                       = "v0.0.1"
        self._DEV_PAGE                      = "https://github.com/ch3ll0v3k/picoPi2"
        # --------------------------------------------------------------
        self._argv                          = " ".join(argv);
        self._argc                          = len(argv);
        self._app_w                         = 80;
        self._cmd_w                         = 24;

        # --------------------------------------------------------------
        self.MAKE                           = False;
        self.LANG_TTS                       = "USA";
        self.LANG_BIN_PATH                  = "../lang/";
        self.LANGS                          = ["GBR", "USA", "FRA", "DEU", "SPA", "ITA"];

        self.TO_STD                         = True; # True -> STD. False -> .wav

        self._SYS_MSG                       = "    Welcome to picoPi2";

        # --------------------------------------------------------------
        self.ParseArgs(self._argv);
        # --------------------------------------------------------------

    # ===========================================================================
    def Confiruge(self):
        
        # --------------------------------------------------------------
        os.system("clear");
        

        # --------------------------------------------------------------

    # ===========================================================================
    def ParseArgs(self, _argv):
        
        # --------------------------------------------------------------
        if "--help" in self._argv: 
            self.PrintUsage();
        elif "--version" in self._argv: 
            print(" Version: "+self._VERSION);
            exit();
        elif self._argc < 2:
            self.PrintUsage();

        # --------------------------------------------------------------
        if "--make" in self._argv:
            self.MAKE = True;
        else:
            self._SYS_MSG = " ERROR: (--make) must be presented!";
            self.PrintUsage();
            
        # --------------------------------------------------------------
        if "--lang-tts" in self._argv:

            lang_tts_arg = self.GetCmdArg("--lang-tts");
            if lang_tts_arg[1].strip() in self.LANGS:

                self.LANG_TTS = lang_tts_arg[1].strip();
            else:
                self._SYS_MSG = " ERROR: Unknown TTS lang";
                self.PrintUsage();
        # --------------------------------------------------------------
        if "--lang-bin" in self._argv:

            path = os.path.abspath(self.GetCmdArg("--lang-bin-dir")[1].strip());

            if os.path.isdir(path):
                self.LANG_BIN_PATH = path;
            else:
                self._SYS_MSG = " ERROR: Directory NOT-EXITST ->\n\t"+path;
                self.PrintUsage();

        # --------------------------------------------------------------
        if "--2wav" in self._argv:

            self.TO_STD = False;
        
        # --------------------------------------------------------------
        self.DisplayConfSetting();
        # --------------------------------------------------------------

    # ===========================================================================
    def PrintUsage(self):

        # --------------------------------------------------------------
        tb = "    ";
        # --------------------------------------------------------------
        os.system("clear");
        self._line();
        # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # Usage menu body!
        print("   "+"\033[01;31m"+self._SYS_MSG+"\033[0m");
        self._line();
        self._print(" Usage: ./configure.py [ --make | --lang-tts | lang-bin ]");
        self._print("");
        self._print(tb+"--help:",           "Display this help");
        self._print(tb+"--make:",           "Compile");
        self._print(tb+"--lang-tts:",       "Select TTS language "+tb+"(Default USA)");
        self._print(tb+" ",                 tb+"[GBR, USA, FRA, DEU, SPA, ITA]");
        self._print(tb+"--lang-bin-dir:",   "Binary languages directory "+tb+'(Default "../lang/")');
        self._print(tb+"--2wav:",           "Binary wil output sound to '.wav' file. (Default) ");
        self._print(tb+"--2std:",           "Binary wil output sound to std.");
        self._print(tb+"--version:",        "Print current Version");


        self._line();
        self._print(" Sample:");
        self._print("");
        self._print(tb+"[./configure.py --make ] -> Compile (default setting) ");

        self._print(tb+"[./configure.py --2wav ] picoPi2Wav -w file.wav 'this' && aplay file.wav ");
        self._print(tb+"[./configure.py ] picoPi2Stdout | aplay  ");
        self._print(tb+"[./configure.py --make lang-tts SPA] -> Compile Spanish TTS ");
        self._print(tb+"[./configure.py --make lang-bin \"/path/to/\"] -> Force to use this dir ");
        # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        self._line();
        self._print("");
        self._print(" Version: "+self._VERSION);
        self._print(" Dev-Page: "+self._DEV_PAGE);
        self._print("");
        self._line();
        # --------------------------------------------------------------
        print("");
        self._SYS_MSG = "";
        exit(0);
        # --------------------------------------------------------------

    # ===========================================================================
    def DisplayConfSetting(self):

        # --------------------------------------------------------------
        tb = "    ";
        os.system("clear");
        self._line();
        # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # Usage menu body!
        self._print(" This SETTING wil be used:");
        self._print("");
        self._print(tb+"--make:",           '"'+str(self.MAKE)+'"');
        self._print(tb+"--lang-tts:",       '"'+self.LANG_TTS+'"');
        self._print(tb+"--lang-bin-dir:",   '"'+self.LANG_BIN_PATH+'"');
            
        if self.TO_STD:
            self._print(tb+"--2std:",       '"True"');
            self._print(tb+" Binary name:", '"./picoPi2Stdout"');
        else:
            self._print(tb+"--2wav:",       '"True"');
            self._print(tb+" Binary name:", '"./picoPi2Wav"');


        self._print("");
        self._line();
        self._print("");
        
        ANSW = str(raw_input(" Correct SETTINGS (Y/N) $> "))

        while ANSW != "Y" and ANSW != "N":

            ANSW = raw_input("   "+"\033[01;31m Invalid option select (Y/N)\033[0m $> ")
            if ANSW == "N":
                exit(0);
            elif ANSW == "Y":
                break;

        # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        self._print("");
        self._line();
        # --------------------------------------------------------------
        print("");
        # --------------------------------------------------------------
        print(" Compiling ... \n");

        if self.TO_STD:

            subprocess.call("make LANGTTS=" +self.LANG_TTS + " -f ./Makefile.picoPi2Std.mkf", shell=True);
        else:

            subprocess.call("make -f ./Makefile.picoPi2Wav.mkf", shell=True);

        # --------------------------------------------------------------

    # ===========================================================================
    def _print(self, _str_A="", _str_B=""): 

        # --------------------------------------------------------------
        if _str_B == "":

            _STD = "|{0:"+str(self._app_w)+"}|";
            print(str("|{0:"+str(self._app_w)+"}|").format(_str_A));

        elif _str_A != "" and _str_B != "":
            
            _STD = "|{0:"+str(self._cmd_w)+"}{1:"+str(self._app_w-self._cmd_w)+"}|";
            print(_STD.format(_str_A, _str_B));

        else:

            print("|{0:22}{1:22}| ".format(_str_A, _str_B));
        # --------------------------------------------------------------

    # ===========================================================================
    def GetCmdArg(self, _cmd):

        # --------------------------------------------------------------
        return (_cmd, self._argv.strip().split(_cmd)[1].strip().split(" ")[0].strip());
        # --------------------------------------------------------------

    # ===========================================================================
    def _line(self): print("|"+("-"*self._app_w)+"|");

    # ===========================================================================


##############################################################################################
if __name__ == "__main__":

    _ConfirugePicoPi = ConfirugePicoPi(sys.argv);
