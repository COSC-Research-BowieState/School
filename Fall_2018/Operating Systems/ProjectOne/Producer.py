#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 10:46:42 2018

@author: hkyeremateng-boateng
"""

#import MessageQueue as messageQueue
from threading import Thread, Lock, Condition
condition = Condition()

class Producer(Thread):
    
    def run(self):
        condition.acquire()
        pass
    
    
    @classmethod
    def generateMessage(self,maxNumber):
        if maxNumber is None:
            maxNumber = 10
        print("The Producer will be generate ",maxNumber," number of message",threading.current_thread().name)
        for i in range(maxNumber):
            print("Message ",i,threading.current_thread().name)