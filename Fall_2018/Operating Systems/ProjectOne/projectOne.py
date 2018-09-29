#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 10:20:53 2018

@author: hkyeremateng-boateng
"""
#import MessageQueue as messageQueue
#from Producer import Producer as producer
from threading import Thread, Condition
import random
''' Message Queue '''
messageQueue = []
condition = Condition()

class ProjectOne:
    print("===== Welcome Fall 2018 - Operating System Project 1 =====");

class Consumer(Thread):
    def run(self):
        global messageQueue
        while True:
            condition.acquire()
            if not queue:
                print("Waiting for Producer")
                condition.wait()
                print("Producer added message amd Consumer knows")
            worker =    messageQueue.pop(0)
            print("Consumer just consumes this message",worker)
            condition.release()
        
class Producer(Thread):

    def run(self):
        global messageQueue
        workers = range(10)
        while True:
            worker = random.choice(workers);
            condition.acquire()
            
            messageQueue.append(worker)
            print("Message",worker,"as being sent")
            condition.notify()
            condition.release()
p = ProjectOne()

'''
p = ProjectOne()
t = Thread(target = producer.generateMessage, args= (3,))
s = Thread(target = producer.generateMessage, args= (4,))
t.start()
s.start()
t.join()
s.join()

#producer.generateMessage(3).
'''
