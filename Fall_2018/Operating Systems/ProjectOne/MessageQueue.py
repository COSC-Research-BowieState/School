#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 10:23:05 2018

@author: hkyeremateng-boateng
"""
from datetime import datetime

class MessageQueue:
    messageId = 0;
    message = None;
    dateCreated = datetime.now()
    
    def toString():
        return messageId