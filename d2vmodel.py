# coding: utf-8
import sys
# sys.setdefaultencoding() does not exist, here!
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')

from os import listdir
from os.path import isfile, join


docLabels = []
docLabels = [f for f in listdir("/home/deepak/New/docs") if 
	f.endswith('.txt')]
print docLabels


data = []



f = open('/home/deepak/New/docs/'+ docLabels[2]).read()
print f
