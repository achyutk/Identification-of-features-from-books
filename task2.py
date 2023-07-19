######################################################################
# (c) Copyright University of Southampton, 2021
#
# Copyright in this software belongs to University of Southampton,
# Highfield, University Road, Southampton SO17 1BJ
#
# Created By : Stuart E. Middleton
# Imporoved and worked by: Achyut Karnani
# Project : Teaching
#
######################################################################

# Importing Necessary Packages
from __future__ import absolute_import, division, print_function, unicode_literals
import sys, codecs, json, math, time, warnings, re, logging
warnings.simplefilter( action='ignore', category=FutureWarning )
import nltk, numpy, scipy, sklearn, sklearn_crfsuite, sklearn_crfsuite.metrics

#Creating a logger for evaluation
LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
logger.info('logging started')

# Function to build Regex to list all the questions in a chapter
def exec_regex_questions( file_chapter = None ) :

	#Open Book document which is in txt file
	readHandle2 = codecs.open(file_chapter, 'r', 'utf8', errors='replace')
	listLines = readHandle2.readlines()
	readHandle2.close()

	# Regex to find Questions
	# regex = re.compile('\.(.*)\?')
	regex = re.compile('(?:^|\.|\?|;|!|:|\"|‘|“)([^.?!]*\?)')
	# regex = re.compile('(?:^|\.\s)([\w\s]*?(?:Mr\.|Mrs\.|Ms\.|Dr\.)?\s*[\w\s]*\?)')

	# Code to combine line \n from sentences and combinin the entire pagraphs
	l = []
	for i in listLines:
		if len(l) == 0:
			l.append(i)
		else:
			if i == "\r\n":
				l.append(i)
			else:
				if l[-1] == "\r\n":
					l.append(i.strip())
				else:
					l[-1] = l[-1].rstrip('\r\n') + " " + i.strip()

	ques = [] # list to store questions 


	# # Removing Index from the book to get rid of confusion
	data = "".join(l)
	data = re.sub(u"(\u2018|\u2019)", "'", data)
	index = re.sub('CONTENTS([\s\S]*?)\r\n\r\n\r\n', '', data, flags=re.IGNORECASE)
	listLines = index.split("\r\n")

	#Iterating over lines
	for i in listLines:
		q_match = regex.findall(i)
		if q_match != None:
			for i in q_match:
				ques.append(i.lstrip('. \'\",!?;:-').rstrip('\'\"'))

	#Loop to format text and remove , \' from the question
	for i in range(len(ques)):
		if ques[i].rfind(', \'') != -1:
			ques[i] = ques[i][ques[i].rfind(', \'') + 3:]

	#Loop to format text and remove ; \' from the question
	for i in range(len(ques)):
		if ques[i].rfind('; \'') != -1:
			ques[i] = ques[i][ques[i].rfind('; \'') + 3:]

	#Loop to format text and remove : \' from the question
	for i in range(len(ques)):
		if ques[i].rfind(': \'') != -1:
			ques[i] = ques[i][ques[i].rfind(': \'') + 3:]

	#Loop to format text and remove  “ from the question
	for i in range(len(ques)):
		if ques[i].rfind(' “') != -1:
			ques[i] = ques[i][ques[i].rfind(' “') + 2:]

	#Loop to format text and remove ;  from the question
	for i in range(len(ques)):
		if ques[i].rfind('; ') != -1:
			ques[i] = ques[i][ques[i].rfind('; ') + 2:]

	#Loop to format text and remove :  from the question
	for i in range(len(ques)):
		if ques[i].rfind(': ') != -1:
			ques[i] = ques[i][ques[i].rfind(': ') + 2:]

	#Loop to format text and remove -- from the question
	for i in range(len(ques)):
		if ques[i].rfind('--') != -1:
			ques[i] = ques[i][ques[i].rfind('--') + 2:]

	#Loop to format text and remove -\'  from the question
	for i in range(len(ques)):
		if ques[i].rfind('-\' ') != -1:
			ques[i] = ques[i][ques[i].rfind('-\' ') + 3:]

	#Loop to format text and remove —“  from the question
	for i in range(len(ques)):
		if ques[i].rfind('—“') != -1:
			ques[i] = ques[i][ques[i].rfind('—“') + 2:]

	#Loop to format text and remove ,—  from the question
	for i in range(len(ques)):
		if ques[i].rfind(',—') != -1:
			ques[i] = ques[i][ques[i].rfind(',—') + 2:]

	#Loop to format text and remove ,\' and ,\" from the question
	for i in range(len(ques)):
		if ques[i].rfind(',\' ') != -1:
			ques[i] = ques[i][ques[i].rfind(',\' ') + 3:]

		if ques[i].rfind(',\" ') != -1:
			ques[i] = ques[i][ques[i].rfind(',\' ') + 3:]

	#Loop to format text and remove punctuations from the end and the start from the question	
	for i in range(len(ques)):
		ques[i] = ques[i].lstrip('\'_-—)( “”')
		ques[i] = ques[i].rstrip('\'')

	#Loop to format text and remove "  from the question
	for i in range(len(ques)):
		if ques[i].rfind('"') != -1:
			ques[i] = ques[i][ques[i].rfind('"') + 1:]

	#Remove text incase there are bank spaces 
	for i in ques:
		if len(i) == 0:
			ques.remove(i)

	setQuestions = set(ques)	

	# THE BELOW CODE WRITES THE DICTIONARY IN JSON
	writeHandle = codecs.open( 'questions.txt', 'w', 'utf-8', errors = 'replace' )
	for strQuestion in setQuestions :
		writeHandle.write( strQuestion + '\n' )
	writeHandle.close()

#Code below is for taking arguments
if __name__ == '__main__':
	if len(sys.argv) < 4 :
		raise Exception( 'missing command line args : ' + repr(sys.argv) )
	ontonotes_file = sys.argv[1]
	book_file = sys.argv[2]
	chapter_file = sys.argv[3]

	logger.info( 'ontonotes = ' + repr(ontonotes_file) )
	logger.info( 'book = ' + repr(book_file) )
	logger.info( 'chapter = ' + repr(chapter_file) )

	exec_regex_questions( chapter_file ) #Calling the execution function

