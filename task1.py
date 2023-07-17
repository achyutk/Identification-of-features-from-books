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

# Importing Necessary Files
from __future__ import absolute_import, division, print_function, unicode_literals
import sys, codecs, json, math, time, warnings, re, logging
warnings.simplefilter( action='ignore', category=FutureWarning )
import nltk, numpy, scipy, pandas, sklearn, sklearn_crfsuite, sklearn_crfsuite.metrics

#Creating a loffer for ecaluation
LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
logger.info('logging started')

# Function to build Regex for table of contents
def exec_regex_toc(file_book = None ) :

	readHandle2 = codecs.open(file_book, 'r', 'utf8', errors='replace') 	#Open Book document which is in txt file
	data = readHandle2.read() # Reading book by lines

	book_regex = re.compile("^(BOOK|PART|VOLUME) ([A-Z\s]+|\d+)\.?$")		# Regex to capture title of books. Used in cases where there are multiple books
	chapter_regex = re.compile("^\s?(STAVES?|Staves?|Chapters?|CHAPTERS?).*") # Regex to capture title of chapters in the book re.compile("[a-z]+")
	lowercase_regex = re.compile("[a-z]+") 	#Regex to identify text in lowercase.
	coninue_regex = re.compile("(continued|contd\.)") 	# Regex to capture title of the book where the tite is contiued in cople of pages


	# Removing Index from the book to get rid of confusion
	data = re.sub(u"(\u2018|\u2019)", "'", data)
	index = re.sub('CONTENTS([\s\S]*?)\r\n\r\n\r\n\r\n', '', data, flags=re.IGNORECASE)
	listLines = index.split("\r\n")
	listLines

	# List to store titles and its numbers
	book_list = []	# List to store the names of books
	count_chaps_book = [] # List to store the number of chapters in each book
	title_list = [] # List to store the names of chapters
	number_list = [] # List to store the number of chapters
	count_chap = 0		#Counter to count the number of chapters


	#Iterating over each lines of the book
	for i in range(len(listLines)):
		book_match = book_regex.search(listLines[i])	#Matching regex pattern  for book title
		title_match = chapter_regex.search(listLines[i]) #Matching regex pattern for chapter

		#Condition to check if the regex for book matched for the line
		if book_match != None:
			#Condtion to keep count of chapters in each book. This condition will be true only at the end of a book and start of new book
			if count_chap != 0:
				count_chaps_book.append(count_chap) 	#Appending count of chaps of previous book
				count_chap = 0 	#Setting the counter to 0
			book_list.append(book_match.group())	# Adding the book title to  book list

		#Condition to check if the regex for chapter matched for the line
		if title_match != None:
			count_chap = count_chap + 1
			number_list.append(title_match.group().split()[1])		#Appendning the title number to a list
			title = ""		#String to store the name of chapter
			try:
				title = title + " ".join(title_match.group().split()[2:]) 		#Appedning to String from the match
			except:
				title = ""
			title = title + " " + listLines[i + 1]

			#Looping over multiple lines (used in cases where the title is in multiple lines)
			for j in range(i + 2, i + 6):
				# Generally, if the title is continued on another line, then the following line will either have Ccontd written on it or it will be in lower case. Hence the code belo.
				if lowercase_regex.search(listLines[j]) == None:
					title = title + " " + listLines[j]
				else:
					if coninue_regex.search(listLines[j]) != None and j == i + 2:
						title = title + " " + listLines[j]
			if title.isspace():
				title = title + listLines[i + 1] + listLines[i + 2] + listLines[i + 3]

			title_list.append(title)		#Appending title to the list
	count_chaps_book.append(count_chap)	#Appending count of chapters to the list

	#Creating dictionary for proper output
	dictTOC = {}

	#Code if there are multiple books in the document
	if len(book_list) != 0:
		count = 0
		#Iterating over books 
		for i in range(len(book_list)):
			#Iterating over chapters 
			for j in range(count_chaps_book[i]):
				dictTOC["(" + book_list[i].rstrip("-:. ") + ") " + number_list[count].rstrip("-:. ")] = title_list[
					count].strip()
				count = count + 1
	else:
		for i in range(len(number_list)):
			dictTOC[number_list[i].rstrip("-:. ")] = title_list[i].strip()




	# THE BELOW CODE WRITES THE DICTIONARY IN JSON
	try:
		writeHandle = codecs.open( 'toc.json', 'w', 'utf-8', errors = 'replace' )
	except:
		print("error")
	strJSON = json.dumps( dictTOC, indent=2 )
	writeHandle.write( strJSON + '\n' )
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

	# DO NOT CHANGE THE CODE IN THIS FUNCTION
	print(book_file)
	print("**********")
	exec_regex_toc( book_file )