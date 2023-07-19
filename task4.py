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

#Function to create a dataset from the ontonotes file
def create_dataset(max_files=None, ontonotes_file=None, file_chapter=None):
	# load parsed ontonotes dataset
	readHandle = codecs.open(ontonotes_file, 'r', 'utf-8', errors='replace')
	str_json = readHandle.read()
	readHandle.close()
	dict_ontonotes = json.loads(str_json)

	list_train_files = list(dict_ontonotes.keys())
	if max_files != None:
		if len(list_train_files) > max_files:
			list_train_files = list_train_files[:max_files]

	# making a test set from the testing chapter
	readHandle2 = codecs.open(file_chapter, 'r', 'utf8', errors='replace')
	listLines = readHandle2.readlines()
	readHandle2.close()

	list_test_files = listLines

	# sent = (tokens, pos, IOB_label)
	list_train = []
	#Creating training set from the parsed set
	for str_file in list_train_files:
		for str_sent_index in dict_ontonotes[str_file]:
			# ignore sents with non-PENN POS tags
			if 'XX' in dict_ontonotes[str_file][str_sent_index]['pos']:
				continue
			if 'VERB' in dict_ontonotes[str_file][str_sent_index]['pos']:
				continue

			list_entry = []

			# compute IOB tags for named entities (if any)
			ne_type_last = None
			for nTokenIndex in range(len(dict_ontonotes[str_file][str_sent_index]['tokens'])):
				strToken = dict_ontonotes[str_file][str_sent_index]['tokens'][nTokenIndex]
				strPOS = dict_ontonotes[str_file][str_sent_index]['pos'][nTokenIndex]
				ne_type = None
				if 'ne' in dict_ontonotes[str_file][str_sent_index]:
					dict_ne = dict_ontonotes[str_file][str_sent_index]['ne']
					if not 'parse_error' in dict_ne:
						for str_NEIndex in dict_ne:
							if nTokenIndex in dict_ne[str_NEIndex]['tokens']:
								ne_type = dict_ne[str_NEIndex]['type']
								break
				if ne_type != None:
					if ne_type == ne_type_last:
						strIOB = 'I-' + ne_type
					else:
						strIOB = 'B-' + ne_type
				else:
					strIOB = 'O'
				ne_type_last = ne_type

				list_entry.append((strToken, strPOS, strIOB))    #Appending (Word, POS tag, BIO NER tag)

			list_train.append(list_entry)

    #Creating test set
	list_test = []      #List to store the test data
	for str_file in list_test_files:
		py_token = nltk.tokenize.sent_tokenize(str_file)
		for elements in py_token:
			list_entry = []
			py_lword = nltk.word_tokenize(elements)
			py_tag = nltk.pos_tag(py_lword)
			for (word, tag) in py_tag:
				# ignore sents with non-PENN POS tags
				if 'XX' == tag:
					continue
				if 'VBZ' == tag:
					continue

				list_entry.append((word, tag, ''))  #Appending (Word, POS tag, " ")

			list_test.append(list_entry)

	return list_train, list_test

#we define some helper functions to generate feature sets for each sentence, which the CRF model will use to train with.
# Function to extract features
def sent2features(sent, word2features_func = None):
	return [word2features_func(sent, i) for i in range(len(sent))]

# Function to extract labels
def sent2labels(sent):
	return [label for token, postag, label in sent]

# Function to extract tokens
def sent2tokens(sent):
	return [token for token, postag, label in sent]

#Creating feature set-dictionary for words
def task1_word2features(sent, i):

    word = sent[i][0]
    postag = sent[i][1]
    suffix=('st','rd','nd','th')

    features = {
        'word' : word,
        'postag': postag,

        # token shape
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.length()' : len(word),
        # token suffix
        'word.suffix_check': word.endswith(suffix),
        'word.suffix': word.lower()[-3:],

        # POS prefix
        'postag[:2]': postag[:2],
    }
    if i ==1:
        word_prev = sent[i-1][0]
        postag_prev = sent[i-1][1]
        features.update({
            '-1:word.lower()': word_prev.lower(),
            '-1:postag': postag_prev,
            '-1:word.lower()': word_prev.lower(),
            '-1:word.isupper()': word_prev.isupper(),
            '-1:word.istitle()': word_prev.istitle(),
            '-1:word.isdigit()': word_prev.isdigit(),
            '-1:word.suffix_check': word_prev.endswith(suffix),
            '-1:word.suffix': word_prev.lower()[-3:],
            '-1:word.length()' : len(word_prev),
            '-1:postag[:2]': postag_prev[:2],
        })
    elif i==2:
        word_prev = sent[i-1][0]
        word_prev2 = sent[i-2][0]
        postag_prev = sent[i-1][1]
        postag_prev2 = sent[i-2][1]
        features.update({
            '-1:word.lower()': word_prev.lower(),
            '-1:postag': postag_prev,
            '-1:word.lower()': word_prev.lower(),
            '-1:word.isupper()': word_prev.isupper(),
            '-1:word.istitle()': word_prev.istitle(),
            '-1:word.isdigit()': word_prev.isdigit(),
            '-1:word.suffix': word_prev.lower()[-3:],
            '-1:word.suffix_check': word_prev.endswith(suffix),
            '-1:word.length()' : len(word_prev),
            '-1:postag[:2]': postag_prev[:2],
            '-2:word.lower()': word_prev2.lower(),
            '-2:postag': postag_prev2,
            '-2:word.lower()': word_prev2.lower(),
            '-2:word.isupper()': word_prev2.isupper(),
            '-2:word.istitle()': word_prev2.istitle(),
            '-2:word.isdigit()': word_prev2.isdigit(),
            '-2:word.suffix': word_prev2.lower()[-3:],
            '-2:word.suffix_check': word_prev2.endswith(suffix),
            '-2:word.length()' : len(word_prev2),
            '-2:postag[:2]': postag_prev2[:2],
        })
    elif i>2:
        word_prev = sent[i-1][0]
        word_prev2 = sent[i-2][0]
        word_prev3 = sent[i-3][0]
        postag_prev = sent[i-1][1]
        postag_prev2 = sent[i-2][1]
        postag_prev3 = sent[i-3][1]
        features.update({
            '-1:word.lower()': word_prev.lower(),
            '-1:postag': postag_prev,
            '-1:word.lower()': word_prev.lower(),
            '-1:word.isupper()': word_prev.isupper(),
            '-1:word.istitle()': word_prev.istitle(),
            '-1:word.isdigit()': word_prev.isdigit(),
            '-1:word.suffix': word_prev.lower()[-3:],
            '-1:word.suffix_check': word_prev.endswith(suffix),
            '-1:word.length()' : len(word_prev),
            '-1:postag[:2]': postag_prev[:2],
            '-2:word.lower()': word_prev2.lower(),
            '-2:postag': postag_prev2,
            '-2:word.lower()': word_prev2.lower(),
            '-2:word.isupper()': word_prev2.isupper(),
            '-2:word.istitle()': word_prev2.istitle(),
            '-2:word.isdigit()': word_prev2.isdigit(),
            '-2:word.suffix': word_prev2.lower()[-3:],
            '-2:word.suffix_check': word_prev2.endswith(suffix),
            '-2:word.length()' : len(word_prev2),
            '-2:postag[:2]': postag_prev2[:2],
            '-3:word.lower()': word_prev3.lower(),
            '-3:postag': postag_prev3,
            '-3:word.lower()': word_prev3.lower(),
            '-3:word.isupper()': word_prev3.isupper(),
            '-3:word.istitle()': word_prev3.istitle(),
            '-3:word.isdigit()': word_prev3.isdigit(),
            '-3:word.suffix': word_prev3.lower()[-3:],
            '-3:word.suffix_check': word_prev3.endswith(suffix),
            '-3:word.length()' : len(word_prev3),
            '-3:postag[:2]': postag_prev3[:2],
        })
    else:
        features['BOS'] = True

    if i == len(sent)-2:
        word_next = sent[i+1][0]
        postag_next = sent[i+1][1]
        features.update({
            '+1:word.lower()': word_next.lower(),
            '+1:postag': postag_next,
            '+1:word.lower()': word_next.lower(),
            '+1:word.isupper()': word_next.isupper(),
            '+1:word.istitle()': word_next.istitle(),
            '+1:word.isdigit()': word_next.isdigit(),
            '+1:word.suffix': word_next.lower()[-3:],
            '+1:word.suffix_check': word_next.endswith(suffix),
            '+1:word.length()' : len(word_next),
            '+1:postag[:2]': postag_next[:2],
        })
    elif i == len(sent)-3:
        word_next = sent[i+1][0]
        postag_next = sent[i+1][1]
        word_next2 = sent[i+2][0]
        postag_next2 = sent[i+2][1]
        features.update({
            '+1:word.lower()': word_next.lower(),
            '+1:postag': postag_next,
            '+1:word.lower()': word_next.lower(),
            '+1:word.isupper()': word_next.isupper(),
            '+1:word.istitle()': word_next.istitle(),
            '+1:word.isdigit()': word_next.isdigit(),
            '+1:word.suffix': word_next.lower()[-3:],
            '+1:word.suffix_check': word_next.endswith(suffix),
            '+1:word.length()' : len(word_next),
            '+1:postag[:2]': postag_next[:2],
            '+2:word.lower()': word_next2.lower(),
            '+2:postag': postag_next2,
            '+2:word.lower()': word_next2.lower(),
            '+2:word.isupper()': word_next2.isupper(),
            '+2:word.istitle()': word_next2.istitle(),
            '+2:word.isdigit()': word_next2.isdigit(),
            '+2:word.suffix': word_next2.lower()[-3:],
            '+2:word.suffix_check': word_next2.endswith(suffix),
            '+2:word.length()' : len(word_next2),
            '+2:postag[:2]': postag_next2[:2],
        })
    elif i < len(sent)-3:
        word_next = sent[i+1][0]
        word_next2 = sent[i+2][0]
        word_next3 = sent[i+3][0]
        postag_next = sent[i+1][1]
        postag_next2 = sent[i+2][1]
        postag_next3 = sent[i+3][1]
        features.update({
            '+1:word.lower()': word_next.lower(),
            '+1:postag': postag_next,
            '+1:word.lower()': word_next.lower(),
            '+1:word.isupper()': word_next.isupper(),
            '+1:word.istitle()': word_next.istitle(),
            '+1:word.isdigit()': word_next.isdigit(),
            '+1:word.suffix': word_next.lower()[-3:],
            '+1:word.suffix_check': word_next.endswith(suffix),
            '+1:word.length()' : len(word_next),
            '+1:postag[:2]': postag_next[:2],
            '+2:word.lower()': word_next2.lower(),
            '+2:postag': postag_next2,
            '+2:word.lower()': word_next2.lower(),
            '+2:word.isupper()': word_next2.isupper(),
            '+2:word.istitle()': word_next2.istitle(),
            '+2:word.isdigit()': word_next2.isdigit(),
            '+2:word.suffix': word_next2.lower()[-3:],
            '+2:word.suffix_check': word_next2.endswith(suffix),
            '+2:word.length()' : len(word_next2),
            '+2:postag[:2]': postag_next2[:2],
            '+3:word.lower()': word_next3.lower(),
            '+3:postag': postag_next3,
            '+3:word.lower()': word_next3.lower(),
            '+3:word.isupper()': word_next3.isupper(),
            '+3:word.istitle()': word_next3.istitle(),
            '+3:word.isdigit()': word_next3.isdigit(),
            '+3:word.suffix': word_next3.lower()[-3:],
            '+3:word.suffix_check': word_next3.endswith(suffix),
            '+3:word.length()' : len(word_next3),
            '+3:postag[:2]': postag_next3[:2],
        })
    else:
        features['EOS'] = True

    return features

#Function to return a trained CRF model
def task1_train_crf_model( X_train, Y_train, max_iter, labels ) :
    # train the basic CRF model
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs',c1=40,c2=0.1,max_iterations=max_iter,all_possible_transitions=False)
    crf.fit(X_train, Y_train)
    return crf


#Function to perfrom named entity ecognition
def exec_ner( file_chapter = None, ontonotes_file = None ) :

	#Creating datasets
	train_sents, test_sents = create_dataset(max_files=3000, ontonotes_file=ontonotes_file, file_chapter=file_chapter)

	# create feature vectors for every sent
	X_train = [sent2features(s, word2features_func=task1_word2features) for s in train_sents]
	Y_train = [sent2labels(s) for s in train_sents]

	X_test = [sent2features(s, word2features_func=task1_word2features) for s in test_sents]
	Y_test = [sent2labels(s) for s in test_sents]

	# get the label set
	set_labels = set([])
	for data in [Y_train]:
		for n_sent in range(len(data)):
			for str_label in data[n_sent]:
				set_labels.add(str_label)
	labels = list(set_labels)

	# remove 'O' label as we are not usually interested in how well 'O' is predicted
	labels.remove('O')

	# Train CRF model
	crf = task1_train_crf_model(X_train, Y_train, max_iter=20, labels=labels)
	Y_pred = crf.predict(X_test)

	sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))

	word_list = []  #List for storing the word
	ne_list = []    #List for storing identified named entity

    #Iterating over predicitons of each sentences and collecting word and NE
	for sentence_index in range(0, len(X_test)):
		word_index = 0
		
        #Loop to iterate over all the words in a sentence
		while word_index < len(X_test[sentence_index]):
			word = ''
			ne = ''
			count = 0
			#Iterating over words again to keep track for continous words
			while word_index < len(X_test[sentence_index]) and Y_pred[sentence_index][word_index] != 'O':
				word = word + ' ' + X_test[sentence_index][word_index]['word']
				ne = Y_pred[sentence_index][word_index].split('-')[1]
				count = count + 1
				word_index = word_index + 1

			word_index = word_index + 1
			word_list.append(word)
			ne_list.append(ne)

	output = []
	for i in range(len(word_list)):
		output.append((ne_list[i], word_list[i]))

	dictNE = {}     #Dictionary to store named entity recognition
	for i in output:
		dictNE.setdefault(i[0], []).append(i[1].strip())

	for k, v in dictNE.items():
		dictNE[k] = list(set(v))
	
    # FILTER NE dict by types required for task 3
	listAllowedTypes = [ 'DATE', 'CARDINAL', 'ORDINAL', 'NORP' ,'PERSON']
	listKeys = list( dictNE.keys() )
	for strKey in listKeys :
		for nIndex in range(len(dictNE[strKey])) :
			dictNE[strKey][nIndex] = dictNE[strKey][nIndex].strip().lower()
		if not strKey in listAllowedTypes :
			del dictNE[strKey]

    #Reading dataset to perform Regex to match Titles in the word like Dr, Professor, Mr, Mrs, Miss etc
	readHandle2 = codecs.open(file_chapter, 'r', 'utf8', errors='replace')
	data = readHandle2.read()

	people_regex = re.compile("\s(Mr|Mrs|Miss|Dr|Professor)(\.|\s)([A-Z][a-z]+)")   #Regex to match
	a = re.findall(people_regex, data)  #Finding matches
	characters = list(set(["".join(x).lower() for x in a]))
	dictNE['PERSON'] = dictNE['PERSON'] + characters    #Appednding to the dictionary

	# write out all PERSON entries for character list for subtask 4 . THE BELOW CODE WRITES THE DICTIONARY IN JSON
	writeHandle = codecs.open( 'characters.txt', 'w', 'utf-8', errors = 'replace' )
	if 'PERSON' in dictNE :
		for strNE in dictNE['PERSON'] :
			writeHandle.write( strNE.strip().lower()+ '\n' )
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


	exec_ner( chapter_file, ontonotes_file )     #Calling the execution function

