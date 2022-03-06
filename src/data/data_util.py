import pandas as pd
import textstat, string
import numpy as np
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
import torch
from sklearn import preprocessing
from wordfreq import word_frequency
import pickle
import copy
import logging
import sys
from indicnlp.syllable import  syllabifier
from RU.ru_transformers.yt_encoder import YTEncoder
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModel, AutoTokenizer, BertTokenizer, AutoModelWithLMHead
import math


modelEN = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizerEN = GPT2Tokenizer.from_pretrained("gpt2")

modelRU = GPT2LMHeadModel.from_pretrained("../RU/gpt2/m_checkpoint-3364613")
tokenizerRU = YTEncoder.from_pretrained("../RU/gpt2/m_checkpoint-3364613")

modelHI = GPT2LMHeadModel.from_pretrained("surajp/gpt2-hindi")
tokenizerHI = AutoTokenizer.from_pretrained("surajp/gpt2-hindi")

modelZH = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
tokenizerZH = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

modelNL = GPT2LMHeadModel.from_pretrained("GroNLP/gpt2-small-dutch")
tokenizerNL = AutoTokenizer.from_pretrained("GroNLP/gpt2-small-dutch")

modelDE = AutoModelWithLMHead.from_pretrained("dbmdz/german-gpt2")
tokenizerDE = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")



class DataLoader():

	@staticmethod
	def load_dataset(dsname, withhead=True, istest=False):
		if withhead:
			names = ['language',	'sentence_id', 'word_id',	'word',	'FFDAvg','FFDStd',	'TRTAvg',	'TRTStd']
			dataset = pd.read_csv(dsname)
		else:
			dataset = pd.read_csv(dsname, header=None)

		df = dataset
		df = df.astype({'word_id': int})
		sent_n = 0
		sent_arr = []

		word_n = 0
		word_arr = []
		for i, rows in df.iterrows():
			
			if i == 0:
				sent_id = df['sentence_id'][i]
				word_n = -1

			if df['sentence_id'][i] != sent_id:
				sent_n +=1
				sent_id = df['sentence_id'][i]
				word_n = 0
			else:
				word_n += 1
			
			df.loc[i, ['word']] = df.loc[i, ['word']].replace(" ","")

			sent_arr.append(sent_n)
			word_arr.append(word_n)

		
		df['sent_n'] = sent_arr
		df['word_n'] = word_arr
			
		if istest:
			rawtext = df.iloc[:,[8,9,3,0]].to_numpy()
			labels = None
		else:
			rawtext = df.iloc[:, [8,9,3,0]].to_numpy()
			labels = df.iloc[:, [4,5,6,7,8,0]].to_numpy()
		
		lang_lab = {}
		for l in labels:
			if l[5] not in lang_lab:
				lab = []
			lab.append(l[0:5])
			lang_lab[l[5]] = np.array(lab, dtype='float64')
		
		
		
		#return rawtext, labels
		return rawtext, lang_lab

	@staticmethod
	def merge_sent(rawtext):
		cur_sent_len = 0
		rawtext = np.insert(rawtext, 0, values=0, axis = 1)
		

		sents = []
		sent = ""

		for i in range(len(rawtext) - 1, -1, -1):
			item = rawtext[i]
			sent_len = item[0]
			sent_id = item[1]
			wordseq_id = item[2]
			wordtext = item[3]
			
			if cur_sent_len < wordseq_id:
				cur_sent_len = wordseq_id
				rawtext[i, 0] = cur_sent_len

			if wordseq_id == 0:
				sent = wordtext + ' ' + sent
				#sents.insert(0, sent[:-6])
				sents.insert(0, sent)
				rawtext[i, 0] = cur_sent_len
				cur_sent_len = 0
				sent = ""
			else:
				sent = wordtext + ' ' + sent
				rawtext[i, 0] = cur_sent_len
		

		# mergedtext = copy.deepcopy(rawtext)
		
		return rawtext, sents

class FeatureExtraction():

	@staticmethod
	def extract_vocabulary(list_files):

		# returns vocabulary of the train and of the test file

		vocab = {}
		stop_words = stopwords.words('english')

		for f in list_files:

			with open(f, "r", encoding='utf-8') as fl:
				for line in fl:

					# if line.startswith("id") or len(line.split("\t")) != 5: continue

					line = line.strip()
					try:
						_, _, sent, tok, _ = line.split("\t")
					except:
						_, _, sent, tok = line.split("\t")
					tok = tok.lower()
					sent = sent.translate(str.maketrans('', '', string.punctuation))
					sent = sent.lower()

					for word in sent.split(" "):
						if word in stop_words: continue
						vocab[word] = 0

					vocab[tok] = 0

		return vocab

	@staticmethod
	def get_vec(word, embs, PRINT_WORD=False):

		# Returns a vector for word from vector matrix embs
		print("IN GET_VEC")

		if word in embs.keys():
			try:
				return embs['_matrix_'][embs[word], :]
			except:
				print('{} should have been there but something went wrong when loading it!'.format(word))
				return []
		else:
			if PRINT_WORD:
				print('{} not in the dataset.'.format(word))

			return []

	@staticmethod
	def load_embeddings(fname, ds_words):
		# Load the embeddings from fname file, only for the words in the variable ds_words

		emb = {}
		matrix = []
		dims = 0
		with open(fname, 'r', encoding='utf-8', errors="ignore") as f:
			for line in f:
				line = line.strip().split()
				if dims == 0:
					if len(line) == 2:
						continue
					else:
						dims = len(line) - 1

				word = line[0]

				if word not in ds_words: continue

				if word in ['', ' ', '\t', '\n']:
					print('Word {} has no value.'.format(word))
					continue
				try:
					vec = [float(x) for x in line[1:]]
					if len(vec) == dims:
						array = np.array(vec)
					else:
						continue
				except:
					continue
				emb[word] = len(emb)
				matrix.append(array)

		emb['_matrix_'] = np.array(matrix)  # normalize()

		return emb
	

	@staticmethod
	def Dict_feature_extract(mergedtext, sents):
		features = []

		logging.info('USING FEATURES: ' + \
					 'CUR_WORD_POS, ' + \
					 'CUR_WORD_LEN, ' + \
					 'PREV_WORD_LEN, ' + \
					 'CUR_WORD_LOGFREQ, ' + \
					 'PREV_WORD_LOGFREQ, ' + \
					 'IS_UPPER, ' + \
					 'IS_CAPITAL, ' + \
					 'SYLLABLE_COUNT, ' + \
					 'SURPRISAL SCORE, '
					 )

		feat_dict = {}

		for i in range(0, len(mergedtext)):

		
			item = mergedtext[i]


			feat = []

			sentlen = item[0]
			sent_id = item[1]
			wordseq_id = item[2]
			wordtext = item[3]
			wordtext = wordtext.replace('<EOS>','')
			wordlanguage = item[4]
			
			if wordlanguage not in feat_dict:
				features = []


			wordpos = wordseq_id
			#CUR_WORD_POS	
			if sentlen != 0:		
				feat.append(wordpos / sentlen)
			else:
				feat.append(0.0)

			wordlen = len(wordtext)
			#print(wordtext, wordlen)
			#CUR_WORD_LEN
			feat.append(wordlen)
			#PREV_WORD_LEN
			if wordseq_id != 0:
				feat.append(len(mergedtext[i-1][3]))
			else:
				feat.append(0.0)

			# append with wordfreq lib
			#CUR_WORD_LOGFREQ
			word_freq_lib = word_frequency(wordtext.lower(), wordlanguage)

			if word_freq_lib == 0.0:
				feat.append(0.0)
			else:
				feat.append(-np.log(word_freq_lib))
			
			#PREV_WORD_LOGFREQ
			if wordseq_id != 0:
				word_freq_lib = word_frequency(mergedtext[i-1][3].lower(), wordlanguage)
				if word_freq_lib == 0.0:
					feat.append(0.0)
				else:
					feat.append(-np.log(word_freq_lib))
			else:
				feat.append(0.0)

			
			# IS_UPPER
			if wordtext == wordtext.upper():
				feat.append(1.0)
			else:
				feat.append(0.0)

			# one if the initial is upper case
			# IS_CAPITAL
			if wordtext[0] == wordtext[0].upper():
				feat.append(1.0)
			else:
				feat.append(0.0)

			# append the first features: the syllable count and the word length of the target
			# SYLLABLE_COUNT
			if wordlanguage not in ["hi", "zh"]:
				textstat.set_lang(wordlanguage)
				syllab = textstat.syllable_count(wordtext)
			if wordlanguage == "zh":
				syllab = wordlen
			if wordlanguage == "hi":
				syllab=len(syllabifier.orthographic_syllabify(wordtext,'hi'))
			feat.append(syllab)		
			#print(wordtext, textstat.syllable_count(wordtext))

			splittedSent = sents[sent_id].split(' ')
			index_token = splittedSent.index(wordtext)


			if wordlanguage == "en":
				tokenizer = tokenizerEN
				model = modelEN
			if wordlanguage == "zh":
				tokenizer = tokenizerZH
				model = modelZH
			if wordlanguage == "hi":
				tokenizer = tokenizerHI
				model = modelHI
			if wordlanguage == "ru":
				tokenizer = tokenizerRU
				model = modelRU
			if wordlanguage == "nl":
				tokenizer = tokenizerNL
				model = modelNL
			if wordlanguage == "de":
				tokenizer = tokenizerDE
				model = modelDE

			tok_tens = torch.tensor(tokenizer.encode(sents[sent_id]))
			loss = model(tok_tens, labels=tok_tens)
			surprisal = -1 * math.log(-1 * loss[0].item())
			feat.append(surprisal)


			features.append(feat)
			feat_dict[wordlanguage] = np.array(features, dtype='float64')

		#features_array = np.array(features, dtype='float64')

		# fpout = open('../output/features_array.txt', 'w', encoding='utf-8')
		# for item in features_array:
		#	 fpout.writelines(str(item) + '\n')
		# fpout.close()

		return feat_dict
