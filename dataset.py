import os;
import ast;
import math;
import pickle;
import numpy as np;
import collections;
import nltk;
nltk.data.path.append('/home/arun/Softwares/NLTK/');

from corpus.cornell import *;
from corpus.opensubtitles import *;

class Dataset(object):
	def __init__(self,args,dataset):
		self.train = True if args.phase=='train' else False;
		self.dataset_dir = args.dataset_dir;
		self.corpus_dir = args.corpus_dir;
		self.sentence_length = args.sentence_length;
		self.vocabulary_size = args.vocabulary_size;
		self.batch_size = args.batch_size;
		self.sample_file = self.corpus_dir+dataset+'.pickle'
		self.filtered_file = self.corpus_dir+dataset+'-length{}-vocab{}.pickle'.format(args.sentence_length,args.vocabulary_size);
		self.sentences = [];
		self.word2idx = {};
		self.idx2word= {};
		self.idx_count = {};
		self.count = len(self.sentences);
		self.batches = int(self.count*1.0/args.batch_size);
		self.index = 0;
		self.indices = list(range(self.count));
		if(dataset=='cornell'):
			if not os.path.isfile(self.filtered_file):
				print('Constructing the full dataset......');
				if not os.path.isfile(self.sample_file):
					cornell = Cornell(self.dataset_dir);
					self.create_corpus(cornell.conversations);
					self.save_samples(self.sample_file);
				else:
					self.load_samples(self.sample_file);
				self.filter(args.sentence_length,args.vocabulary_size);
				self.save_samples(self.filtered_file);
			else:
				self.load_samples(self.filtered_file);
		print('The number of training samples is',self.get_data_size());
		print('The vocabulary_size is',self.get_vocabulary_size());

	def reset(self):
		self.index = 0
		np.random.shuffle(self.indices);

	def next_batch(self):
		if(self.index+self.batch_size<=self.count):
			start = self.index;
			end = self.index+self.batch_size;
			current = self.indices[start:end];
			enocder_inputs = [];
			decoder_inputs = [];
			decoder_targets = [];
			decoder_weights = [];
			for i in range(self.batch_size):
				encoder_inputs.append([self.pad]*(self.sentence_length-len(sentences[current[i]][0]))+sentences[current[i]][0]);
				decoder_inputs.append([self.go]+sentences[current[i]][1]+[self.eos]+[self.pad]*(self.sentence_length-len(sentences[current[i]][1])-2));
				decoder_targets.append(sentences[current[i]][1]+[self.eos]+[self.pad]*(self.sentence_length-len(sentences[current[i]][1])-1));
				decoder_weights.append([1.0]*(len(sentences[current[i][1]])+1)+[0.0]*(self.sentence_length-len(sentences[current[i]][1])-1));
			return encoder_inputs,decoder_inputs,decoder_targets,decoder_weights;

	def get_data_size(self):
		return len(self.sentences);

	def get_vocabulary_size(self):
		return len(self.word2idx);

	def get_word(self,idx):
		return idx2word[idx];

	def create_corpus(self,conversations):
		self.pad = self.get_word_id('<pad>'); 
		self.go = self.get_word_id('<go>');
		self.eos = self.get_word_id('<eos>'); 
		self.unknown = self.get_word_id('<unknown>'); 
		for conversation in conversations:
			self.extract_conversation(conversation);

	def extract_conversation(self, conversation):
		for i in range(len(conversation['lines'])-1):
			input_line  = conversation['lines'][i];
			target_line = conversation['lines'][i+1];
			inputs  = self.extract_text(input_line['text']);
			targets = self.extract_text(target_line['text']);
			if inputs and targets:
				self.sentences.append([inputs, targets]);

	def extract_text(self, line):
		sentences = [];
		sentence_tokens = nltk.sent_tokenize(line)
		for i in range(len(sentence_tokens)):
			tokens = nltk.word_tokenize(sentence_tokens[i]);
			words = [];
			for token in tokens:
				words.append(self.get_word_id(token));
			sentences.append(words)
		return sentences;

	def filter(self,sentence_length,vocabulary_size):
		def merge(sentences,question=False):
			merged = [];
			if question:
				sentences = reversed(sentences);
			for sentence in sentences:
				if len(merged)+len(sentence)<=sentence_length:
					if question:
						merged = sentence+merged
					else:
						merged = merged+sentence
				else:
					for word in sentence:
						self.idx_count[word] -= 1
			return merged;

		samples = [];
		for inputs,targets in self.sentences:
			inputs = merge(inputs,question=True)
			targets = merge(targets,question=False);
			samples.append([inputs,targets]);
		words = [];
		special_tokens = {self.pad, self.go, self.eos, self.unknown};
		mapping = {};
		new_id = 0;

		selected_word_ids = collections \
			.Counter(self.idx_count) \
			.most_common(vocabulary_size or None);
		selected_word_ids = {k for k, v in selected_word_ids if v>1};
		selected_word_ids |= special_tokens;

		for word_id,count in [(i, self.idx_count[i]) for i in range(len(self.idx_count))]:
			if word_id in selected_word_ids:
				mapping[word_id] = new_id;
				word = self.idx2word[word_id];
				del self.idx2word[word_id];
				self.word2idx[word] = new_id;
				self.idx2word[new_id] = word;
				new_id += 1;
			else:
				mapping[word_id] = self.unknown;
				del self.word2idx[self.idx2word[word_id]];
				del self.idx2word[word_id];

		def replace_words(words):
			valid = False;
			for i,word in enumerate(words):
				words[i] = mapping[word];
				if words[i]!=self.unknown:
					valid = True;
			return valid;

		self.sentences.clear();
		for inputs, targets in samples:
			valid = True;
			valid &= replace_words(inputs);
			valid &= replace_words(targets);
			valid &= targets.count(self.unknown)==0 
			if valid:
				self.sentences.append([inputs,targets]);
		self.idx_count.clear();

	def save_samples(self,file_name):
		with open(os.path.join(file_name),'wb') as f:
			data = {'word2idx': self.word2idx,'idx2word': self.idx2word,'idx_count': self.idx_count,'sentences': self.sentences};
			pickle.dump(data,f, -1);

	def load_samples(self,file_name):
		with open(os.path.join(file_name),'rb') as f:
			data = pickle.load(f);
			self.word2idx = data['word2idx'];
			self.idx2word = data['idx2word'];
			self.idx_count = data.get('idx_count', None);
			self.sentences = data['sentences'];
			self.pad = self.word2idx['<pad>'];
			self.go = self.word2idx['<go>'];
			self.eos = self.word2idx['<eos>'];
			self.unknown = self.word2idx['<unknown>'];

	def get_word_id(self,word,create=True):
		word = word.lower();
		if not create:
			word_id = self.word2idx.get(word,self.unknown);
		elif word in self.word2idx:
			word_id = self.word2idx[word];
			self.idx_count[word_id] += 1;
		else:
			word_id = len(self.word2idx);
			self.word2idx[word] = word_id;
			self.idx2word[word_id] = word;
			self.idx_count[word_id] = 1;
		return word_id;

def train_data(args):
	dataset = Dataset(args,args.dataset);
	return dataset;
