import tensorflow as tf;
import numpy as np;
import os;
import sys;
import pickle;

from tqdm import tqdm;
from gensim.models import KeyedVectors as Word2Vec;
from nn import *;
from dataset import *;

class Model(object):
	def __init__(self,params,dataset,phase):
		self.params = params;
		self.dataset = dataset;
		self.phase = phase;
		self.model = params.model;
		self.rnn_units = params.rnn_units;
		self.hidden_size = params.hidden_size;
		self.word_embed = params.word_embed;
		self.batch_size = params.batch_size;
		self.vocabulary_size = dataset.get_vocabulary_size();
		self.sentence_length = params.sentence_length;
		self.global_step = tf.Variable(0,name='global_step',trainable=False);
		self.save_dir = os.path.join(params.save_dir,params.solver+'/');
		self.saver = tf.train.Saver(max_to_keep = 100);
		self.build();

	def build(self):
		print('Building the model......');
		encoder_inputs = [tf.placeholder(tf.int32,[self.batch_size]) for i in range(self.sentence_length)];
		decoder_inputs = [tf.placeholder(tf.int32,[self.batch_size]) for i in range(self.sentence_length)];
		decoder_targets = [tf.placeholder(tf.int32,[self.batch_size]) for i in range(self.sentence_length)];
		decoder_weights = [tf.placeholder(tf.float32,[self.batch_size]) for i in range(self.sentence_length)];
		word2vec = self.load_embeddings();

		if(self.model=='lstm'):
			encoder_cells = [];
			for i in range(self.rnn_units):
				encoder_cell = tf.contrib.rnn.LSTMCell(self.hidden_size,activation=tf.nn.tanh);
				if(self.phase=='train'):
					encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell,input_keep_prob=1.0,output_keep_prob=0.9);
				encoder_cells.append(encoder_cell);
			decoder_cells = [];
			for i in range(self.rnn_units):
				decoder_cell = tf.contrib.rnn.LSTMCell(self.hidden_size,activation=tf.nn.tanh);
				if(self.phase=='train'):
					decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell,input_keep_prob=1.0,output_keep_prob=0.9);
				decoder_cells.append(decoder_cell);
		else:
			encoder_cells = [];
			for i in range(self.rnn_units):
				encoder_cell = tf.contrib.rnn.GRUCell(self.hidden_size,activation=tf.nn.tanh);
				if(self.phase=='train'):
					encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell,input_keep_prob=1.0,output_keep_prob=0.9);
				encoder_cells.append(encoder_cell);
			encoder = tf.contrib.rnn.MultiRNNCell(encoder_cells);
			decoder_cells = [];
			for i in range(self.rnn_units):
				decoder_cell = tf.contrib.rnn.GRUCell(self.hidden_size,activation=tf.nn.tanh);
				if(self.phase=='train'):
					decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell,input_keep_prob=1.0,output_keep_prob=0.9);
				decoder_cells.append(decoder_cell);
			decoder = tf.contrib.rnn.MultiRNNCell(decoder_cells);

		encoder = tf.contrib.rnn.MultiRNNCell(encoder_cells);
		embedding = tf.get_variable('embedding',initializer=word2vec);
		embedded_encoder_inputs = [];
		for i in range(len(encoder_inputs)):
			embedded_encoder_inputs.append(tf.nn.embedding_lookup(embedding,encoder_inputs[i]));
		encoder_output,encoder_state = tf.contrib.rnn.static_rnn(encoder,embedded_encoder_inputs,dtype=tf.float32);	
		embedded_decoder_inputs = [];
		for i in range(len(decoder_inputs)):
			embedded_decoder_inputs.append(tf.nn.embedding_lookup(embedding,decoder_inputs[i]));
		decoder = tf.contrib.rnn.MultiRNNCell(decoder_cells);
		state = encoder_state;
		outputs = [];
		words = [];
		with tf.variable_scope('decoder') as scope:
			for decoder_input in embedded_decoder_inputs:
				if(self.phase=='train'):
					decoder_output,state = decoder(decoder_input,state);
					embedded_decoder_output = fully_connected(decoder_output,self.word_embed,'decode');
					decode_embedding = tf.transpose(embedding);
					output = tf.matmul(embedded_decoder_output,decode_embedding);
					words.append(tf.argmax(output,1));
					outputs.append(output);
				else:
					if(i==0):
						output,state = decoder(decoder_input,state);
					else:
						output,state = decoder(output,state);
					embedded_decoder_output = fully_connected(decoder_output,self.word_embed,'decode');
					decode_embedding = tf.transpose(embedding);
					output = tf.matmul(embedded_decoder_output,decode_embedding);
					words.append(tf.argmax(output,1));
					outputs.append(output);
				scope.reuse_variables();

		losses = [];
		for i in range(len(outputs)):
			cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs[i],labels=decoder_targets[i]);
			losses.append(cross_entropy*decoder_weights[i]);

		loss = tf.add_n(losses);
		total = tf.add_n(decoder_weights);
		total += 1e-12; 
		loss /= total;
		loss = tf.reduce_mean(loss)/self.batch_size;

		self.encoder_inputs = encoder_inputs;
		self.decoder_inputs = decoder_inputs;
		self.decoder_targets = decoder_targets;
		self.decoder_weights = decoder_weights;
		self.outputs = outputs;
		self.words = words;
		self.loss = loss;

		if self.params.solver == 'adam':
			solver = tf.train.AdamOptimizer(self.params.learning_rate);
		elif self.params.solver == 'momentum':
			solver = tf.train.MomentumOptimizer(self.params.learning_rate,self.params.momentum);
		elif self.params.solver == 'rmsprop':
			solver = tf.train.RMSPropOptimizer(self.params.learning_rate,self.params.weight_decay,self.params.momentum);
		else:
			solver = tf.train.GradientDescentOptimizer(self.params.learning_rate);

		tensorflow_variables = tf.trainable_variables();
		gradients,_ = tf.clip_by_global_norm(tf.gradients(self.loss,tensorflow_variables),3.0);
		optimizer = solver.apply_gradients(zip(gradients,tensorflow_variables),global_step=self.global_step);
		self.optimizer = optimizer;
		print('Model completed......');

	def Train(self,sess):
		print("Training the model.......");
		epochs = self.params.epochs;
		for epoch in tqdm(list(range(epochs)),desc='Epoch'):
			for i in tqdm(list(range(dataset.batches)),desc='Batch'):
				batch = dataset.next_batch();
				feed_dict = self.feed(batch);
				global_step,loss,outputs,_ = sess.run([self.global_step,self.loss,self.words,self.optimizer],feed_dict=feed_dict);
			print(' Step = %f,Loss = %f'%(global_step,loss));
			if (global_step%self.params.save_period)==0:
					print('Outputs for the current batch...');
					for i in range(len(self.batch_size)):
						sentence = '';
						for j in range(self.sentence_length):
							sentence += dataset.get_word(words[j][i])+' ';
						print(sentence);
					self.save(sess);
			data.reset();
		self.save(sess);
		print('Training completed......');				

	def Test(self,sess):
		pass;

	def load_embeddings(self):
		print('Loading pre-trained word embeddings......');
		self.word_embedding_file = './word2vec/word2vec.pickle';
		if not os.path.isfile(self.word_embedding_file):
			google_word2vec = Word2Vec.load_word2vec_format(self.params.word_embedding_file,binary=True);
			word2vec = np.random.uniform(-0.25,0.25,(self.vocabulary_size,self.word_embed)).astype(np.float32);
			count = 0;
			for word in google_word2vec.vocab.keys():
				if word in self.dataset.word2idx:
						word2vec[self.dataset.word2idx[word]] = google_word2vec[word];
						count += 1;
			print('Word embeddings loaded for %d words......'%count);
			
			with open(os.path.join(self.word_embedding_file),'wb') as f:
				pickle.dump(word2vec,f,-1);
				return word2vec;
		else:
			with open(os.path.join(self.word_embedding_file),'rb') as f:
				word2vec = pickle.load(f);
				return word2vec;

	def load(self,sess):
		print('Loading model.....');
		checkpoint = tf.train.get_checkpoint_state(self.save_dir);
		if checkpoint is None:
			print("Error: No saved model found. Please train first.");
			sys.exit(0);
		self.saver.restore(sess, checkpoint.model_checkpoint_path);

	def save(self,sess):
		print(('Saving model to %s......' % self.save_dir));
		self.saver.save(sess,self.save_dir,self.global_step);

	def feed(self,batch):
		encoder_inputs = [];
		decoder_inputs = [];
		decoder_targets = [];
		decoder_weights = [];
		for i in range(self.sentence_length):
			encoder_input = [];
			decoder_input = [];
			decoder_target = [];
			decoder_weight = [];
			for j in range(self.batch_size):
				encoder_input.append(batch.encoder_inputs[j][i]);
				decoder_input.append(batch.decoder_inputs[j][i]);
				decoder_target.append(batch.decoder_targets[j][i]);
				decoder_weight.append(batch.decoder_weights[j][i]);
			encoder_inputs.append(encoder_input);
			decoder_inputs.append(decoder_input);
			decoder_targets.append(decoder_target);
			decoder_weights.append(decoder_weight);
		
		feed[self.encoder_inputs] = encoder_inputs;
		feed[self.decoder_inputs] = decoder_inputs;
		feed[self.decoder_targets] = decoder_targets;
		feed[self.decoder_weights] = decoder_weights;
		return feed;
