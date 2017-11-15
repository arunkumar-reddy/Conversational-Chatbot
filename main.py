import sys;
import os;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3';

import tensorflow as tf;
import argparse;
from model import *;
from dataset import *;

def main(argv):
	parser = argparse.ArgumentParser();
	'''Model Architecture'''
	parser.add_argument('--phase', default='train', help='Train,Validate or Test');
	parser.add_argument('--load', action='store_true', default=False, help='Load the trained model');
	parser.add_argument('--model', default='lstm', help='Recurrent neural network model: Can be lstm, gru')
	parser.add_argument('--dataset', default='cornell', choices=['cornell','opensubtitles'], help='Datatset to train the model upon');
	parser.add_argument('--rnn_units', type=int, default=2, help='Number of RNNs to use: Can be 1 or 2');
	parser.add_argument('--hidden_size', type=int, default=1024, help='Hidden state dimension in each RNN unit');
	parser.add_argument('--word_embed', type=int, default=300, help='Dimension of the word embedding');
	parser.add_argument('--sentence_length', type=int, default=10, help='Maximum Length of the generated caption');
	parser.add_argument('--vocabulary_size', type=int, default=40000, help='Maximum vocabulary size');
	'''Files and Directories'''
	parser.add_argument('--dataset_dir', default='/home/arun/Datasets/Cornell/', help='Directory containing the data');
	parser.add_argument('--corpus_dir', default='./corpus/cornell/', help='Directory to store the processed data');
	parser.add_argument('--word_embedding_file', default='/home/arun/Datasets/Word2Vec/word2vec.bin', help='Word vector file for word embeddings');
	parser.add_argument('--save_dir', default='./models/', help='Directory to contain the trained model');
	parser.add_argument('--save_period', type=int, default=5000, help='Period to save the trained model');
	'''Hyper Parameters'''
	parser.add_argument('--solver', default='sgd', help='Gradient Descent Optimizer to use: Can be adam, momentum, rmsprop or sgd') 
	parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs');
	parser.add_argument('--batch_size', type=int, default=256, help='Batch size');
	parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate');
	parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay');
	parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (for some optimizers)'); 
	parser.add_argument('--decay', type=float, default=0.9, help='Decay (for some optimizers)'); 
	parser.add_argument('--batch_norm', action='store_true', default=False, help='Turn on to use batch normalization');

	args = parser.parse_args();
	with tf.Session() as sess:
		if(args.phase=='train'):
			data = train_data(args);
			model = Model(args,data,'train');
			sess.run(tf.global_variables_initializer());
			if(args.load):
				model.load(sess);
			#model.Train(sess);
		else:
			model = Model(args,'test');
			sess.run(tf.global_variables_initializer());
			model.load(sess);
			model.Test(sess);

main(sys.argv);