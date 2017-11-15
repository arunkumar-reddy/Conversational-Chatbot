import os;
import ast;

class Cornell():
	def __init__(self,dataset_dir):
		self.lines = self.load_lines(os.path.join(dataset_dir,'movie_lines.txt'));
		self.conversations = self.load_conversations(os.path.join(dataset_dir,'movie_conversations.txt'));

	def load_lines(self,filename):
		lines = {};
		fields = ['lineID','characterID','movieID','character','text'];
		with open(filename,'r',encoding='iso-8859-1') as f:
			for line in f:
				values = line.split(' +++$+++ ');
				line_object = {};
				for i,field in enumerate(fields):
					line_object[field] = values[i];
				lines[line_object['lineID']] = line_object;
			return lines;

	def load_conversations(self,filename):
		conversations = [];
		fields = ['character1ID','character2ID','movieID','utteranceIDs'];
		with open(filename,'r',encoding='iso-8859-1') as f:
			for line in f:
				values = line.split(' +++$+++ ');
				conversation_object = {};
				for i,field in enumerate(fields):
					conversation_object[field] = values[i];
				lineIds = ast.literal_eval(conversation_object['utteranceIDs']);
				conversation_object['lines'] = [];
				for lineId in lineIds:
					conversation_object['lines'].append(self.lines[lineId]);
				conversations.append(conversation_object);
			return conversations;