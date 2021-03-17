"""
Handles reading and writing datasets in the CoNLL-U format (v1 and v2) [0]:

	dataset = conllu.Dataset(path)
	graphs = [graph for graph in dataset.gen_graphs()]

	another = conllu.Dataset(path)
	another.write_graphs(graphs)

The graphs in question are nx.DiGraph instances, one for each sentence.

[0] http://universaldependencies.org/format.html
"""

from collections import namedtuple

import networkx as nx

import ud



"""
Represents a word, i.e. a row with an integer ID (excluding multiword tokens
and empty nodes). With the exception of ID and HEAD which are integers, and
FEATS which is a dict, the other fields are non-empty strings.
"""
Word = namedtuple('Word', [
	'ID', 'FORM', 'LEMMA',
	'UPOSTAG', 'XPOSTAG', 'FEATS',
	'HEAD', 'DEPREL', 'DEPS', 'MISC'])



class ConlluError(ValueError):
	"""
	Raised when there is a problem with reading or writing a file in CoNLL-U
	format. Should be inited with a human-friendly error message.
	"""
	pass



class Dataset:

	def __init__(self, file_path, doc_id_prefix, sent_id_prefix, node_labels_type, ud_version=2):
		"""
		Constructor. Expects the path to the .conllu dataset file. The latter
		is not opened until one of the gen_* methods is invoked.

		The dataset's POS tags are checked against the UD version specified by
		the keyword argument.
		"""
		self.file_path = file_path
		self.doc_id_prefix = doc_id_prefix
		self.sent_id_prefix = sent_id_prefix
		self.node_labels_type = node_labels_type

		if ud_version == 1:
			self.POS_TAGS = ud.POS_TAGS_V1
			self.DEP_RELS = ud.DEP_RELS_V1
			self.MORPH_FEATURES = ud.MORPH_FEATURES_V1
		elif ud_version == 2:
			self.POS_TAGS = ud.POS_TAGS_V2
			self.DEP_RELS = ud.DEP_RELS_V2
			self.MORPH_FEATURES = ud.MORPH_FEATURES_V2
		else:
			raise ValueError('Unknown UD version: {}'.format(ud_version))

		self.ud_version = ud_version


	"""
	Reading ConLL-U
	"""

	def _read_word(self, line):
		"""
		Returns a Word named tuple from the given line; the latter is expected
		to be a [] of strings. If the line does not define a word with an
		integer ID, the method returns None.

		Raises a ValueError or an AssertionError if the line does not conform
		to the format specified by the UD version in self.ud_version.

		Helper for self.gen_sentences.
		"""
		assert len(line) == 10 and all([item for item in line])

		if line[0].isdigit():  # id
			line[0] = int(line[0])
		else:
			assert all([c.isdigit() or c in ('-', '.') for c in line[0]])

		assert line[3] in self.POS_TAGS or line[3] == '_'  # upostag

		if line[5] == '_':  # feats
			line[5] = {}
		else:
			line[5] = {key: frozenset(value.split(','))
				for key, value in map(lambda x: x.split('='), line[5].split('|'))}

		if line[6] != '_':  # head
			line[6] = int(line[6])

		return Word._make(line)


	def gen_sentences(self, include_multiwords=False):
		"""
		Generator that yields a tuple of Word named tuples for each sentence in
		the dataset.

		Raises a ConlluError if the file does not conform to the CoNLL-U format
		of the version specified by self.ud_version.
		"""
		sent = []

		try:
			with open(self.file_path, encoding='utf-8') as f:
				doc_id = ''
				sent_id = ''
				sentence_counter = 0
				for line_num, line in enumerate(map(lambda x: x.strip(), f)):
					if line.startswith('#'):  # comments
						if self.doc_id_prefix in line:
							doc_id = line.partition(self.doc_id_prefix)[-1]
						elif self.sent_id_prefix in line:
							sent_id = line.partition(self.sent_id_prefix)[-1]
						continue

					elif line:
						word = self._read_word(line.split('\t'))
						if include_multiwords or type(word.ID) is int:
							sent.append(word)

					else:  # empty lines mark sentence boundaries
						if doc_id == '':
							doc_id = 0
						if sent_id == '':
							sent_id = sentence_counter
						sentence_counter += 1
						doc_sent_name = str(doc_id) + '_' + str(sent_id)
						yield doc_sent_name, tuple(sent)
						sent = []

		except IOError:
			raise ConlluError('Could not open {}'.format(self.file_path))

		except (AssertionError, ValueError):
			raise ConlluError('Could not read {}:{}'.format(self.file_path, line_num))


	def gen_graphs(self, edgeless=False):
		"""
		Generator that yields a nx.DiGraph instance for each sentence in the
		dataset. If the flag is set to True, the graph will have only nodes.

		The graph nodes are the indices (0 being the root) of the respective
		words; FORM, LEMMA, UPOSTAG, and FEATS are stored as node attributes.
		The graph edges are directed from parent to child; DEPREL is stored as
		an edge attribute.

		Unlike self.gen_sentences, the output here has an explicit root node
		with its form (\xa0), lemma (\xa0), and POS tag (ROOT) which are not
		part of the UD standard, but still needed for the purposes of mstnn.

		Raises a ConlluError if the file does not conform to the CoNLL-U format
		of the version specified by self.ud_version.
		"""
		for doc_sent_name, sent in self.gen_sentences(include_multiwords=True):
			graph = nx.DiGraph()
			graph.add_node('0', label='\xa0')

			for word in sent:
				if type(word.ID) is int:
					# choose the type of the label
					if self.node_labels_type == 'FORM':
						node_label = word.FORM
					elif self.node_labels_type == 'LEMMA':
						node_label = word.LEMMA
					elif self.node_labels_type == 'POS':
						node_label = word.UPOSTAG

					graph.add_node(str(word.ID),
							label=node_label)

					if not edgeless and word.HEAD != '_':
						graph.add_edge(str(word.HEAD), str(word.ID), label=word.DEPREL)
				else:
					graph.graph[str(word.ID)] = str(word.ID)

			yield doc_sent_name, graph


	"""
	Writing ConLL-U
	"""

	@staticmethod
	def format_word(word):
		"""
		Returns a single-line string representing a parsed word in CoNLL-U
		format (excluding the newline character at the end). Expects a Word
		named tuple as its sole argument.

		This method is static and is used as a helper for self.format_sentence
		(and, hence, for self.write_sentence and self.write_graphs as well).
		"""
		if word.FEATS:
			feats = '|'.join(['{}={}'.format(key, ','.join(sorted(value)))
				for key, value in sorted(word.FEATS.items())])
		else:
			feats = '_'

		return '\t'.join([str(word.ID), word.FORM, word.LEMMA,
				word.UPOSTAG, word.XPOSTAG, feats,
				str(word.HEAD), word.DEPREL, word.DEPS, word.MISC])


	@staticmethod
	def format_sentence(sentence):
		"""
		Returns a multi-line string representing a parsed sentence in CoNLL-U
		format (including the empty line at the end). Expects a sequence of
		well-formated (i.e. no empty-string fields) Word named tuples.

		This method is static and is used as a helper for self.write_sentences
		(and, hence, for self.write_graphs as well).
		"""
		lines = [Dataset.format_word(word) for word in sentence]
		return '\n'.join(lines + ['', ''])


	def write_sentences(self, sentences):
		"""
		Writes the sentences to the dataset's file path. The sentences are
		expected to be a sequence of sequences of Word named tuples.

		Raises a ConlluError if the file cannot be written or if the sentences
		do not conform to the spec.
		"""
		try:
			with open(self.file_path, 'w', encoding='utf-8') as f:
				for sentence in sentences:
					assert all([isinstance(word, Word) for word in sentence])
					f.write(Dataset.format_sentence(sentence))

		except IOError:
			raise ConlluError('Could not write {}'.format(self.file_path))

		except AssertionError:
			raise ConlluError('Sentences must be sequences of Word tuples')


	def write_graphs(self, graphs):
		"""
		Writes the given sentence graphs to the dataset's file path. The graphs
		should be a sequence of DiGraph instances, each conforming to the spec
		outlined in self.gen_graphs.

		Raises a ConlluError if the file cannot be written or if the graphs do
		not conform to the spec.
		"""
		sentences = []

		for graph in graphs:
			sentence = []

			for key, data in graph.nodes(data=True):
				if key == 0:
					continue  # skip the ROOT

				try:
					edge = graph.in_edges(key, data=True)[0]
					assert 'FORM' in data
					assert 'LEMMA' in data
					assert 'UPOSTAG' in data
				except (StopIteration, AssertionError):
					raise ConlluError('Graphs do not conform to the spec')

				sentence.append(Word(key,
					data['FORM'],
					data['LEMMA'],
					data['UPOSTAG'],
					data['XPOSTAG'] if 'XPOSTAG' in data else '_',
					data['FEATS'],
					edge[0],
					edge[2]['DEPREL'] if 'DEPREL' in edge[2] else '_',
					data['DEPS'] if 'DEPS' in data else '_',
					data['MISC'] if 'MISC' in data else '_'))

			for key, word in graph.graph.items():
				try:
					key = int(key.split('-')[0])
					i = next((i for i, w in enumerate(sentence) if w.ID == key))
					sentence.insert(i, word)
				except (StopIteration, ValueError):
					raise ConlluError('Graphs do not conform to the spec')

			sentences.append(sentence)

		self.write_sentences(sentences)
