
import numpy as np

from keras import backend as K
from keras.engine.topology import Layer
from keras import activations
from keras import initializers
from keras import regularizers
from keras import metrics

import tensorflow as tf

class TargetEmbedding(Layer):
	def __init__(self, source_embedding = None, num_classes = None, num_samples=100, mode='sampled_softmax', **kwargs):
		self.num_classes = num_classes
		self.num_samples = num_samples
		self.source_embedding = source_embedding
		self.mode = mode
		self.supports_masking = True
		super(TargetEmbedding, self).__init__(**kwargs)

	def get_num_classes(self):
		return self.num_classes if self.num_classes else self.source_embedding.input_dim

	def get_embeddings(self):
		return self.embeddings if self.source_embedding is None else self.source_embedding.embeddings[:self.get_num_classes()]

	def compute_output_shape(self, input_shape):
		if self.mode == 'variants':
			features_shape, variants_shape = input_shape
			return variants_shape
		elif self.mode == 'full':
			return input_shape[:-1] + (self.get_num_classes(),)
		else:
			# pass computation to loss
			return input_shape

	def call(self, inputs, mask=None):
		if self.mode == 'variants':
			features, variants = inputs
			weights = K.gather(self.get_embeddings(), variants)
			biases = K.gather(self.bias, variants)
			return K.bias_add(K.dot(features, K.transpose(weights)), biases)
		elif self.mode == 'full':
			return K.bias_add(K.dot(inputs, K.transpose(self.get_embeddings())), self.bias)
		else:
			# pass computation to loss
			return inputs

	def build(self, input_shape):
		if self.source_embedding is None:
			if self.mode == 'variants':
				features_shape, variants_shape = input_shape
				embeddings_dim = features_shape[-1]
			else:
				embeddings_dim = input_shape[-1]
			
			self.embeddings = self.add_weight(shape=(self.get_num_classes(), embeddings_dim),
				initializer = 'glorot_normal',
				name='embeddings')
		self.bias = self.add_weight((self.get_num_classes(),), 
			initializer=initializers.Constant(value=1.0 / np.arange(1, self.get_num_classes()+1) - 1.0),
			name='bias')

	def get_config(self):
		config = {"num_classes": self.num_classes,
			"num_samples": self.num_samples,
			"mode": self.mode}
		base_config = super(TargetEmbedding, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	def loss(self, y_true, y_pred):
		if self.mode in ['variants', 'full']:
			return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
		else:
			pred_shape = K.shape(y_pred)
			flat_y_true = K.reshape(y_true, (-1,1))
			flat_y_pred = K.reshape(y_pred, (-1,pred_shape[-1]))
			flat_loss = tf.nn.sampled_softmax_loss(self.get_embeddings(), self.bias, 
				flat_y_true, flat_y_pred, 
				self.num_samples, self.num_classes)
			return K.reshape(flat_loss, pred_shape[:-1])
	
	def calc_y_pred(self, x):
		return K.bias_add(K.dot(x, K.transpose(self.get_embeddings())), self.bias)

	def accuracy(self, y_true, y_pred):
		if self.mode in ['variants', 'full']:
			return metrics.sparse_categorical_accuracy(y_true, y_pred)
		else:
			return tf.cond(K.learning_phase(),
				#do nothing in train phase lest metric calculation dominate loss
				lambda : K.cast(K.equal(y_true, y_true)[...,0], K.floatx()),
				lambda : metrics.sparse_categorical_accuracy(y_true, self.calc_y_pred(y_pred)))
		
	def entropy(self, y_true, y_pred):
		if self.mode in ['variants', 'full']:
			return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
		else:
			return tf.cond(K.learning_phase(),
				#do nothing in train phase lest metric calculation dominate loss
				lambda : K.cast(K.not_equal(y_true, y_true)[...,0], K.floatx()),
				lambda : K.sparse_categorical_crossentropy(y_true, self.calc_y_pred(y_pred), from_logits=True))


