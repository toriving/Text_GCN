import tensorflow as tf
from configs import FLAGS
# support = A (Adjacency matrix PPI, tf-idf)
# input = X (Onehot encoding docu and words)
# weight matrix = W 
    
class GCN:
    def __init__(self, parameters):
        self.parameters = parameters
        self._build_placeholder()
        self.build(parameters)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.opt_op = self.optimizer.minimize(self.loss)
        
    def build(self, parameters, scope="GCN"):
        with tf.variable_scope(scope):       
            layer1 = self._GraphConvolutionLayer(self.features, sparse_input=True, featureless=True)
            layer2 = self._GraphConvolutionLayer(layer1, final=True)
            
            self.outputs = layer2
            self.predicts = tf.cast(tf.argmax(self.outputs, axis=1), tf.int32)
            self.accuracy = self._masked_accuracy(self.outputs, self.labels, self.labels_mask)
            self.loss = self._masked_softmax_cross_entropy(self.outputs, self.labels, self.labels_mask)

    
    def _GraphConvolutionLayer(self, inputs, final=False, sparse_input=False, featureless=False):
        if sparse_input:
            inputs = self._sparse_dropout(inputs, 1- self.dropout, self.num_features_nonzero)
        else:
            inputs = tf.nn.dropout(inputs, 1 - self.dropout)
            
        if final:
            W, b = self._build_weight([self.parameters[1]['input_dim'], self.parameters[1]['output_dim']], scope="final")
        else:
            W, b = self._build_weight([self.parameters[0]['input_dim'], self.parameters[0]['output_dim']])
            
        if featureless:
            pre_sup = W
        else:
            if sparse_input:
                pre_sup = tf.sparse_tensor_dense_matmul(inputs, W)
            else:
                pre_sup = tf.matmul(inputs, W)
 
        outputs = tf.sparse_tensor_dense_matmul(self.support, pre_sup) + b
            
        if not final:
            outputs = tf.nn.relu(outputs)

        return outputs
    
    
    def _sparse_dropout(self, x, keep_prob, noise_shape):
        """Dropout for sparse tensors."""
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(x, dropout_mask)
        return pre_out * (1./keep_prob)
    
    def _masked_softmax_cross_entropy(self, preds, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=preds, labels=labels)
        loss = tf.boolean_mask(loss, mask)
        return tf.reduce_mean(loss)


    def _masked_accuracy(self, preds, labels, mask):
        """Accuracy with masking."""
        prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
        accuracy = tf.cast(prediction, tf.float32)
        accuracy = tf.boolean_mask(accuracy, mask)
        return tf.reduce_mean(accuracy)

    def _build_weight(self, shape, scope="weight"):
         with tf.variable_scope(scope):
            W = tf.get_variable(name="W", shape=[shape[0], shape[1]], dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name="b", shape=[shape[1]], dtype=tf.float32, initializer = tf.zeros_initializer())
            return W, b

    
    def _build_placeholder(self):
        self.support = tf.sparse_placeholder(tf.float32)
        self.features = tf.sparse_placeholder(tf.float32, shape=tf.constant(self.parameters[-1]['feature_shape'], dtype=tf.int64))
        self.labels = tf.placeholder(tf.float32, shape=(None, self.parameters[-1]['label_shape']))
        self.labels_mask = tf.placeholder(tf.int32, shape=[None])
        self.dropout = tf.placeholder_with_default(0., shape=())
        self.num_features_nonzero = tf.placeholder(tf.int32)

