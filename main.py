import os, glob
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable gpu allocation log information printing.
import tensorflow as tf
import data
import random
from configs import FLAGS, PARAMS
import numpy as np
from utils import *
from models.gcn import GCN

def main(self):
    if len(FLAGS.dataset) =='dataset':
        sys.exit("input dataset name : python main.py <dataset>")
    
    # Load data and settings
    
    datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']

    if FLAGS.dataset not in datasets:
        sys.exit("check your dataset name : ['20ng', 'R8', 'R52', 'ohsumed', 'mr']")

    # Set random seed
    seed = random.randint(1, 2019)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
    FLAGS.dataset)


    shapes = {'feature_shape': features[2], 'label_shape': y_train.shape[1]}

    PARAMS.append(shapes)
    PARAMS[0]['input_dim'] = features[2][1]
    
    # Build model

    if FLAGS.model == 'GCN':
        support = preprocess_adj(adj)
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
    
    
    model = GCN(PARAMS)
    sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(FLAGS.epochs):
            train_pred, train_acc, train_loss = run(sess, model, support, features, y_train, train_mask, FLAGS.dropout)
            val_pred, val_acc, val_loss = run(sess, model, support, features, y_val, val_mask, 0, train=False)
            print('epoch %d : train acc %f, train loss %f, val acc %f, val loss %f:' %(epoch, train_acc, train_loss, val_acc, val_loss))
    
        test_pred, test_acc, test_loss = run(sess, model, support, features, y_test, test_mask, 0, train=False)
        print('test acc %f, test loss %f' %(test_acc, test_loss))
        
def run(sess, model, support, features, labels, mask, dropout, train=True):
        
    feed_dict = {model.support: support,
                 model.features: features,
                 model.labels: labels,
                 model.labels_mask: mask,
                 model.dropout: dropout,
                 model.num_features_nonzero: features[1].shape
                }
    if train:
        outs_val = sess.run([model.opt_op, model.predicts, model.accuracy, model.loss], feed_dict=feed_dict)
        return outs_val[1], outs_val[2], outs_val[3]
    else:
        outs_val = sess.run([model.predicts, model.accuracy, model.loss], feed_dict=feed_dict)
        return outs_val[0], outs_val[1], outs_val[2]
    
if __name__ =='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
    