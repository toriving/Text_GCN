import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'GCN', 'Model name')
flags.DEFINE_string('dataset', 'dataset', 'Dataset name')
flags.DEFINE_float('learning_rate', 0.02, 'Learning rate')
flags.DEFINE_string('optimizer','adam', 'Optimizer name')
flags.DEFINE_integer('epochs', 200, 'Number of the epoch train')
flags.DEFINE_integer('train_step', 20, 'Number of the max step')
flags.DEFINE_integer('n_class', 2, 'Number of the class')
flags.DEFINE_integer('batch_size', 512, 'Number of the batch size')
flags.DEFINE_integer('embedding_size', 300, 'Word embedding size')
flags.DEFINE_integer('hidden_dim', 200, 'Hidden unit size')
flags.DEFINE_integer('window_size', 20, 'Convolution window size')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate')
tf.app.flags.DEFINE_bool('train', True, 'run trainig')
tf.app.flags.DEFINE_string('output_path', './output/', 'output path')
tf.app.flags.DEFINE_string('ckpt_path', './output/ckpt/', 'checkpoint path')
tf.app.flags.DEFINE_string('best_ckpt_path', './output/best_ckpt/', 'best_checkpoint path')

PARAMS = [{'input_dim': 300,
            'output_dim': 200,
            'featureless' : True,
            'final': False,
            'dropout_rate': 0.5},
          
          {'input_dim': 200,
            'output_dim': FLAGS.n_class,
            'featureless' : False,
            'final' : True,
            'dropout_rate': 0.5}
         ]

