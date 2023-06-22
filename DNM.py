'''
@author: 
Zhengxin Zhang (zzxynu@gmail.com)

@references: 
Hao Wu, Zhengxin Zhang, Jiacheng Luo, Kun Yue, Ching-Hsien Hsu. 
Multiple Attributes QoS Prediction via Deep Neural Model with Contexts. 
IEEE Transactions on Services Computing, 2021, DOI: 10.1109/TSC.2018.2859986
'''

import math
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from time import time
import argparse
import LoadData as DATA
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run DNM.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='5/',
                        help='Choose a dataset with given training ratio.')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size (128-512).')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--layers', nargs='?', default='[128]',
                        help="Size of each layer.")
    parser.add_argument('--layers_rt', nargs='?', default='[64]',
                        help="Size of each layer.")
    parser.add_argument('--layers_tp', nargs='?', default='[64]',
                        help="Size of each layer.")
    parser.add_argument('--keep_prob', nargs='?', default='[1.0, 0.8]',
                        help='Keep probability for each deep layer and the Interaction layer. 1: no dropout. Note that the last index is for the Interaction layer.')
    parser.add_argument('--lamda', type=float, default=0,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='L1_loss',
                        help='Specify a loss type (L1_loss or L2_loss).')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=1,
                    help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--activation', nargs='?', default='relu',
                    help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
    parser.add_argument('--alpha', type=float, default=0.95,
                    help='The weight between two loss functions (0.95-0.99)')
    parser.add_argument('--wsdl_support', type=int, default=0,
                    help='Whether to use WSDL for pretraining service embeddings')
    return parser.parse_args()


class DNM(BaseEstimator, TransformerMixin):

    def __init__(self, features_M_user, features_M_service, hidden_factor, layers, layers_rt, layers_tp, loss_type, epoch, batch_size, learning_rate, lamda_bilinear,
                 keep_prob, optimizer_type, batch_norm, activation_function, verbose, alpha, random_seed=2019):
        # bind params to class
        self.batch_size = batch_size
        self.hidden_factor = hidden_factor
        self.layers = layers
        self.layers_rt = layers_rt
        self.layers_tp = layers_tp
        self.loss_type = loss_type
        self.features_M_user = features_M_user
        self.features_M_service = features_M_service
        self.lamda_bilinear = lamda_bilinear
        self.epoch = epoch
        self.random_seed = random_seed
        self.keep_prob = np.array(keep_prob)
        self.no_dropout = np.array([1 for i in range(len(keep_prob))])
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.verbose = verbose
        self.activation_function = activation_function
        self.alpha = alpha
        # performance of each epoch
        self.train_rmse, self.train_mae, self.test_rmse_rt, self.test_rmse_tp, self.test_mae, self.test_mae_all = [], [], [], [], [], []
        
        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.train_user_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_U
            self.train_service_features = tf.placeholder(tf.int32, shape=[None, None])  # None * feature_I
            self.train_labels1 = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1
            self.train_labels2 = tf.placeholder(tf.float32, shape=[None, 1])
            self.dropout_keep = tf.placeholder(tf.float32, shape=[None])
            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights = self._initialize_weights()

            # Model.
            # _________ Embedding Layer _____________
            left_embeddings = tf.nn.embedding_lookup(self.weights['left_embeddings'], self.train_user_features)
            right_embeddings = tf.nn.embedding_lookup(self.weights['right_embeddings'], self.train_service_features)
            # serid_embeddings = tf.nn.embedding_lookup(self.weights['serid_embeddings'], self.train_serid_features)
            all_embeddings = tf.concat([left_embeddings, right_embeddings], 1)
            self.summed_features_left_emb = tf.reduce_sum(left_embeddings, 1)  # None * K
            self.summed_features_right_emb = tf.reduce_sum(right_embeddings, 1)  # None * K
            self.summed_features_all_emb = tf.reduce_sum(all_embeddings, 1)
            
            # _________ Interaction Layer _____________
            self.CSMF = tf.multiply(self.summed_features_left_emb, self.summed_features_right_emb)  # None * K
            self.CSMF = tf.expand_dims(self.CSMF, 1)
            self.summed_features_all_emb = tf.expand_dims(self.summed_features_all_emb, 1)
            self.CSMF = tf.concat([self.summed_features_all_emb, self.CSMF], 1)
            self.CSMF = tf.reduce_sum(self.CSMF, 1)
            if self.batch_norm:
                self.CSMF = self.batch_norm_layer(self.CSMF, train_phase=self.train_phase, scope_bn='bn_CSMF')
            self.CSMF = tf.nn.dropout(self.CSMF, self.dropout_keep[-1])  # dropout at the interactin layer
            self.PL = self.CSMF
            
            # ________ Perception Layers __________
            for i in range(0, len(self.layers)):
                self.PL = tf.add(tf.matmul(self.PL, self.weights['layer_%d' % i]), self.weights['bias_%d' % i])  # None * layer[i] * 1
                if self.batch_norm:
                    self.PL = self.batch_norm_layer(self.PL, train_phase=self.train_phase, scope_bn='bn_%d' % i)  # None * layer[i] * 1
                self.PL = self.activation_function(self.PL)
                # self.PL = tf.nn.dropout(self.PL, self.dropout_keep[i])  # dropout at each Deep layer
            self.RT = self.PL
            self.TP = self.PL
            # ________ Task Layers __________
            for i in range(0, len(self.layers_rt)):
                self.RT = tf.add(tf.matmul(self.RT, self.weights['rt_layer_%d' % i]), self.weights['rt_bias_%d' % i])  # None * layer[i] * 1
                if self.batch_norm:
                    self.RT = self.batch_norm_layer(self.RT, train_phase=self.train_phase, scope_bn='rt_%d' % i)  # None * layer[i] * 1
                self.RT = self.activation_function(self.RT)
                # self.RT = tf.nn.dropout(self.RT, self.dropout_keep[i])  # dropout at each Deep layer
            self.RT = tf.matmul(self.RT, self.weights['rt_prediction'])  # None * 1

            for i in range(0, len(self.layers_tp)):
                self.TP = tf.add(tf.matmul(self.TP, self.weights['tp_layer_%d' % i]), self.weights['tp_bias_%d' % i])  # None * layer[i] * 1
                if self.batch_norm:
                    self.TP = self.batch_norm_layer(self.TP, train_phase=self.train_phase, scope_bn='tp_%d' % i)  # None * layer[i] * 1
                self.TP = self.activation_function(self.TP)
                # self.TP = tf.nn.dropout(self.TP, self.dropout_keep[i])  # dropout at each Deep layer
            self.TP = tf.matmul(self.TP, self.weights['tp_prediction'])  # None * 1
            # _________out_rt _________
            self.Out_RT = tf.reduce_sum(self.RT, 1, keepdims=True)  # None * 1
      
            # _________out_tp _________   
            self.Out_TP = tf.reduce_sum(self.TP, 1, keepdims=True)
            # self.Out_RT=self.Out_RT-tf.ones_like(self.Out_RT) 
            # self.Out_TP=self.Out_RT-tf.ones_like(self.Out_TP)
            
            self.joint_loss = 0
            # Compute the loss.
            if self.loss_type == 'L1_loss':
                self.loss1 = tf.reduce_sum(tf.abs(tf.subtract(self.train_labels1, self.Out_RT)))
                self.loss2 = tf.reduce_sum(tf.abs(tf.subtract(self.train_labels2, self.Out_TP))) 
                self.joint_loss = self.alpha * self.loss1 + (1 - self.alpha) * self.loss2
                
            if self.loss_type == 'L2_loss':
                self.loss1 = tf.reduce_sum(tf.square(tf.subtract(self.train_labels1, self.Out_RT)))
                self.loss2 = tf.reduce_sum(tf.square(tf.subtract(self.train_labels2, self.Out_TP))) 
                self.joint_loss = self.alpha * self.loss1 + (1 - self.alpha) * self.loss2
                
            if self.lamda_bilinear > 0:
                self.joint_loss = self.joint_loss + tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights['left_embeddings'], self.weights['right_embeddings']) 

            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.joint_loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.joint_loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.joint_loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.joint_loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()  # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print ("#params: %d" % total_parameters)

    def _initialize_weights(self):
        all_weights = dict()
        # without pretrain
        all_weights['left_embeddings'] = tf.Variable(tf.random_normal([self.features_M_user, self.hidden_factor], 0.0, 0.01), name='left_embeddings')
        all_weights['right_embeddings'] = tf.Variable(self.features_M_service, name='right_embeddings', dtype=np.float32)  # features_M * K
    
        # deep layers
        num_layer = len(self.layers)
        num_layer1 = len(self.layers_rt)
        num_layer2 = len(self.layers_tp)
        if num_layer > 0 and num_layer1 > 0 and num_layer2 > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor + self.layers[0]))
            glorot_rt = np.sqrt(2.0 / (self.hidden_factor + self.layers_rt[0]))
            glorot_tp = np.sqrt(2.0 / (self.hidden_factor + self.layers_tp[0]))
            # loc:mean  scale:biao zhun cha   size: out shape
            all_weights['layer_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.hidden_factor, self.layers[0])), dtype=np.float32)
            all_weights['bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.layers[0])), dtype=np.float32)  # 1 * layers[0]
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.layers[i - 1] + self.layers[i]))
                # 64 layer  per layer 64  size
                all_weights['layer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.layers[i - 1], self.layers[i])), dtype=np.float32)  # layers[i-1]*layers[i]
                all_weights['bias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.layers[i])), dtype=np.float32)  # 1 * layer[i]
            all_weights['rt_layer_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot_rt, size=(self.layers[num_layer - 1], self.layers_rt[0])), dtype=np.float32)
            all_weights['tp_layer_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot_tp, size=(self.layers[num_layer - 1], self.layers_tp[0])), dtype=np.float32)
            all_weights['rt_bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot_rt, size=(1, self.layers_rt[0])), dtype=np.float32)  # 1 * layers[0]
            all_weights['tp_bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot_tp, size=(1, self.layers_tp[0])), dtype=np.float32)  # 1 * layers[0]
            
            for i in range(1, num_layer1):
                glorot = np.sqrt(2.0 / (self.layers_rt[i - 1] + self.layers_rt[i]))
                # 64 layer  per layer 64  size
                all_weights['rt_layer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.layers_rt[i - 1], self.layers_rt[i])), dtype=np.float32)  # layers[i-1]*layers[i]
                all_weights['rt_bias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.layers_rt[i])), dtype=np.float32)  # 1 * layer[i]
            for i in range(1, num_layer2):
                glorot = np.sqrt(2.0 / (self.layers_tp[i - 1] + self.layers_tp[i]))
                # 64 layer  per layer 64  size
                all_weights['tp_layer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.layers_tp[i - 1], self.layers_tp[i])), dtype=np.float32)  # layers[i-1]*layers[i]
                all_weights['tp_bias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.layers_tp[i])), dtype=np.float32)  # 1 * layer[i]
	        # prediction layer
            glorot_rt = np.sqrt(2.0 / (self.layers_rt[-1] + 1))
            all_weights['rt_prediction'] = tf.Variable(np.random.normal(loc=0, scale=glorot_rt, size=(self.layers_rt[-1], 1)), dtype=np.float32)  # layers[-1] * 1
            glorot_tp = np.sqrt(2.0 / (self.layers_tp[-1] + 1))
            all_weights['tp_prediction'] = tf.Variable(np.random.normal(loc=0, scale=glorot_tp, size=(self.layers_tp[-1], 1)), dtype=np.float32)  # layers[-1] * 1
        else:
            all_weights['prediction'] = tf.Variable(np.ones((self.hidden_factor, 1), dtype=np.float32))  # hidden_factor * 1
        
        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
            is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
            is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.train_user_features: data['X_U'],
                     self.train_service_features: data['X_I'],
                     self.train_labels1: data['Y1'],
                      self.train_labels2: data['Y2'],
                      self.dropout_keep: self.keep_prob,
                      self.train_phase: True}
        joint_loss, opt = self.sess.run((self.joint_loss, self.optimizer), feed_dict=feed_dict)
        return joint_loss

    def get_random_block_from_data(self, data, batch_size):  # generate a random block of training data
        start_index = np.random.randint(0, len(data['Y1']) - batch_size)
        X_U, X_I, Y1 , Y2 = [], [], [], []
        # forward get sample
        i = start_index
        while len(X_U) < batch_size and i < len(data['X_U']):
            # if len(data['X_U'][i]) == len(data['X_U'][start_index]) | len(data['X_I'][i]) == len(data['X_I'][start_index]):
                Y1.append([data['Y1'][i]])
                Y2.append([data['Y2'][i]])
                X_U.append(data['X_U'][i])
                X_I.append(data['X_I'][i])
                i = i + 1
        # backward get sample
        i = start_index
        while len(X_U) < batch_size and i >= 0:
            # if len(data['X_U'][i]) == len(data['X_U'][start_index]) | len(data['X_I'][i]) == len(data['X_I'][start_index]):
                Y1.append([data['Y1'][i]])
                Y2.append([data['Y2'][i]])
                X_U.append(data['X_U'][i])
                X_I.append(data['X_I'][i])
                i = i - 1
        return {'X_U': X_U, 'X_I': X_I, 'Y1': Y1, 'Y2':Y2}

    def shuffle_in_unison_scary(self, a, b, c, d):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)

    def get_block_from_test_data(self, data, batch_size, i):  # generate a random block of training data
        start_index = i * batch_size
        X_U, X_I, Y1 , Y2 = [], [], [], []
        # forward get sample
        i = start_index
        while len(X_U) < batch_size and i < len(data['X_U']):
                Y1.append([data['Y1'][i]])
                Y2.append([data['Y2'][i]])
                X_U.append(data['X_U'][i])
                X_I.append(data['X_I'][i])
                i = i + 1
        return {'X_U': X_U, 'X_I': X_I, 'Y1': Y1, 'Y2':Y2}

    def part_evaluate(self, data):  # evaluate the results for an input set
        feed_dict = {self.train_user_features: data['X_U'],
                     self.train_service_features: data['X_I'],
                     self.train_labels1: data['Y1'],
                     self.train_labels2: data['Y2'],
                     self.dropout_keep: self.no_dropout,
                    self.train_phase: False}             
        predictions1, predictions2 = self.sess.run((self.Out_RT, self.Out_TP), feed_dict=feed_dict)
        predictions1 = predictions1.tolist()
        predictions2 = predictions2.tolist()
        return predictions1, predictions2
        
    def evaluate_test(self, data, predictions1, predictions2, num):  # evaluate the results for an input set
        num_example1 = num
        num_example2 = num
        y_pred1 = np.reshape(predictions1, (num_example1,))
        y_pred2 = np.reshape(predictions2, (num_example2,))
        y_true1 = np.reshape(data['Y1'][:num], (num_example1,))
        y_true2 = np.reshape(data['Y2'][:num], (num_example2,))
        
        predictions_bounded1 = np.maximum(y_pred1, np.ones(num_example1) * min(y_true1))  # bound the lower values
        predictions_bounded1 = np.minimum(predictions_bounded1, np.ones(num_example1) * max(y_true1))  # bound the higher values
        RMSE1 = math.sqrt(mean_squared_error(y_true1, predictions_bounded1))
        MAE1 = mean_absolute_error(y_true1, predictions_bounded1)
        NMAE1 = MAE1 / (max(y_true1) - min(y_true1))
        predictions_bounded2 = np.maximum(y_pred2, np.ones(num_example2) * min(y_true2))  # bound the lower values
        predictions_bounded2 = np.minimum(predictions_bounded2, np.ones(num_example2) * max(y_true2))  # bound the higher values
        RMSE2 = math.sqrt(mean_squared_error(y_true2, predictions_bounded2))
        MAE2 = mean_absolute_error(y_true2, predictions_bounded2)
        NMAE2 = MAE2 / (max(y_true2) - min(y_true2))
        RMSE = [RMSE1, RMSE2]
        MAE = [MAE1, MAE2]
        NMAE = [NMAE1, NMAE2]
        return RMSE, MAE, NMAE
            
    def train(self, Train_data, Test_data):  # fit a dataset
        # Check Init performance
        if self.verbose > 0:
            t2 = time()
            init_train_rm, init_train_ma = self.evaluate_train(Train_data)
            init_train_ma_all = init_train_ma[0] + init_train_ma[1]
            # init_test_rm, init_test_ma = self.evaluate(Test_data)
            if len(Test_data['Y1']) % self.batch_size == 0:
                total_batch2 = int(len(Test_data['Y1']) / self.batch_size)
            else:
                total_batch2 = int(len(Test_data['Y1']) / self.batch_size) + 1
            inPredictions1 = []
            inPredictions2 = []
            for i in range(total_batch2):
            # generate a batch
                batch_xs = self.get_block_from_test_data(Test_data, self.batch_size, i)
            # evaluate
                predictions1, predictions2 = self.part_evaluate(batch_xs) 
                inPredictions1.extend(predictions1)
                inPredictions2.extend(predictions2)
            
            inPredictions1 = np.array(inPredictions1)
            inPredictions2 = np.array(inPredictions2)
            init_test_rm, init_test_ma , init_test_nma = self.evaluate_test(Test_data, inPredictions1, inPredictions2, inPredictions1.shape[0])
            init_test_ma_all = self.alpha * init_test_ma[0] + (1 - self.alpha) * init_test_ma[1]
            # init_test_rm, init_test_ma = self.evaluate(Test_data)
            print("Init:tr_rmse=[%.4f ,%.4f],tr_mae=[%.4f, %.4f] ,te_rmse=[%.4f ,%.4f],\
                        te_mae=[%.4f ,%.4f] ,te_nmae=[%.5f ,%.5f],te_mae_all=[%.3f] [%.1f s]" 
                        % (init_train_rm[0], init_train_rm[1], init_train_ma[0], init_train_ma[1], \
                           init_test_rm[0] , init_test_rm[1], init_test_ma[0], init_test_ma[1], init_test_nma[0], init_test_nma[1], init_test_ma_all, time() - t2))
            
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Train_data['X_U'], Train_data['X_I'], Train_data['Y1'], Train_data['Y2'])
            total_batch = int(len(Train_data['Y1']) / self.batch_size)
            for i in range(total_batch):
                # generate a batch
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                # Fit training
                self.partial_fit(batch_xs)
            t2 = time()
            if len(Test_data['Y1']) % self.batch_size == 0:
                total_batch1 = int(len(Test_data['Y1']) / self.batch_size)
            else:
                total_batch1 = int(len(Test_data['Y1']) / self.batch_size) + 1
            Predictions1 = []
            Predictions2 = []
            for i in range(total_batch1):
            # generate a batch
                batch_xs = self.get_block_from_test_data(Test_data, self.batch_size, i)
            # evaluate
                predictions1, predictions2 = self.part_evaluate(batch_xs)
                Predictions1.extend(predictions1)
                Predictions2.extend(predictions2)
            Predictions1 = np.array(Predictions1)
            Predictions2 = np.array(Predictions2)
            # output validation
            train_result_rmse, train_result_mae = self.evaluate_train(Train_data)
            train_result_ma_all = train_result_mae[0] + train_result_mae[1]
            test_result_rmse, test_result_mae, test_result_nmae = self.evaluate_test(Test_data, Predictions1, Predictions2, Predictions1.shape[0])
            test_result_mae_all = self.alpha * test_result_mae[0] + (1 - self.alpha) * test_result_mae[1]

            self.train_rmse.append(train_result_rmse)
            self.train_mae.append(train_result_mae)
            self.test_rmse_rt.append(test_result_rmse[0])
            self.test_rmse_tp.append(test_result_rmse[1])
            self.test_mae.append(test_result_mae)
            self.test_mae_all.append(test_result_mae_all)
            if self.verbose > 0 and epoch % self.verbose == 0:
                print("Epoch %d [%.1f s] tr_rmse=[%.4f ,%.4f],tr_mae=[%.4f ,%.4f] , \
                        te_rmse=[%.4f ,%.4f],te_mae=[%.4f ,%.4f],te_mae_all=[%.3f]  [%.1f s]"
                      % (epoch + 1, t2 - t1, train_result_rmse[0], train_result_rmse[1], train_result_mae[0], train_result_mae[1], 
                         test_result_rmse[0], test_result_rmse[1], test_result_mae[0], test_result_mae[1], test_result_mae_all, time() - t2))

    def evaluate_train(self, data):  # evaluate the results for an input set
        num_example1 = len(data['Y1'])
        num_example2 = len(data['Y2'])
        feed_dict = {self.train_user_features: data['X_U'], 
                     self.train_service_features: data['X_I'], 
                     self.train_labels1: [[x_ext] for x_ext in data['Y1']], 
                     self.train_labels2: [[x_ext] for x_ext in data['Y2']], 
                     self.dropout_keep: self.no_dropout, self.train_phase: False}              
        
        predictions_rt, predictions_tp = self.sess.run((self.Out_RT, self.Out_TP), feed_dict=feed_dict)
        y_pred1 = np.reshape(predictions_rt, (num_example1,))
        y_pred2 = np.reshape(predictions_tp, (num_example1,))
        y_true1 = np.reshape(data['Y1'], (num_example1,))
        y_true2 = np.reshape(data['Y2'], (num_example2,))
         
        predictions_bounded = np.maximum(y_pred1, np.ones(num_example1) * min(y_true1))  # bound the lower values
        predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example1) * max(y_true1))  # bound the higher values
        RMSE1 = math.sqrt(mean_squared_error(y_true1, predictions_bounded))
        MAE1 = mean_absolute_error(y_true1, predictions_bounded)
        
        predictions_bounded2 = np.maximum(y_pred2, np.ones(num_example2) * min(y_true2))  # bound the lower values
        predictions_bounded2 = np.minimum(predictions_bounded2, np.ones(num_example2) * max(y_true2))  # bound the higher values
        RMSE2 = math.sqrt(mean_squared_error(y_true2, predictions_bounded2))
        MAE2 = mean_absolute_error(y_true2, predictions_bounded2)

        return  [RMSE1, RMSE2], [MAE1, MAE2]


if __name__ == '__main__':
    # Data loading
    args = parse_args()
    wsdl_support = args.wsdl_support
    if wsdl_support == 1:
        data = DATA.LoadData(args.path, args.dataset, '_wsdl/', args.hidden_factor)
    elif wsdl_support == 0:
        data = DATA.LoadData(args.path, args.dataset)
        
    if args.verbose > 0:
        print("DNM: dataset=%s, hidden_factor=%d, dropout_keep=%s, layers=%s, layers_rt=%s, layers_tp=%s, \
                loss_type=%s,  #epoch=%d, batch=%d, lr=%.4f, lambda=%.4f, optimizer=%s, batch_norm=%d, activation=%s" 
              % (args.dataset, args.hidden_factor, args.keep_prob, args.layers, args.layers_rt, args.layers_tp, args.loss_type, 
                 args.epoch, args.batch_size, args.lr, args.lamda, args.optimizer, args.batch_norm, args.activation))
    activation_function = tf.nn.relu
    # Training
    t1 = time()
    if wsdl_support == 1:
        x = np.random.normal(0.0, 0.01, [data.features_M_service - 4107, args.hidden_factor])
        x_ext = data.features_M_wsdl
        x_ext = np.array(x_ext)
        # print(x.shape, x_ext.shape)
        # print(x_ext)
        features_M_service = np.concatenate((x_ext, x), axis=0)
    elif wsdl_support == 0:
        features_M_service = np.random.normal(0.0, 0.01, [data.features_M_service, args.hidden_factor])
        
    model = DNM(data.features_M_user, features_M_service, args.hidden_factor, eval(args.layers), eval(args.layers_rt), eval(args.layers_tp), args.loss_type,
                args.epoch, args.batch_size, args.lr, args.lamda, eval(args.keep_prob), args.optimizer, args.batch_norm, activation_function, args.verbose, args.alpha)
    model.train(data.Train_data, data.Test_data)
    
    # Find the best validation result across iterations 
    best_valid_score = 0
    if args.loss_type == 'L1_loss':
        best_test_score = min(model.test_mae_all)
    best_epoch = model.test_mae_all.index(best_test_score)
    print(best_epoch + 1)
