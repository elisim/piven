"""
This file:
    - defines NN structure and loss
    - defines train and predict functions
    - visualize training
"""

import numpy as np
import tensorflow as tf

DEVICE = "/cpu:0"  # change to gpu if needed


class TfNetwork:
    def __init__(self, x_size, y_size, h_size,
                 alpha=0.05,
                 soften=160.,
                 lambda_in=15.,
                 sigma_in=0.1,
                 out_biases=[2., -2.],
                 method='piven',
                 **kwargs):
        """
        Create the DNN for UCI experiment, and define the loss

        @param x_size: input size
        @param y_size: output size
        @param h_size: list of neurons number in each hidden layer
        @param alpha: confidence level (1-alpha)
        @param soften: soften parameter for qd loss
        @param lambda_in: lambda parameter for qd loss
        @param sigma_in: stddev of initializing normal distribution
        @param out_biases: used for PIs initialization
        @param method: method name
        @param kwargs:
                - patience: patience for early stopping (default -1, i.e. no early stopping)
                - dataset: dataset name (optional)
        """

        self.method = method
        self.patience = kwargs.get('patience', -1)
        self.dataset = kwargs.get('dataset', "dataset placeholder")

        # set up input and output
        X = tf.placeholder(tf.float32, [None, x_size])
        y_true = tf.placeholder(tf.float32, [None, 1])  # force to one for PI

        # set up parameters
        W = []  # list of variables
        b = []  # biases
        layer_in = []  # before activation
        layer = []  # post-activation

        # first layer
        W.append(tf.Variable(tf.random_normal([x_size, h_size[0]], stddev=sigma_in)))  # W = [(input * h_size[0]=50)]
        b.append(tf.Variable(np.zeros(h_size[0]) + 0.1, dtype=tf.float32))

        # add hidden layers
        for i in range(1, len(h_size)):
            W.append(tf.Variable(tf.random_normal([h_size[i - 1], h_size[i]], stddev=sigma_in)))
            b.append(tf.Variable(np.zeros(h_size[i]) + 0.1, dtype=tf.float32))

        # add final layer
        W.append(tf.Variable(tf.random_normal([h_size[-1], y_size], stddev=sigma_in)))
        if self.method == 'deep-ens':
            b.append(tf.Variable([0.0, 1.0]))  # zero mean, 1.0 variance
        else:
            b.append(tf.Variable(out_biases))  # PI initialization

        # define model - first layer
        with tf.device(DEVICE):
            layer_in.append(tf.matmul(X, W[0]) + b[0])
            layer.append(tf.nn.relu(layer_in[-1]))

        # hidden layers
        for i in range(1, len(h_size)):
            with tf.device(DEVICE):
                layer_in.append(tf.matmul(layer[i - 1], W[i]) + b[i])
                layer.append(tf.nn.relu(layer_in[-1]))

        # create metric list
        metric = []
        metric_name = []

        # finish defining network
        with tf.device(DEVICE):
            layer_in.append(tf.matmul(layer[-1], W[-1]) + b[-1])  # W[-1].shape = [h_size[-1], y_size]

            v = tf.Variable(tf.random_normal([h_size[-1], 1], stddev=sigma_in))
            b_v = tf.Variable(tf.zeros(1) + 0.01)
            if self.method == 'piven':
                v_out = tf.nn.sigmoid(tf.matmul(layer[-1], v) + b_v)
            if self.method == 'only-rmse':
                v_out = tf.nn.relu(tf.matmul(layer[-1], v) + b_v)

        y_pred = layer_in[-1]  # since it's linear no need for fn

        if self.method == 'piven' or self.method == 'only-rmse':
            y_pred = tf.concat([y_pred, v_out], axis=1)  # y_pred.shape = (n_samples, 3)

        # get components
        y_U = y_pred[:, 0]
        y_L = y_pred[:, 1]
        y_T = y_true[:, 0]
        if self.method == 'piven' or self.method == 'only-rmse':
            y_v = y_pred[:, 2]

        N_ = tf.cast(tf.size(y_T), tf.float32)  # batch size
        alpha_ = tf.constant(alpha)
        lambda_ = tf.constant(lambda_in)

        with tf.device(DEVICE):
            # === Loss definition
            # soft uses sigmoid
            k_soft = tf.multiply(tf.sigmoid((y_U - y_T) * soften), tf.sigmoid((y_T - y_L) * soften))

            # hard uses sign step function
            k_hard = tf.multiply(tf.maximum(0., tf.sign(y_U - y_T)), tf.maximum(0., tf.sign(y_T - y_L)))

            MPIW_capt = tf.divide(tf.reduce_sum(tf.abs(y_U - y_L) * k_hard),
                                    tf.reduce_sum(k_hard) + 0.001)

            PICP_soft = tf.reduce_mean(k_soft)

            # in QD implementation, they actually used sqrt(batch_size)
            # https://github.com/TeaPearce/Deep_Learning_Prediction_Intervals/blob/master/code/DeepNetPI.py#L195
            qd_rhs_soft = lambda_ * tf.sqrt(N_) * tf.square(tf.maximum(0., (1. - alpha_) - PICP_soft))
            qd_loss = MPIW_capt + qd_rhs_soft  # final qd loss form

        # set main loss type
        loss = qd_loss  # qd method
        if self.method == 'piven':
            y_eli = y_v * y_U + (1 - y_v) * y_L
            y_eli = tf.reshape(y_eli, (-1, 1))
            # same as set beta=0.5
            loss += tf.losses.mean_squared_error(y_true, y_eli)
        elif self.method == 'only-rmse':
            y_eli = 0.5 * y_U + 0.5 * y_L  # eli
            y_eli = tf.reshape(y_eli, (-1, 1))
            loss += tf.losses.mean_squared_error(y_true, y_eli)
        elif self.method == 'deep-ens':
            # gaussian log likelihood
            # y_U = mean, y_L = variance
            # from deep ensemble paper
            y_mean = y_U
            y_var_limited = tf.minimum(y_L, 10.)  # need to limit otherwise causes nans occasionally
            y_var = tf.maximum(tf.log(1. + tf.exp(y_var_limited)), 10e-6)
            gauss_loss = tf.log(y_var) / 2. + tf.divide(tf.square(y_T - y_mean), 2. * y_var)
            loss = tf.reduce_mean(gauss_loss)

        # add metrics
        with tf.device(DEVICE):
            MPIW = tf.reduce_mean(tf.subtract(y_U, y_L))
            PICP = tf.reduce_mean(k_hard)
            metric.append(PICP)
            metric_name.append('PICP')
            metric.append(MPIW)
            metric_name.append('MPIW')

        # save for training
        self.X = X
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss = loss
        self.metric = metric
        self.metric_name = metric_name

    def train(self,
              sess,
              X_train,
              y_train,
              X_val,
              y_val,
              n_epoch,
              l_rate=0.01,
              n_batch=100,
              is_batch=True,
              decay_rate=0.95,
              is_early_stop=False,
              is_use_val=False,
              optim='adam',
              is_print_info=True):
        """
        Perform training process

        @param sess: tensorflow session object
        @param optim: optimizer name
        @param is_batch: training in batches, or fit whole data in one batch 
        @param n_batch: batch size 
        @param is_print_info: printing training process or not
        """
        global_step = tf.Variable(0, trainable=False)  # keep track of which epoch on
        decayed_l_rate = tf.train.exponential_decay(l_rate, global_step,
                                                    decay_steps=50, decay_rate=decay_rate, staircase=False)
        # eqn: decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        with tf.device(DEVICE):
            if optim == 'SGD':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=decayed_l_rate)
            elif optim == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=decayed_l_rate)
            else:
                raise ValueError(f'Invalid optimizer name: {optim}')

            train_step = optimizer.minimize(self.loss, global_step=global_step)

        sess.run(tf.global_variables_initializer())  # init variables
        loss_log = []

        # train
        val_loss_prev = 0
        patience = 0
        lr_zero_epochs = 0
        for epoch in range(n_epoch):
            if is_batch:
                # shuffle order
                perm = np.random.permutation(X_train.shape[0])
                X_train_shuff = X_train[perm]
                y_train_shuff = y_train[perm]

                loss_train = 0
                # for each batch
                n_batches = int(round(X_train.shape[0] / n_batch))
                for b in range(n_batches):

                    # if last batch use all data
                    if b == int(round(X_train.shape[0] / n_batch)):
                        X_train_b = X_train_shuff[b * n_batch:]
                        y_train_b = y_train_shuff[b * n_batch:]
                    else:
                        X_train_b = X_train_shuff[b * n_batch:(b + 1) * n_batch]
                        y_train_b = y_train_shuff[b * n_batch:(b + 1) * n_batch]
                    _, loss_train_b = sess.run([train_step, self.loss],
                                               feed_dict={self.X: X_train_b, self.y_true: y_train_b})
                    loss_train += loss_train_b / n_batches
            else:
                # whole dataset
                _, loss_train = sess.run([train_step, self.loss],
                                         feed_dict={self.X: X_train, self.y_true: y_train})

            # print info
            if epoch % int(n_epoch / 10) == 0 or epoch == n_epoch - 1:
                if is_use_val:
                    loss_val, l_rate_epoch = sess.run([self.loss, decayed_l_rate],
                                                      feed_dict={self.X: X_val, self.y_true: y_val})
                else:
                    loss_val = loss_train  # quicker for training
                    l_rate_epoch = sess.run(decayed_l_rate,
                                            feed_dict={self.X: X_val, self.y_true: y_val})

                if is_print_info:
                    print('\nep:', epoch, ' \ttrn loss', round(loss_train, 4), '  \tval loss', round(loss_val, 4),
                          end='\t')

                # the metrics don't really make sense for gauss likelihood
                if self.method != 'deep-ens':
                    ops_to_run = []
                    for i in range(0, len(self.metric)):
                        ops_to_run.append(self.metric[i])
                    _1, _2 = sess.run(ops_to_run,
                                      feed_dict={self.X: X_val, self.y_true: y_val})

                    if is_print_info:
                        print(self.metric_name[0], round(_1, 4), '\t',
                              self.metric_name[1], round(_2, 4),
                              end='\t')
                if is_print_info:
                    print('l_rate', round(l_rate_epoch, 5), end='\t')
                loss_log.append((epoch, loss_train, loss_val))

            # stop because lr is zero
            if patience != -1:
                l_rate_epoch = sess.run(decayed_l_rate,
                                        feed_dict={self.X: X_val, self.y_true: y_val})
                if l_rate_epoch < 1e-7:
                    lr_zero_epochs += 1
                if lr_zero_epochs > 20:
                    print(f"\n\t\t========== lr = 0!!! TRAINING STOP AT EPOCH {epoch} ==========\n")
                    break

            # check for stopping criteria for val loss
            if is_early_stop:
                val_loss = sess.run(self.loss, feed_dict={self.X: X_val, self.y_true: y_val})
                if val_loss > val_loss_prev:
                    patience += 1
                else:
                    patience = 0

                if patience > self.patience:
                    print(f"\n\t\t========== TRAINING STOP AT EPOCH {epoch} ==========\n")
                    break
                val_loss_prev = val_loss

        self.loss_log = np.array(loss_log)  # convert list to array
        self.last_loss_trn = self.loss_log[-1, 1]

    def predict(self, sess, X_test, y_test):
        """
        run prediction
        """
        y_pred_out = sess.run(self.y_pred, feed_dict={self.X: X_test, self.y_true: y_test})
        y_loss = sess.run(self.loss, feed_dict={self.X: X_test, self.y_true: y_test})
        return y_loss, y_pred_out
