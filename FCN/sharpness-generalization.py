import os
import tensorflow as tf
import numpy as np
import sys
import copy
import scipy as sp
import matplotlib.pyplot as plt

def binary_string(order):
    '''
    Produce all 2^order possible binary strings
    '''
    if order < 1:
        raise(Exception, 'Invalid Order!')
    if order == 1:
        return [[0.],[1.]]
    else:
        cp1 = copy.deepcopy(binary_string(order-1))
        cp2 = copy.deepcopy(binary_string(order-1))
        for item in cp1:
            item.append(0.)
        for item in cp2:
            item.append(1.)
        return cp1 + cp2

def list2str(y_train):
    '''
    Convert a 128 bit function y_train (list) to string form
    '''
    s = ''
    for i in y_train:
        s = s + str(int(i))
    return s


def get_y_train_from_string(string, symmetry=False):
    '''
    convert the 128-bits function string to a np.array
    '''
    l = len(string)
    y = np.zeros([l,1])
    for i in range(l):
        if symmetry == False:
            y[i][0] = float(string[i])
        elif symmetry == True:
            if string[i] == '0':
                y[i][0] = -1.
            if string[i] == '1':
                y[i][0] = 1.
    return y


def read_string_from_combine(file):
    '''
    read each 128-bits string function from the concatenated
    random sampling file
    '''
    with open(file, 'r') as f:
        return(f.readlines()[1].split()[1])

def flat_a_list(params):
    for i in range(len(params)):
        params[i] = params[i].flatten()
    return(np.concatenate(params))

def clip_params(eps, params, new_params):
    for i in range(len(new_params)):
        diff = new_params[i] - params[i]
        eps_mtx = eps * (np.abs(params[i]) + 1)
        outer_up = np.where(diff>eps_mtx)
        diff[outer_up] = eps_mtx[outer_up]
        outer_low = np.where(diff<-eps_mtx)
        diff[outer_low] = -eps_mtx[outer_low]
        new_params[i] = params[i] + diff
    return new_params

# Model function
def model_fun(X, params):
    l_2 = tf.nn.relu(tf.add(tf.matmul(X, params[0]), params[1]))
    l_3 = tf.nn.relu(tf.add(tf.matmul(l_2, params[2]), params[3]))
    l_4 = tf.add(tf.matmul(l_3, params[4]), params[5]) # No Sigmoid here because Cost Function includes sigmoid
    return l_4

def label_corruption(y_train, k):
    '''
    k is the ratio of label corruption
    '''
    l = len(y_train)
    y_train_corrupted = copy.deepcopy(y_train)
    corruption = int(np.floor(l*k))
    random_idx = np.random.choice(l, corruption , replace=False)
    for i in random_idx:
        if y_train[i][0] == 1.:
            y_train_corrupted[i][0] = 0
        elif y_train[i][0] == 0.:
            y_train_corrupted[i][0] = 1
    return y_train_corrupted

def flipping_label(y_attack):
    l = len(y_attack)
    for i in range(l):
        if y_attack[i][0] == 1.:
            y_attack[i][0] = 0.
        elif y_attack[i][0] == 0.:
            y_attack[i][0] = 1.
    return y_attack

def flatten(params):
    """
    Flattens the list of tensor(s) into a 1D tensor

    Args:
        params: List of model parameters (List of tensor(s))

    Returns:
        A flattened 1D tensor
    """
    return tf.concat([tf.reshape(_params, [-1]) \
                      for _params in params], axis=0)

def binary_crossentropy(yhat, y):
    return tf.reduce_mean(-y*tf.log(yhat) - (1-y)*tf.log(1-yhat))

def binary_cross_entropy(output, target, epsilon=1e-8, name='bce_loss'):
    return tf.reduce_mean(
            tf.reduce_sum(
                    -(target * tf.math.log(output + epsilon)
                    + (1. - target) * tf.math.log(1. - output + epsilon)),
                    axis=1),
            name=name)

def mse(y, yhat):
    return tf.reduce_mean(tf.squared_difference(yhat, y))

def turn_onehot_onto_binary(y_train):
    temp = np.zeros([y_train.shape[0], 1])
    for i in range(y_train.shape[0]):
        if np.where(y_train[i]==1)[0][0] >= 5:
            temp[i] = 1
    return temp

def shuffle(x_train, y_train):
    temp = list(zip(x_train, y_train))
    np.random.shuffle(temp)
    x_train, y_train = zip(*temp)
    return None

def cal_hessian(x_train, y_train,x_test,y_test,train_size,x_train_genuine,y_train_genuine,sample_times=5):
    # Model architecture; number of neurons, layer-wise.
    # e.g. feed-forward neural network

    T1, T2, T3, T4 = 784, 40, 40, 1

    sharpness_sample = []
    generalization_sample = []
    sample = 0
    sample_succeed = 0
    while True:
        tf.reset_default_graph()

        W1 = tf.Variable(tf.random.normal((T1, T2), stddev=np.sqrt(2/T1), dtype='float32'))
        W2 = tf.Variable(tf.random.normal((T2, T3), stddev=np.sqrt(2/T2), dtype='float32'))
        W3 = tf.Variable(tf.random.normal((T3, T4), stddev=np.sqrt(2/T3), dtype='float32'))

        b2 = tf.Variable(tf.random.normal((T2, ), stddev=np.sqrt(1), dtype='float32'))
        b3 = tf.Variable(tf.random.normal((T3, ), stddev=np.sqrt(1), dtype='float32'))
        b4 = tf.Variable(tf.random.normal((T4, ), stddev=np.sqrt(1), dtype='float32'))

        # Stack weights and biases layer-wise
        params = [W1, b2, W2, b3, W3, b4]

        params_size = \
        np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

        # Input-output variables
        X = tf.placeholder(dtype='float32', shape=(None, T1))
        y = tf.placeholder(dtype='float32', shape=(None, T4))

        yhat_logits = model_fun(X, params)
        model_output = tf.nn.sigmoid(yhat_logits)

        # Cost function for Pyhessian Input
        def cost_fun(y, yhat_logits, params):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=yhat_logits, labels=y, name=None))

        cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=yhat_logits,
                                                   labels=y, name=None))
        # Learning Rate Setting
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.01

        learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate,
                                                             global_step,
                                           100000, 1, staircase=True)
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,
                                                        global_step=global_step)

        #train_op = tf.train.AdamOptimizer().minimize(cost)

        #train_op = tf.compat.v1.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(cost)

        #train_op = tf.compat.v1.train.AdagradOptimizer(0.01).minimize(cost)


        # For Sharpness Calculation
        new_cost = - tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=yhat_logits,
                                                   labels=y, name=None))

        sharpness_train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(new_cost)

        preds = tf.cast((yhat_logits >= 0.), tf.float32)
        y_label_0_1 = tf.cast((y > 1.e-4), tf.float32)
        correct_prediction = tf.equal(preds, y_label_0_1)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        W1_update_placeholder = tf.placeholder(W1.dtype, shape=W1.get_shape())
        b2_update_placeholder = tf.placeholder(b2.dtype, shape=b2.get_shape())
        W2_update_placeholder = tf.placeholder(W2.dtype, shape=W2.get_shape())
        b3_update_placeholder = tf.placeholder(b3.dtype, shape=b3.get_shape())
        W3_update_placeholder = tf.placeholder(W3.dtype, shape=W3.get_shape())
        b4_update_placeholder = tf.placeholder(b4.dtype, shape=b4.get_shape())
        W1_update_op = W1.assign(W1_update_placeholder)
        b2_update_op = b2.assign(b2_update_placeholder)
        W2_update_op = W2.assign(W2_update_placeholder)
        b3_update_op = b3.assign(b3_update_placeholder)
        W3_update_op = W3.assign(W3_update_placeholder)
        b4_update_op = b4.assign(b4_update_placeholder)

        batch_size = 32
        dataset_size = x_train.shape[0]

        '''Start Training!!!'''
        with tf.Session() as sess:
            init = tf.initialize_all_variables()
            sess.run(init)

            ''' A block meant to make sure 0 training error'''
            total_epochs = 50000
            for epochs_needed_to_reach_0_training_error in range(total_epochs):
                shuffle(x_train, y_train)
                for i in range(dataset_size // batch_size + 1):
                    # build batches
                    start = (i * batch_size) % dataset_size
                    end = min(batch_size + start, dataset_size)
                    sess.run(train_op,
                             feed_dict={X: x_train[start:end], y: y_train[start:end]})
                # Measure train accuracy only on genuine training data
                train_accuracy = accuracy.eval(feed_dict={X: x_train_genuine, y: y_train_genuine})
                total_loss = sess.run(cost, feed_dict={X:x_train_genuine,y:y_train_genuine})
                if ((epochs_needed_to_reach_0_training_error+1)%50 == 0):
                    print('%d epoch(s),training loss on all data is %f, training accuracy is %f'
                        %(epochs_needed_to_reach_0_training_error+1,total_loss,train_accuracy))

                if train_accuracy == 1.:
                    print('epochs needed to reach 0 training_error: %d'
                          % (epochs_needed_to_reach_0_training_error+1))

                    break
            if epochs_needed_to_reach_0_training_error < total_epochs-1:
                sample_succeed += 1
                test_accuracy = accuracy.eval(feed_dict={X: x_test, y: y_test})
                print('Accuracy on test set is %f'%test_accuracy)
                generalization_sample.append(test_accuracy)

                # Save xs and ys for volume calculation
                ys = np.concatenate((y_train_genuine,preds.eval(feed_dict={X: x_test})), axis = 0)
                np.save('ys_no_attack_%d'%(sample_succeed), ys)

                # get w here
                w = sess.run(params)
                L_w = sess.run(cost, feed_dict={X:x_train_genuine,y:y_train_genuine})

                # Sharpness
                dataset_size = train_size
                max_iter_epochs = 100
                max_value = 0
                max_value_list = []
                for sharpness_epoch in range(max_iter_epochs):
                    for i in range(dataset_size // batch_size + 1):
                        start = (i * batch_size) % dataset_size
                        end = min(batch_size + start, dataset_size)
                        sess.run( sharpness_train_op,
                                 feed_dict={X: x_train_genuine[start:end], \
                                         y: y_train_genuine[start:end]} )
                        new_w = sess.run(params)
                        new_w = clip_params(1e-4, w, new_w)

                        sess.run(W1_update_op,{W1_update_placeholder:new_w[0]})
                        sess.run(b2_update_op,{b2_update_placeholder:new_w[1]})
                        sess.run(W2_update_op,{W2_update_placeholder:new_w[2]})
                        sess.run(b3_update_op,{b3_update_placeholder:new_w[3]})
                        sess.run(W3_update_op,{W3_update_placeholder:new_w[4]})
                        sess.run(b4_update_op,{b4_update_placeholder:new_w[5]})

                        max_value = max(max_value, -sess.run(new_cost,
                                    feed_dict={X:x_train_genuine,y:y_train_genuine}))
                        max_value_list.append(max_value)
                np.save('max_value_list_%d.npy'%sample_succeed, max_value_list)
                sharpness = 100 * (max_value - L_w) / (1 + L_w)
                sharpness_sample.append(np.log10(sharpness))

                print('Sharpness (log) is: %f'%
                      (np.log10(sharpness)))
            else:
                print('%d epochs and still can not reach 0 training error...'
                      %total_epochs)

        tf.get_default_graph().finalize()
        sample += 1
        print('sample %d, %d/%d succeed' % (sample,sample_succeed,sample_times))
        if sample_succeed == sample_times: break

    return(sharpness_sample,generalization_sample)

if __name__ == '__main__':

    DATAPATH = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))),
            'data')
    if not os.path.exists(DATAPATH):
        DATAPATH = os.path.join(os.path.dirname(
                os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))),
                'data')

    x_train_20000 = np.load(os.path.join(DATAPATH,'train_x_20000.npy'))
    y_train_20000 = turn_onehot_onto_binary(np.load(os.path.join(DATAPATH,'train_y_20000.npy')))
    x_test = np.load(os.path.join(DATAPATH,'test_x_1000.npy'))
    y_test = turn_onehot_onto_binary(np.load(os.path.join(DATAPATH,'test_y_1000.npy')))

    attack_size = int(sys.argv[1])
    print('Attack set size: %d' % attack_size)
    train_size = 500

    x_train = x_train_20000[:train_size+attack_size]
    y_train = np.concatenate((y_train_20000[:train_size],
            flipping_label(y_train_20000[train_size:train_size+attack_size])))

    x_train_genuine = x_train[:train_size]
    y_train_genuine = y_train[:train_size]

    (sharpness_sample,generalization_sample) = cal_hessian(
            x_train,y_train,x_test,y_test,train_size,x_train_genuine,y_train_genuine,5)

    np.save('sharpness_sample.npy',sharpness_sample)
    np.save('generalization_sample.npy',generalization_sample)





