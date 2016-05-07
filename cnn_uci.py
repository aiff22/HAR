import numpy as np
import math
from sklearn.metrics import classification_report
import tensorflow as tf

## CNN parameters

segment_size = 128
num_training_iterations = 20000
batch_size = 50
num_input_channels = 6

def weight_variable(shape, stddev):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)
  
def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv1d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_1x4(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding='SAME')

## Loading the dataset

print('Loading UCI dataset...')

# Reading training data

f = open("data_processing/uci_data/all_data.csv")
data_train = np.loadtxt(fname = f, delimiter = ',')
f.close();

# Reading test data

f = open("data_processing/uci_data/all_data_test.csv")
data_test = np.loadtxt(fname = f, delimiter = ',')
f.close();

# Reading training labels

fa = open("data_processing/uci_data/answers.csv")
labels_train = np.loadtxt(fname = fa, delimiter = ',')
fa.close()

# Reading test labels

fa = open("data_processing/uci_data/answers_test.csv")
labels_test = np.loadtxt(fname = fa, delimiter = ',')
fa.close()


train_size = data_train.shape[0];
test_size = data_test.shape[0];

data_test = np.reshape(data_test, [data_test.shape[0], segment_size * num_input_channels])
labels_test = np.reshape(labels_test, [labels_test.shape[0], 6])
labels_test_unary = np.argmax(labels_test, axis=1)

print("Dataset was uploaded\n")

## creating CNN

print("Creating CNN architecture\n")


# Convolutional and Pooling layers

W_conv1 = weight_variable([1, 12, num_input_channels, 196], stddev=0.001)
b_conv1 = bias_variable([196])

x = tf.placeholder(tf.float32, [None, segment_size * num_input_channels])
x_image = tf.reshape(x, [-1, 1, segment_size, num_input_channels])

h_conv1 = tf.nn.relu(conv1d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_1x4(h_conv1)


# Fully connected layer with Dropout

W_fc1 = weight_variable([int(math.ceil(segment_size/4))*196, 1024], stddev=0.01)
b_fc1 = bias_variable([1024])

h_pool1_flat = tf.reshape(h_pool1, [-1, int(math.ceil(segment_size/4))*196])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

keep_prob_1 = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_1)


# Softmax layer

W_softmax = weight_variable([1024, 6], stddev=0.01)
b_softmax = bias_variable([6])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_softmax) + b_softmax)
y_ = tf.placeholder(tf.float32, [None, 6])


# Cross entropy loss function and L2 regularization term

cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
cross_entropy += 5e-4 * (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1))


# Training step

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Run Tensorflow session

sess = tf.Session()
sess.run(tf.initialize_all_variables())

# Train CNN
print("Training CNN... ")

for i in range(num_training_iterations):
    
    idx_train = np.random.randint(0, train_size, batch_size)          
    
    xt = np.reshape(data_train[idx_train], [batch_size, segment_size * num_input_channels])
    yt = np.reshape(labels_train[idx_train], [batch_size, 6])
        
    sess.run(train_step, feed_dict={x: xt, y_: yt, keep_prob_1: 0.5}) #, keep_prob_2: 0.5})
                    
    if i == num_training_iterations - 1:

        train_accuracy, train_entropy, y_pred = sess.run([accuracy, cross_entropy, y_conv], 
                                                        feed_dict={ x : data_test, y_: labels_test, keep_prob_1: 1})
        
        print("step %d, entropy %g"%(i, train_entropy))
        print("step %d, accuracy %g"%(i, train_accuracy))
        print(classification_report(labels_test_unary, np.argmax(y_pred, axis=1), digits=4))
