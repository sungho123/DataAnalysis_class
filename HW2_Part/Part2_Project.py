# coding: utf-8

# In[ ]:


# required library
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Load the data and transform each image into 28x28 matrix

width = 28
height = 28
mnist = input_data.read_data_sets("MNIST_CNN/")
train_x = np.array([np.reshape(x, (width, height)) for x in mnist.train.images])
train_y = mnist.train.labels
test_x = np.array([np.reshape(x, (width, height)) for x in mnist.test.images])
test_y = mnist.test.labels

## Hyperparameters
batch_size = 90             # 실제 학습 시킬 데이터 sampling 
learning_rate = 0.005      # *20 한번에 가능한 이동,학습률 
evaluation_size = 500       # test accuracy를 위한 sampling
iter_count = 800           # Generation : 학습 반복 수
target_size = max(train_y) + 1
num_channels = 1            # 색 -> only gray
conv1_num_features = 50
conv2_num_features = 75     # conv filter의 수 
#conv3_num_features = 60
max_pool_size1 = 2   
max_pool_size2 = 2          # pooling filter의 크기
#max_pool_size3 = 2
hidden_layer_nodes = 40   # fully connected layer에 쓰인 노드의 수
conv1_size = 4
conv2_size = 4              # conv filter의 크기
# activation function : ReLU 


# Declare the placeholder for the data (i.e., images) and target (i.e., 10-length vector)
x_data = tf.placeholder(shape=(batch_size, width, height, num_channels), dtype=tf.float32)
print(x_data)
y_target = tf.placeholder(shape=(batch_size), dtype=tf.int32)
eval_data = tf.placeholder(shape=(evaluation_size, width, height, num_channels), dtype=tf.float32)
eval_target = tf.placeholder(shape=(evaluation_size), dtype=tf.int32)

# Declare convolution weights and biases
conv1_W = tf.Variable(tf.truncated_normal([conv1_size, conv1_size, num_channels, conv1_num_features], stddev=0.1, dtype=tf.float32))
conv1_b = tf.Variable(tf.zeros([conv1_num_features], dtype=tf.float32))
conv2_W = tf.Variable(tf.truncated_normal([conv2_size, conv2_size, conv1_num_features, conv2_num_features], stddev=0.1, dtype=tf.float32))
conv2_b = tf.Variable(tf.zeros([conv2_num_features], dtype=tf.float32))

# Declare fully connected weights and biases for the last two layers
resulting_width = width // (max_pool_size1 * max_pool_size2)
resulting_height = height // (max_pool_size1 * max_pool_size2)
full_size = resulting_width * resulting_height * conv2_num_features
full1_W = tf.Variable(tf.truncated_normal([full_size, hidden_layer_nodes], stddev=0.1, dtype=tf.float32))
full1_b = tf.Variable(tf.truncated_normal([hidden_layer_nodes], stddev=0.1, dtype=tf.float32))
full2_W = tf.Variable(tf.truncated_normal([hidden_layer_nodes, target_size], stddev=0.1, dtype=tf.float32))
full2_b = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))

# Declare the model
def conv_net(input_data):
    # First Conv-ReLU-MaxPool Layer
    conv1 = tf.nn.conv2d(input_data, conv1_W, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(conv1 + conv1_b)
    max_pool1 = tf.nn.max_pool(relu1, ksize=[1, max_pool_size1, max_pool_size1, 1],
                               strides=[1, max_pool_size1, max_pool_size1, 1], padding='SAME')

    # Second Conv-ReLU-MaxPool Layer
    conv2 = tf.nn.conv2d(max_pool1, conv2_W, strides=[1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(conv2 + conv2_b)
    max_pool2 = tf.nn.max_pool(relu2, ksize=[1, max_pool_size2, max_pool_size2, 1],
                               strides=[1, max_pool_size2, max_pool_size2, 1], padding='SAME')

    # Transform output into a 1 X N layer for the next fully connected layer
    final_shape_list = max_pool2.get_shape().as_list()
    final_shape = final_shape_list[1] * final_shape_list[2] * final_shape_list[3]
    flat_output = tf.reshape(max_pool2, [final_shape_list[0], final_shape])

    # Fully-connected layer
    hidden_output = tf.nn.relu(tf.add(tf.matmul(flat_output, full1_W), full1_b))
    final_output = tf.add(tf.matmul(hidden_output, full2_W), full2_b)
    return final_output

# Declare the model on the training data and the test data
final_output = conv_net(x_data)
test_output = conv_net(eval_data)

# Loss function  + test Loss function 을 구하기 위한 loss2 함수 추가
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_target, logits=final_output))
loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=eval_target, logits=test_output))

# Declare optimizing algorithm
opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate*20)
train_step = opt.minimize(loss)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# Training iterations
train_accuracies = []
test_accuracies = []
train_loss = []
test_loss = []

train_prediction = tf.nn.softmax(final_output)
test_prediction = tf.nn.softmax(test_output)
def get_accuracy(logits, targets):
    batch_predictions = np.argmax(logits, axis=1)
    num_correct = np.sum(np.equal(batch_predictions, targets))
    return 100. * num_correct/batch_predictions.shape[0]

for i in range(iter_count):
    # take a random sample from MNIST train set
    rand_index = np.random.choice(len(train_x), size=batch_size)
    rand_x = np.expand_dims(train_x[rand_index], 3)
    rand_y = train_y[rand_index]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    train_output = sess.run(train_prediction, feed_dict={x_data: rand_x, y_target: rand_y})
    train_accuracy = get_accuracy(train_output, rand_y)
    train_accuracies.append(train_accuracy)

    eval_rand_index = np.random.choice(len(test_x), size=evaluation_size)
    eval_x = np.expand_dims(test_x[eval_rand_index], 3)
    eval_y = test_y[eval_rand_index]

    test_output = sess.run(test_prediction, feed_dict={eval_data: eval_x, eval_target: eval_y})
    test_accuracy = get_accuracy(test_output, eval_y)
    test_accuracies.append(test_accuracy)
    
    # train , test data에 대한 cost 
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    train_loss.append(np.sqrt(temp_loss))

    test_temp_loss = sess.run(loss2, feed_dict={eval_data: eval_x, eval_target: eval_y})
    test_loss.append(np.sqrt(test_temp_loss))
    
    if (i+1) % 10 == 0:
        print("Generation: %d. Train accuracy = %f. Test accuracy = %f, Cost = %f " %(i+1, train_accuracy,
                                                                                      test_accuracy, temp_loss))
print("Max Test accuracy = %f" %max(test_accuracies))

plt.plot(train_accuracies, 'k-', label='Train Accuracy')
plt.plot(test_accuracies, 'r--', label='Test Accuracy')
plt.title('Match Ratio per Generation')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.show()


# In[ ]:


plt.plot(train_loss, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()


# In[ ]:
