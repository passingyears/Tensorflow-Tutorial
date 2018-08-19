"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

tf.set_random_seed(1)
np.random.seed(1)

# fake data
n_data = np.ones((1000, 2))
x0 = np.random.normal(2*n_data, 1)      # class0 x shape=(1000, 2)
y0 = np.zeros(1000)                      # class0 y shape=(1000, 1)
x1 = np.random.normal(-2*n_data, 1)     # class1 x shape=(1000, 2)
y1 = np.ones(1000)                       # class1 y shape=(1000, 1)
x = np.vstack((x0, x1))  # shape (200, 2) + some noise
y = np.hstack((y0, y1))  # shape (200, )

# plot data
plt.scatter(x[:, 0], x[:, 1], c=y, s=100, lw=0, cmap='RdYlGn')
plt.show()

tf_x = tf.placeholder(tf.float32, x.shape)     # input x
tf_y = tf.placeholder(tf.int32, y.shape)     # input y

# neural network layers
l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)          # hidden layer
output = tf.layers.dense(l1, 2)                     # output layer

loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=output)           # compute cost
accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.squeeze(tf_y), predictions=tf.argmax(output, axis=1),)[1]
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_op = optimizer.minimize(loss)
time.sleep(3)
with tf.Session() as sess:
	init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
	sess.run(init_op)
	plt.ion()
	fig, ax = plt.subplots(figsize=(20,16))
	for step in range(100):
		_, acc, pred = sess.run([train_op, accuracy, output],{tf_x:x,tf_y:y})
		if step%2 == 0:
			ax.cla()
			ax.scatter(x[:,0],x[:,1],c=pred.argmax(1),s=100,lw=0,cmap='RdYlGn')
			ax.text(0,-5.5,'Accuracy=%.2f'%acc,fontdict={'size':60,'color':'red'})
			fig.canvas.flush_events()
			time.sleep(0.1)
	plt.ioff()
	plt.show()
