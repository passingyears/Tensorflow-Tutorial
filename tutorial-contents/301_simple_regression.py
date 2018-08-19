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
x = np.linspace(-1, 1, 1000)[:, np.newaxis]          # shape (100, 1)
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise                          # shape (100, 1) + some noise

# plot data
plt.scatter(x, y)
plt.show()

tf_x = tf.placeholder(tf.float32, x.shape)     # input x
tf_y = tf.placeholder(tf.float32, y.shape)     # input y

# neural network layers
l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)          # hidden layer
output = tf.layers.dense(l1, 1)                     # output layer

loss = tf.losses.mean_squared_error(tf_y, output)   # compute cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_op = optimizer.minimize(loss)
time.sleep(3)
with tf.Session() as sess:                          # control training and others
	sess.run(tf.global_variables_initializer())         # initialize var in graph

	plt.ion()   # something about plotting
	fig, ax = plt.subplots(figsize=(20,16))
	
	for step in range(1000):
		# train and net output
		_, l, pred = sess.run([train_op, loss, output], {tf_x: x, tf_y: y})
		if step % 5 == 0:
			# plot and show learning process
			ax.cla()
			ax.scatter(x, y)
			ax.plot(x, pred, 'r-', lw=5)
			ax.text(0, -0.35, 'Loss=%.4f' % l, fontdict={'size': 60, 'color': 'red'})
			fig.canvas.flush_events()
			time.sleep(0.1)
			# ax.pause(0.1)

	plt.ioff()
	plt.show()
