import tensorflow as tf

# Creating Loss Function

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b

y = tf.placeholder(tf.float32)

squared_deltas = tf.square(linear_model - y)

loss = tf.reduce_sum(squared_deltas)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

# reassign values
fixW = tf.assign(W, [-1])
fixb = tf.assign(b, [1])

sess.run([fixW, fixb])

print(sess.run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]} ))
