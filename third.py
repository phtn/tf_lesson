import tensorflow as tf

# Variables

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

print(sess.run(linear_model, {x: [1,2,3,4]}))
