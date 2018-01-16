import tensorflow as tf

# Placeholders

a = tf.placeholder(tf.float32)

b = tf.placeholder(tf.float32)

adder_node = a + b

sess = tf.Session()

# simple addition
print(sess.run(adder_node, {a: 6, b: 3}))
# adding array
print(sess.run(adder_node, {a: [6, 5], b: [3 ,7]}))

add_and_triple = adder_node * 3

print(sess.run(add_and_triple, {a: [6, 5], b: [3 ,7]}))

