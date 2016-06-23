import tensorflow as tf

class SLP:
    def __init__(s, input_size, hidden_size, output_size, alpha=0.01):
        s.input_size = input_size
        s.hidden_size = hidden_size
        s.output_size = output_size
        s.W_hidden = tf.Variable(tf.truncated_normal([input_size, hidden_size]))
        s.b_hidden = tf.Variable(tf.truncated_normal([hidden_size]))
        s.W_output = tf.Variable(tf.truncated_normal([hidden_size, output_size]))
        s.b_output = tf.Variable(tf.truncated_normal([output_size]))
        s.x = tf.placeholder("float", [None, s.input_size])  # "None" as dimension for versatility between batches and non-batches
        s.y_ = tf.placeholder("float", [None, s.output_size])
        y_hidden = tf.sigmoid(tf.matmul(s.x, s.W_hidden) + s.b_hidden)
        # y = tf.tanh(tf.matmul(y_hidden, s.W_output) + s.b_output)
        s.ylogits = tf.matmul(y_hidden, s.W_output) + s.b_output
        s.y = tf.nn.softmax(s.ylogits)
        # s.error_measure = tf.reduce_sum(tf.square(s.y_ - s.y))
        # s.error_measure = tf.reduce_mean(tf.reduce_mean(-s.y_*tf.log(s.y)))
        s.error_measure = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(s.ylogits, s.y_))
        s.train = tf.train.GradientDescentOptimizer(alpha).minimize(s.error_measure)
        s.init = tf.initialize_all_variables()
        s.sess = tf.Session()
        s.sess.run(s.init)

    def feed_batch(s, batch, expected_outputs):
        s.sess.run(s.train, feed_dict={s.x: batch, s.y_: expected_outputs})

    def error(s,batch,expected_outputs):
        return s.sess.run(s.error_measure, feed_dict={s.x: batch, s.y_: expected_outputs})

    def categorize(s, data):
        return s.sess.run(s.y, feed_dict={s.x: data})