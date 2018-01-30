import image_input
import timeit
import tensorflow as tf

import net

TRAINING_STEPS = 100000
BATCH_SIZE = 1

start=timeit.default_timer()
stop=timeit.default_timer()


x_image = tf.placeholder("uint8", [BATCH_SIZE, 60, 80])
x_prev_seg = tf.placeholder("uint8", [BATCH_SIZE, 60, 80])
x_image = tf.cast(x_image, "float32")
x_prev_seg = tf.cast(x_prev_seg, "float32")
x_dir = tf.placeholder("float32", [1])

y_seg, ya, yb, yc = net.deepnn(x_image, x_prev_seg)
y_seg_reshaped = tf.reshape(y_seg, [BATCH_SIZE* 60 * 80])
y_gt = tf.placeholder("uint8", [BATCH_SIZE, 60, 80])
y_gt_reshaped = tf.reshape(tf.cast(y_gt, tf.float32), [BATCH_SIZE* 60 * 80])

loss = tf.losses.mean_pairwise_squared_error(y_seg_reshaped, y_gt_reshaped)

cross_entropy = loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
y_seg_reshaped = tf.where(
    tf.less(y_seg_reshaped, tf.zeros_like(y_seg_reshaped) + 1),
    tf.zeros_like(y_seg_reshaped),
    tf.ones_like(y_seg_reshaped)*255)

correct_prediction = tf.equal(y_seg_reshaped, y_gt_reshaped)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(TRAINING_STEPS):
    image, seg_prev, seg_gt = image_input.get_rand_train_batch(BATCH_SIZE)
    train_step.run(feed_dict={x_image: image, x_prev_seg: seg_prev, y_gt: seg_gt})
    x1, y_sega, yaa, ybb,ycc, y1, y2 = sess.run([x_image, y_seg, ya, yb, yc, y_gt_reshaped, y_seg_reshaped], feed_dict={x_image: image, x_prev_seg: seg_prev, y_gt: seg_gt})

    print("x1", x1, "y1:", y1,"\ny2", y2)
if __name__ == "__main__":
    tf.app.run()
