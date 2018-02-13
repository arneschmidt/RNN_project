import image_input
import timeit
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import visualizer

import net as net

TRAINING_STEPS = 5000
BATCH_SIZE = 1 # batch size (only for train_batch)
SEQ_SIZE = 1 # size of the sequence (only for train_seq)
MODE = 'test' # choose: 'train_batch', 'train_seq', 'predict', 'test'
MODELNAME = 'final_model' # name under which the model is saved/loaded
LOAD_CHECKPOINT = True # choose weather model is loaded


start = timeit.default_timer()
stop = timeit.default_timer()


def classify_by_threshold(image, threshold):
    image = tf.where(tf.less(image, tf.zeros_like(image) + threshold),
                             tf.zeros_like(image),
                             tf.ones_like(image)*255)
    return image


def main(unused_argv):
    x_image = tf.placeholder("uint8", [BATCH_SIZE, 60, 80])
    x_prev_seg = tf.placeholder("uint8", [BATCH_SIZE, 60, 80])
    x_image = tf.cast(x_image, "float32")
    x_prev_seg = tf.cast(x_prev_seg, "float32")
    x_dir = tf.placeholder("float32", [BATCH_SIZE])

    y_seg, hidden_layers = net.deepnn(x_image, x_prev_seg, x_dir)
    y_seg_class = classify_by_threshold(y_seg, 120)
    y_seg_reshaped = tf.reshape(y_seg, [BATCH_SIZE * 60 * 80])
    y_seg_reshaped_class = classify_by_threshold(y_seg_reshaped, 120)
    y_gt = tf.placeholder("uint8", [BATCH_SIZE, 60, 80])
    y_gt = tf.cast(y_gt, tf.float32)
    y_gt_class = classify_by_threshold(y_gt, 120)
    y_gt_reshaped = tf.reshape(y_gt, [BATCH_SIZE * 60 * 80])
    y_gt_reshaped_class = classify_by_threshold(y_gt_reshaped, 120)

    loss = tf.nn.l2_loss(y_seg_reshaped - y_gt_reshaped)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    correct_prediction = tf.equal(y_seg_reshaped_class, y_gt_reshaped_class)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        if LOAD_CHECKPOINT == False:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, "./trained_models/" + MODELNAME + ".ckpt")
            print("Model restored.")

        if MODE == 'train_batch':
            for i in range(TRAINING_STEPS):
                image, seg_prev, seg_gt, dir = image_input.get_rand_train_batch(BATCH_SIZE)
                train_step.run(feed_dict={x_image: image, x_prev_seg: seg_prev, x_dir: dir, y_gt: seg_gt})

                if i % 100 == 0:
                    train_accuracy, y_seg_out, seg_gt_out, seg_prev_out, image_out, hidden_layers_out = \
                            sess.run([accuracy, y_seg_class, y_gt_class, x_prev_seg, x_image, hidden_layers],
                                 feed_dict={x_image: image, x_prev_seg: seg_prev, x_dir: dir, y_gt: seg_gt})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                    visualizer.show_images([seg_prev_out[0], image_out[0], seg_gt_out[0], y_seg_out[0]])
                    visualizer.show_activations(hidden_layers_out, 6)

            if input("Save model? y/n:") == 'y':
                save_path = saver.save(sess, "./trained_models/" + MODELNAME + ".ckpt")
                print("Model saved in file: %s" % save_path)

        elif MODE == 'train_seq':
            for i in range(TRAINING_STEPS):
                image_seq, seg_gt_seq, dir_seq = image_input.get_rand_train_sequence(SEQ_SIZE)
                seg_prev = [seg_gt_seq[0]]

                for j in range(1, SEQ_SIZE):
                    image = [image_seq[j]]
                    seg_gt = [seg_gt_seq[j]]
                    dir = [dir_seq[j]]

                    train_step.run(feed_dict={x_image: image, x_prev_seg: seg_prev, x_dir: dir, y_gt: seg_gt})
                    seg_prev = y_seg_class.eval(feed_dict={x_image: image, x_prev_seg: seg_prev, x_dir: dir, y_gt: seg_gt})

                if i % 100 == 0:
                    train_accuracy, y_seg_out, seg_gt_out, seg_prev_out, image_out, hidden_layers_out = sess.run(
                        [accuracy, y_seg_class, y_gt_class, x_prev_seg, x_image, hidden_layers], feed_dict={
                            x_image: image, x_prev_seg: seg_prev, x_dir: dir, y_gt: seg_gt})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                    visualizer.show_images([seg_prev_out[0], image_out[0], seg_gt_out[0], y_seg_out[0]])
                    #visualizer.show_activations(hidden_layers_out, 5)

            if input("Save model? y/n:") == 'y':
                save_path = saver.save(sess, "./trained_models/" + MODELNAME + ".ckpt")
                print("Model saved in file: %s" % save_path)


        elif MODE == 'test':
            samples = 2000
            image_seq, seg_gt_seq, dir_seq = image_input.get_test_data()
            seg_out = [np.ones_like(image_seq[0])*255]
            sum_accuracy = 0

            for i in range(samples):

                if i % 100 == 0:
                    seg_out = [np.ones_like(image_seq[0]) * 255]
                image = [image_seq[i]]
                seg_gt = [seg_gt_seq[i]]
                dir = [dir_seq[i]]

                accuracy_out, seg_out = sess.run([accuracy, y_seg_class], feed_dict={
                            x_image: image, x_prev_seg: seg_out, x_dir: dir, y_gt: seg_gt})
                sum_accuracy += accuracy_out

                visualizer.show_inference(image[0], seg_out[0])
                #time.sleep(0.2)

            test_accuracy = sum_accuracy / samples
            print("Test accuracy: ", test_accuracy)

        elif MODE == 'predict':
            for i in range(200):
                image, seg_prev, seg_gt, dir = image_input.get_rand_train_batch(1)

                train_accuracy, seg_out, seg_gt_out, seg_prev_out, image_out, hidden_layers_out = sess.run(
                    [accuracy, y_seg_class, y_gt_class, x_prev_seg, x_image, hidden_layers], feed_dict={
                        x_image: image, x_prev_seg: seg_prev, x_dir: dir, y_gt: seg_gt})
                visualizer.show_images([seg_prev_out[0], image_out[0], seg_gt_out[0], seg_out[0]])
                visualizer.show_activations(hidden_layers_out, 5)
                input("Continue?")

if __name__ == "__main__":
    tf.app.run()
    main()
