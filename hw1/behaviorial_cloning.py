# -*- coding: utf-8 -*-

"""

Behavioral Cloning on the Hopper-v1 environment

"""


import gym
import pickle
import random
import tf_util
import argparse
import numpy as np
import tensorflow as tf

NUM_STATE = 11
NUM_ACTIONS = 3
ENV_NAME = 'Hopper-v1'


def fc(x, size, name):
    in_ = x.shape[1]
    with tf.variable_scope(name):
        w = tf.get_variable('w', shape=[in_, size])
        b = tf.get_variable('b', shape=[size])
        result = tf.add(tf.matmul(x, w), b)
    return result


def policy_function(observation_):
    with tf.variable_scope('policy_function'):
        x = fc(observation_, 128, 'fc_1')
        x = fc(x, 64, 'fc_2')
        x = fc(x, NUM_ACTIONS, 'fc_3')
    return x


# def one_hot(lbl, dim):
#
#     label = np.zeros(shape=dim, dtype=np.int16)
#     idx = np.argmax(lbl, axis=1)
#     label[range(dim[0]), idx] = 1
#     return label


def load_exper_data(data_path):

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    return data


def train(sess, model, data, batch_size, epoches=100):

    train_state = data['observations']
    train_action = data['actions']
    train_data = zip(train_state, train_action)
    random.shuffle(train_data)
    ratio = 0.2
    split_index = int(ratio*len(train_data))
    valid_data = train_data[:split_index]
    train_data = train_data[split_index:]
    print(len(train_data))
    num_samples = len(train_data)
    input = tf.placeholder(tf.float32, shape=[None, NUM_STATE])
    output = model(input)
    label = tf.placeholder(tf.int16, shape=[None, NUM_ACTIONS])

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square((output-label)))#(logits=output, labels=label))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('train_step'):
        train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    sess.run(tf.global_variables_initializer())

    for epoch in xrange(epoches):
        random.shuffle(train_data)
        for idx in xrange(num_samples/batch_size):
            batch_data = train_data[idx*batch_size: (idx+1)*batch_size]
            batch_state = np.array([x[0] for x in batch_data])
            batch_action = np.array([x[1] for x in batch_data])
            batch_action = np.reshape(batch_action, (batch_size, NUM_ACTIONS))
            # batch_action = one_hot(batch_action, (batch_size, NUM_ACTIONS))

            _loss, _ = sess.run([loss, train_step], feed_dict={input: batch_state, label: batch_action})

        test_state = np.array([x[0] for x in valid_data])
        test_action = np.array([x[1] for x in valid_data])
        test_action = np.reshape(test_action, (len(valid_data), NUM_ACTIONS))
        # test_action = one_hot(test_action, (len(valid_data), NUM_ACTIONS))

        _loss = sess.run(loss, feed_dict={input: test_state, label: test_action})
        print("[Test][Epoch: {}][Loss: {}]".format(epoch, _loss))


def __test_train__():

    data = load_exper_data('expert_data/Hopper-v1_20.pkl')

    with tf.Session() as sess:
        train(sess, policy_function, data, batch_size=64)


def behavioral_cloning(batch_size=64, epoches=50):

    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=20, help='Number of roll outs')

    args = parser.parse_args()

    print("Loading Expert Data")
    data = load_exper_data('expert_data/Hopper-v1_20.pkl')

    train_state = data['observations']
    train_action = data['actions']
    train_data = zip(train_state, train_action)
    random.shuffle(train_data)
    ratio = 0.2
    split_index = int(ratio*len(train_data))
    valid_data = train_data[:split_index]
    train_data = train_data[split_index:]
    print(len(train_data))
    num_samples = len(train_data)
    input = tf.placeholder(tf.float32, shape=[None, NUM_STATE])
    output = policy_function(input)
    label = tf.placeholder(tf.float32, shape=[None, NUM_ACTIONS])

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square((output-label)))#(logits=output, labels=label))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('train_step'):
        train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    with tf.Session() as sess:

        # training
        sess.run(tf.global_variables_initializer())

        for epoch in xrange(epoches):
            random.shuffle(train_data)
            for idx in xrange(num_samples / batch_size):
                batch_data = train_data[idx * batch_size: (idx + 1) * batch_size]
                batch_state = np.array([x[0] for x in batch_data])
                batch_action = np.array([x[1] for x in batch_data])
                batch_action = np.reshape(batch_action, (batch_size, NUM_ACTIONS))
                #batch_action = one_hot(batch_action, (batch_size, NUM_ACTIONS))

                _loss, _ = sess.run([loss, train_step], feed_dict={input: batch_state, label: batch_action})

            test_state = np.array([x[0] for x in valid_data])
            test_action = np.array([x[1] for x in valid_data])
            test_action = np.reshape(test_action, (len(valid_data), NUM_ACTIONS))

            _loss = sess.run(loss, feed_dict={input: test_state, label: test_action})
            print("[Test][Epoch: {}][Loss: {}]".format(epoch, _loss))

        print("\nFinish Training ...\n")

        # testing
        env = gym.make(ENV_NAME)
        max_steps = env.spec.timestep_limit

        returns = []
        observations = []
        actions = []

        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                obs = np.array([obs])
                action = sess.run(output, feed_dict={input: obs})[0]
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0:
                    print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))



def bc():

    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=20, help='Number of roll outs')

    args = parser.parse_args()

    env = gym.make(ENV_NAME)
    max_steps = env.spec.timestep_limit

    returns = []
    observations = []
    actions = []

    policy = None

    with tf.Session() as sess:




        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                obs = np.array([obs])
                action = policy_function(obs)[0]
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0:
                    print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))


if __name__ == '__main__':

    behavioral_cloning()


