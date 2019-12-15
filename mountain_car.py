"""
This code is modeled off of code from sentdex on youtube.
Credit belongs to him
Changes made were to the Initial Population
"""

import gym
import random
import numpy as np
import tensorflow
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter


LR = 1e-3
env = gym.make("MountainCar-v0")
env.reset()
goal_steps = 300
score_requirement = -180
initial_games = 100


# training data
def initial_population():
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []
    # iterate through however many games we want:
    for _ in range(initial_games):
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []

        """
        Changes start here
        """
        # This is so that the if statements work
        first = 0
        # for each frame in 300
        for i in range(goal_steps):
            # env.render()
            # just something for it to do when prev_observation is null
            if first == 0:
                action = 0
                first = first + 1
            # if the velocity is greater than 0, then push right
            elif prev_observation[1] > 0:
                action = 2
            # if the velocity is less than 0, then push left
            elif prev_observation[1] < 0:
                action = 0
            # if velocity is 0, then don't push
            else:
                action = 0
            """
            Changes end here
            """
            # do it!
            observation, reward, done, info = env.step(action)

            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break

        # IF our score is higher than our threshold, we'd like to save
        # every move we made
        # NOTE the reinforcement methodology here.
        # all we're doing is reinforcing the score, we're not trying
        # to influence the machine in any way as to HOW that score is
        # reached.
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # print("Got here")
                # convert to one-hot (this is the output layer for our neural network)
                if data[1] == 1:
                    output = [2, 0, 1]
                elif data[1] == 0:
                    output = [1, 2, 0]
                elif data[1] == 2:
                    output = [1, 0, 2]

                # saving our training data
                training_data.append([data[0], output])

        # reset env to play again
        env.reset()
        # save overall scores
        scores.append(score)

    # just in case you wanted to reference later
    training_data_save = np.array(training_data)
    np.save('mountain_car.npy', training_data_save)

    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:', mean(accepted_scores))
    print('Median score for accepted scores:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

# initial_population()


# input size
def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 64, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 64, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 3, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, model=False):
    # grabs the observation done, and the one-hot conversion shaping the observations
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)

    # grabs only the one-hot stuff for actions
    y = [i[1] for i in training_data]


    if not model:
        model = neural_network_model(input_size=len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=3, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model


training_data = initial_population()
model = train_model(training_data)


scores = []
choices = []

for each_game in range(10):
    score = 0
    forcer = False
    game_memory = []
    prev_obs = []
    env.reset()
    for i in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            # One final change that just uses a different random
            action = random.randrange(0, 3, 2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if i == 200 and forcer == False or i == 999:
            forcer = False
        elif i < 1000:
            forcer = True
        elif done:
            break
        # if done:
        #     break
    scores.append(score)


print('Average Score:', sum(scores) / len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))
print(score_requirement)

model.save('mountaincar.model')