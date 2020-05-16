import copy
import json
import logging

import gym
from gym import spaces
import numpy as np

import reporting
from utils import estimate_skills
import sys
from tabulate import tabulate

MEAN_START_SKILL_LEVEL = 20
STD_START_SKILL_LEVEL = 10

POPULATION_MEAN_SKILL_GAIN = 2
POPULATION_STD_SKILL_GAIN = 1.8
POPULATION_STD_TYPE_GAIN = 0.01
POPULATION_MIN_SKILL_GAIN = 0.1

STUDENT_SKILL_GAIN_STD = 0.2

TEST_SCORE_STD = 0.5

TARGET_SCORE = 95

REVIEW_RATIO = 0.25

REWARD_FOR_ACHIEVING_TARGET_LEVEL = 100
REWARD_FOR_ACHIEVING_ALL_LEVELS = 1000

PENALTY_FOR_UNNECESSARY_TEST = -700
TIME_PENALTY_FOR_TEST = - 2
GAIN_MULTIPLIER_FOR_TEST = 0

NOT_ADAPTED_DIFFICULTY_PENALTY = 0.25

GAIN_REWARD_RATIO = 0.1


class StudentEnv(gym.Env):
    def __init__(self, num_subjects=3, num_difficulty_levels=3, num_train_types=3):
        super(StudentEnv).__init__()

        # parameters cache
        self.num_subjects = num_subjects
        self.num_difficulty_levels = num_difficulty_levels
        self.num_train_types = num_train_types

        # define constants
        self.difficulty_thresholds = np.linspace(0, 100, num=self.num_difficulty_levels, endpoint=False)
        self.review_ratio = 1 / (self.num_difficulty_levels + 1)

        # define action space & observation space
        self.action_space = spaces.MultiDiscrete([
            2,  # train or test
            num_subjects,  # which subject the action refers to
            num_difficulty_levels,  # test difficulty level (not used if action=train)
            num_train_types,  # train type (not used if action=test)
            num_difficulty_levels  # train difficulty level (not used if action=test)
        ])
        low_bound_observation_space_vector = np.array([
            0,  # min test score
            -100,  # min difference between previous test score and current test score (later called gain)
            *np.repeat(0, num_train_types),  # min number of trainings since last test for each training type
            *np.repeat(-100, num_train_types),  # min gain attributed to each training type
        ])
        high_bound_observation_space_vector = np.array([
            100,  # max test score
            100,  # max difference between previous test score and current test score (later called gain)
            *np.repeat(sys.maxsize, num_train_types),  # max number of trainings since last test for each training type
            *np.repeat(100, num_train_types),  # max gain attributed to each training type
        ])
        self.observation_space = spaces.Box(
            low=np.tile(low_bound_observation_space_vector, (num_subjects, num_difficulty_levels, 1)),
            high=np.tile(high_bound_observation_space_vector, (num_subjects, num_difficulty_levels, 1))
        )

        self.episode = -1

        # define all variables that are to be reset after each episode
        self.skill_levels = None  # comment
        self.state = None  # observed state
        self.mean_skill_gains = None  # comment
        self.cum_train_time = None  # comment
        self.train_counter = None  # number of trainings of each kind since the beginning of the current episode
        self.step_num = None  # number of steps in the current episode
        self.last_action = None  # a brief description of the last action for reporting purposes

        self.reset()

    def render(self, mode='human'):
        action_to_str = ';'.join(f'{k}={v}' for k, v in self.last_action.items())
        last_scores = self.state
        types = {f'Training type number {i + 1}': last_scores[:, :, -self.num_train_types + i].round(3)
                 for i in range(self.num_train_types)}
        table = {'Test matrix': last_scores[:, :, 0].round(1)}
        table.update(types)
        print(f'***\n'
              f'Action: {action_to_str}\n'
              f'{tabulate(table, headers="keys")}\n'
              f'Latent skill level: {self.skill_levels.round(1)}\n'
              f'***')
        logging.info(json.dumps({**self.last_action, 'skills': self.skill_levels, 'step': self.step_num,
                                 'episode': self.episode},
                                cls=reporting.NpEncoder))
        return self.last_action

    def reset(self):
        self.skill_levels = np.maximum(
            np.random.normal(MEAN_START_SKILL_LEVEL, STD_START_SKILL_LEVEL, size=self.num_subjects), 0
        )
        self.state = np.zeros(shape=(
            self.num_subjects,
            self.num_difficulty_levels,
            2 * self.num_train_types + 2
        ))
        self.mean_skill_gains = self._sample_mean_skills_gains()
        self.cum_train_time = np.zeros(self.num_subjects)
        self.train_counter = np.zeros((self.num_subjects, self.num_difficulty_levels, self.num_train_types))

        self.last_action = None

        self.episode += 1
        self.step_num = 0
        return self.state

    def step(self, action):
        assert self.action_space.contains(action)

        is_test, subject, test_difficulty, train_type, train_difficulty = action
        self.last_action = {
            'action': ['train', 'test'][is_test],
            'subject': subject + 1,
            'difficulty': (test_difficulty if is_test else train_difficulty) + 1
        }

        reward = TIME_PENALTY
        is_done = False
        if is_test:
            reward += self._test(subject, test_difficulty)
            self.cum_train_time[subject] = 0
            if (self.state[:, -1, 0] > TARGET_SCORE).all():
                is_done = True
                reward += REWARD_FOR_ACHIEVING_ALL_LEVELS
        else:
            reward += self._train(subject, train_type, train_difficulty)

        self.last_action['reward'] = reward
        self.step_num += 1
        return self.state, reward, is_done, {}

    def _sample_mean_skills_gains(self):
        skill_gain_matrix = np.tile(np.random.normal(POPULATION_MEAN_SKILL_GAIN, POPULATION_STD_SKILL_GAIN,
                                                     size=self.num_train_types), (self.num_subjects, 1))
        skill_gain_matrix += np.random.normal(0, POPULATION_STD_TYPE_GAIN,
                                              size=(self.num_subjects, self.num_train_types))
        return np.maximum(skill_gain_matrix, POPULATION_MIN_SKILL_GAIN)

    def _test(self, subject, difficulty):
        test_mean = self._get_test_mean(subject, difficulty)
        prev_test_score = self.state[subject, difficulty, 0]
        prev_test_scores = copy.copy(self.state[:, :, 0])

        new_test_score = min(max(np.random.normal(test_mean, TEST_SCORE_STD), 0), 100)
        self.last_action['test_score'] = new_test_score

        self.state[subject, difficulty, 0] = new_test_score
        self.state[subject, difficulty, 1] = new_test_score - prev_test_score

        estimated_gain = estimate_skills(self.state[:, :, 0], REVIEW_RATIO)[subject] - \
                                estimate_skills(prev_test_scores, REVIEW_RATIO)[subject]

        self.state[subject, difficulty, -self.num_train_types:] = self._get_mean_type_gain(subject, difficulty, estimated_gain)
        self.state[subject, difficulty, 2:2 + self.num_train_types] = 0

        reward = TIME_PENALTY_FOR_TEST
        if self.cum_train_time[subject] > 0:
            reward += GAIN_MULTIPLIER_FOR_TEST * (self.state[subject, difficulty, 1] / self.cum_train_time[subject])
        if new_test_score >= TARGET_SCORE:
            if prev_test_score < TARGET_SCORE:
                reward += REWARD_FOR_ACHIEVING_TARGET_LEVEL * (difficulty + 1) / self.num_difficulty_levels
            else:
                reward += PENALTY_FOR_UNNECESSARY_TEST
        return reward

    def _get_mean_type_gain(self, subject, difficulty, gain):
        num_trainings_since_last_test = self.state[subject, difficulty, 2:2 + self.num_train_types]
        if np.sum(num_trainings_since_last_test) > 0:
            relative_gain = gain / np.sum(num_trainings_since_last_test)
            result = np.zeros(self.num_train_types)
            for i, last_avg in enumerate(self.state[subject, difficulty, -self.num_train_types:]):
                if self.train_counter[subject, difficulty, i] > 0:
                    result[i] = np.average([last_avg, relative_gain], weights=[
                        self.train_counter[subject, difficulty, i], num_trainings_since_last_test[i]
                    ])
            self.train_counter[subject, difficulty, :] += num_trainings_since_last_test
            return result
        else:
            return self.state[subject, difficulty, -self.num_train_types:]

    def _get_test_mean(self, subject, difficulty):
        proper_difficulty = self._get_proper_difficulty(self.skill_levels[subject])
        if proper_difficulty > difficulty:
            return 100

        scaled_mean = self._get_scaled_mean_score(subject, proper_difficulty)
        if proper_difficulty == difficulty:
            review_part = self.review_ratio if difficulty > 0 else 0
            return review_part * 100 + (1 - review_part) * scaled_mean
        else:
            return self.review_ratio ** (difficulty - proper_difficulty) * scaled_mean

    def _get_scaled_mean_score(self, subject, difficulty):
        return (self.skill_levels[subject] - self.difficulty_thresholds[difficulty]) * self.num_difficulty_levels

    def _train(self, subject, train_type, train_difficulty):
        mean_gain = self.mean_skill_gains[subject, train_type]
        gain = np.random.normal(mean_gain, STUDENT_SKILL_GAIN_STD)
        adjusted_gain = max(0, gain * self._get_not_adapted_train_penalty(self.skill_levels[subject], train_difficulty))

        self.skill_levels[subject] = min(self.skill_levels[subject] + adjusted_gain, 100)
        self.last_action['improvement'] = adjusted_gain
        self.last_action['training_type'] = train_type + 1
        self.cum_train_time[subject] += 1
        self.state[subject, train_difficulty, 2 + train_type] += 1
        # estimated_skill = estimate_skills(self.state[:, :, 0], REVIEW_RATIO)[subject]
        # estimated_penalty = self._get_not_adapted_train_penalty(estimated_skill, train_difficulty)
        # estimated_gain = POPULATION_MEAN_SKILL_GAIN * train_type
        # adapted_learning_reward = estimated_penalty * estimated_gain * GAIN_REWARD_RATIO
        # return 0 - (learning_type + 1) + adapted_learning_reward
        return 0

    def _get_not_adapted_train_penalty(self, skill, train_difficulty):
        proper_difficulty = self._get_proper_difficulty(skill)
        return NOT_ADAPTED_DIFFICULTY_PENALTY ** abs(train_difficulty - proper_difficulty)

    def _get_proper_difficulty(self, skill):
        return sum(self.difficulty_thresholds <= skill) - 1


class StudentEnvBypass(StudentEnv):
    def __init__(self, studentenvcopy, prob_ratio=None):
        num_subjects, num_difficulty_levels, num_learning_types = studentenvcopy.num_subjects, \
                                                                  studentenvcopy.difficulties_levels, \
                                                                  studentenvcopy.learning_type_number
        super(StudentEnvBypass, self).__init__(num_subjects, num_difficulty_levels, num_learning_types)
        self.last_scores = copy.deepcopy(studentenvcopy.last_scores)
        self.cumulative_train_time = copy.deepcopy(studentenvcopy.cumulative_train_time)
        self.train_counter = copy.deepcopy(studentenvcopy.train_counter)
        self.episode = copy.deepcopy(studentenvcopy.episode)
        self.step_num = copy.deepcopy(studentenvcopy.step_num)
        self.prob_ratio = prob_ratio if prob_ratio else [0.8, 0.1, 0.1]
        self.mean_skill_gains = copy.deepcopy(studentenvcopy.mean_skill_gains)
        self.skills_levels = copy.deepcopy(studentenvcopy.skills_levels)

    def step(self, action):
        assert self.action_space.contains(action)
        is_test, subject, test_difficulty, learning_types, learning_difficulty = action
        difficulty_to_log = test_difficulty if is_test else learning_difficulty
        self.last_action = {
            'action': ['train', 'test'][is_test],
            'subject': subject + 1,
            'difficulty': difficulty_to_log + 1
        }

        if is_test:
            reward = self._test(subject, test_difficulty)
            self.cumulative_train_time[subject] = 0
            if (self.last_scores[:, -1, 0] > TARGET_SCORE).all():
                is_done = 1
                reward += REWARD_FOR_ACHIEVING_ALL_LEVELS
            else:
                is_done = 0
        else:
            learning_types = int(np.random.choice(self.difficulties_levels, p=self.prob_ratio))
            reward = self._train(subject, learning_types, learning_difficulty)
            is_done = 0
        reward += - np.sqrt(self.step_num)
        self.last_action['reward'] = reward
        self.step_num += 1
        return self.last_scores, reward, is_done, {}

    def render(self, mode='human'):
        action_to_str = ';'.join(f'{k}={v}' for k, v in self.last_action.items())
        last_scores = self.last_scores
        types = {f'Learning type number {i + 1}': last_scores[:, :, -self.learning_type_number + i].round(3)
                 for i in range(self.learning_type_number)}
        table = {'Test matrix': last_scores[:, :, 0].round(1)}
        table.update(types)
        if self.last_action['action'] == 'test':
            table.update({f'Train counters {i + 1}': last_scores[:, :, 2 + i].round(3)
                          for i in range(self.learning_type_number)})
        print(f'***\n'
              f'Action: {action_to_str}\n' +
              tabulate(table, headers='keys') + '\n'
              f'Latent skill level: {self.skills_levels.round(1)}\n'
                                                f'***')
        logging.info(json.dumps({**self.last_action, 'skills': self.skills_levels, 'step': self.step_num,
                                 'episode': self.episode,
                                 'env': f'bias for {np.argmax(self.prob_ratio)+1} learning type'},
                                cls=reporting.NpEncoder))
        return self.last_action


