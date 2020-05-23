import copy
import json
import logging
import sys

import gym
from gym import spaces
import numpy as np
from tabulate import tabulate

import reporting
import settings
from utils import estimate_skills


class State:
    def __init__(self, num_subjects, num_difficulty_levels, num_train_types):
        self.last_test_scores = np.zeros(shape=(
            num_subjects, num_difficulty_levels)
        )
        self.last_test_improvements = np.zeros(shape=(
            num_subjects, num_difficulty_levels)
        )
        self.trainings_by_type_counter = np.zeros(shape=(
            num_subjects, num_train_types
        ))
        self.estimated_gains = np.zeros(shape=(
            num_train_types
        ))

    def reset(self):
        self.last_test_scores = np.zeros_like(self.last_test_scores)
        self.last_test_improvements = np.zeros_like(self.last_test_improvements)
        self.trainings_by_type_counter = np.zeros_like(self.trainings_by_type_counter)
        self.estimated_gains = np.zeros_like(self.estimated_gains)

    def get_observation_space(self):
        low_bound = np.concatenate([
            np.zeros_like(self.last_test_scores),  # min test score
            np.full_like(self.last_test_improvements, -100), # main difference between previous test score and current test score (later called gain)
            np.zeros_like(self.trainings_by_type_counter), # min number of trainings since last test for each training type
            np.full_like(self.estimated_gains, -100)],  # min gain attributed to each training type
            axis=None)

        high_bound = np.concatenate([
            np.full_like(self.last_test_scores, 100),  # max test score
            np.full_like(self.last_test_improvements, 100), # max difference between previous test score and current test score (later called gain)
            np.full_like(self.trainings_by_type_counter, sys.maxsize), # max number of trainings since last test for each training type
            np.full_like(self.estimated_gains, 100)],  # max gain attributed to each training type
            axis=None)
        return spaces.Box(low=low_bound, high=high_bound)

    def get_observation(self):
        return np.concatenate([
            self.last_test_scores,
            self.last_test_improvements,
            self.trainings_by_type_counter,
            self.estimated_gains
        ], axis=None)


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
        self.state = State(self.num_subjects, self.num_difficulty_levels, self.num_train_types)

        # required by gym.Env
        self.observation_space = self.state.get_observation_space()

        self.episode = -1

        # define all variables that are to be reset after each episode
        self.skill_levels = None  # comment
        self.mean_skill_gains = None  # comment
        self.cum_train_time = None  # comment
        self.train_counter = None  # number of trainings of each kind since the beginning of the current episode
        self.step_num = None  # number of steps in the current episode
        self.last_action = None  # a brief description of the last action for reporting purposes

        self.reset()

    def render(self, mode='human'):
        action_to_str = ';'.join(f'{k}={v}' for k, v in self.last_action.items())
        types = {f'Training type number {i + 1}': self.state.estimated_gains[:, :, i].round(3)
                 for i in range(self.num_train_types)}
        table = {'Test matrix': self.state.last_test_scores.round(1)}
        table.update(types)
        if self.last_action['action'] == 'test':
            table.update({
                f'Train counters {i + 1}': self.state.trainings_by_type_counter[:, :, i].round(3)
                for i in range(self.num_train_types)
            })
        print(f'***\n'
              f'Action: {action_to_str}\n'
              f'{tabulate(table, headers="keys")}\n'
              f'Latent skill level: {self.skill_levels.round(1)}\n'
              f'***')
        logging.info(json.dumps(self._get_dict_to_log(), cls=reporting.NpEncoder))
        return self.last_action

    def reset(self):
        self.state.reset()

        self.skill_levels = np.maximum(
            np.random.normal(settings.MEAN_START_SKILL_LEVEL, settings.STD_START_SKILL_LEVEL, size=self.num_subjects), 0
        )
        self.mean_skill_gains = self._sample_mean_skills_gains()
        self.cum_train_time = np.zeros(self.num_subjects)
        self.train_counter = np.zeros((self.num_subjects, self.num_train_types))

        self.last_action = None

        self.episode += 1
        self.step_num = 0
        return self.state.get_observation()

    def step(self, action):
        assert self.action_space.contains(action)

        is_test, subject, test_difficulty, train_type, train_difficulty = action
        self.last_action = {
            'action': ['train', 'test'][is_test],
            'subject': subject + 1,
            'difficulty': (test_difficulty if is_test else train_difficulty) + 1
        }

        reward = settings.TIME_PENALTY
        is_done = False
        if is_test:
            reward += self._test(subject, test_difficulty)
            self.cum_train_time[subject] = 0
            if self._is_learning_done():
                is_done = True
                reward += settings.REWARD_FOR_ACHIEVING_ALL_LEVELS
        else:
            reward += self._train(subject, train_type, train_difficulty)

        self.last_action['reward'] = reward
        self.step_num += 1
        return self.state.get_observation(), reward, is_done, {}

    def _is_learning_done(self):
        highest_difficulty_scores = self.state.last_test_scores[:, -1]
        return (highest_difficulty_scores > settings.TARGET_SCORE).all()

    def _get_dict_to_log(self):
        return {
            **self.last_action,
            'skills': self.skill_levels,
            'step': self.step_num,
            'episode': self.episode,
            'env': 'standard'
        }

    def _sample_mean_skills_gains(self):
        skill_gain_matrix = np.tile(np.random.normal(settings.POPULATION_MEAN_SKILL_GAIN,
                                                     settings.POPULATION_STD_SKILL_GAIN,
                                                     size=self.num_train_types), (self.num_subjects, 1))
        skill_gain_matrix += np.random.normal(0, settings.POPULATION_STD_TYPE_GAIN,
                                              size=(self.num_subjects, self.num_train_types))
        return np.maximum(skill_gain_matrix, settings.POPULATION_MIN_SKILL_GAIN)

    def _test(self, subject, difficulty):
        test_mean = self._get_test_mean(subject, difficulty)
        prev_test_score = self.state.last_test_scores[subject, difficulty]
        prev_test_scores = copy.copy(self.state.last_test_scores)

        new_test_score = min(max(np.random.normal(test_mean, settings.TEST_SCORE_STD), 0), 100)
        self.last_action['test_score'] = new_test_score

        self.state.last_test_scores[subject, difficulty] = new_test_score
        self.state.last_test_improvements[subject, difficulty] = new_test_score - prev_test_score

        estimated_gain = estimate_skills(self.state.last_test_scores, settings.REVIEW_RATIO)[subject] - \
            estimate_skills(prev_test_scores, settings.REVIEW_RATIO)[subject]

        self.state.estimated_gains = self._get_mean_type_gain(subject, difficulty, estimated_gain)
        self.state.trainings_by_type_counter[subject] = 0

        reward = settings.TIME_PENALTY_FOR_TEST
        if self.cum_train_time[subject] > 0:
            relative_improvement = self.state.last_test_improvements[subject, difficulty] / self.cum_train_time[subject]
            reward += settings.GAIN_MULTIPLIER_FOR_TEST * relative_improvement
        if new_test_score >= settings.TARGET_SCORE:
            if prev_test_score < settings.TARGET_SCORE:
                reward += settings.REWARD_FOR_ACHIEVING_TARGET_LEVEL * (difficulty + 1) / self.num_difficulty_levels
            else:
                reward += settings.PENALTY_FOR_UNNECESSARY_TEST
        return reward

    def _get_mean_type_gain(self, subject, difficulty, gain):
        num_trainings_since_last_test = self.state.trainings_by_type_counter[subject]
        if np.sum(num_trainings_since_last_test) > 0:
            relative_gain = gain / np.sum(num_trainings_since_last_test)
            result = np.zeros(self.num_train_types)
            for i, last_avg in enumerate(self.state.estimated_gains):
                if self.train_counter[subject, i] > 0:
                    result[i] = np.average([last_avg, relative_gain], weights=[
                        self.train_counter[subject, i], num_trainings_since_last_test[i]
                    ])
            self.train_counter[subject, :] += num_trainings_since_last_test
            return result
        else:
            return self.state.estimated_gains

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
        gain = np.random.normal(mean_gain, settings.STUDENT_SKILL_GAIN_STD)
        adjusted_gain = max(0, gain * self._get_not_adapted_train_penalty(self.skill_levels[subject], train_difficulty))

        self.skill_levels[subject] = min(self.skill_levels[subject] + adjusted_gain, 100)
        self.last_action['improvement'] = adjusted_gain
        self.last_action['training_type'] = train_type + 1
        self.cum_train_time[subject] += 1
        self.state.trainings_by_type_counter[subject, train_type] += 1
        # estimated_skill = estimate_skills(self.state[:, :, 0], REVIEW_RATIO)[subject]
        # estimated_penalty = self._get_not_adapted_train_penalty(estimated_skill, train_difficulty)
        # estimated_gain = POPULATION_MEAN_SKILL_GAIN * train_type
        # adapted_learning_reward = estimated_penalty * estimated_gain * GAIN_REWARD_RATIO
        # return 0 - (learning_type + 1) + adapted_learning_reward
        return 0

    def _get_not_adapted_train_penalty(self, skill, train_difficulty):
        proper_difficulty = self._get_proper_difficulty(skill)
        return settings.NOT_ADAPTED_DIFFICULTY_PENALTY ** abs(train_difficulty - proper_difficulty)

    def _get_proper_difficulty(self, skill):
        return sum(self.difficulty_thresholds <= skill) - 1


class StudentEnvBypass(StudentEnv):
    def __init__(self, env: StudentEnv, prob_ratio):
        super(StudentEnvBypass, self).__init__(env.num_subjects, env.num_difficulty_levels, env.num_train_types)

        # deep-copy all
        self.skills_levels = copy.deepcopy(env.skill_levels)
        self.state = copy.deepcopy(env.state)
        self.cum_train_time = copy.deepcopy(env.cum_train_time)
        self.train_counter = copy.deepcopy(env.train_counter)
        self.mean_skill_gains = copy.deepcopy(env.mean_skill_gains)
        self.step_num = copy.deepcopy(env.step_num)
        self.episode = copy.deepcopy(env.episode)
        self.last_action = copy.deepcopy(env.last_action)

        self.prob_ratio = prob_ratio

    def step(self, action):
        forced_train_type = int(np.random.choice(self.num_train_types, p=self.prob_ratio))
        action[3] = forced_train_type
        return super().step(action)

    def _get_dict_to_log(self):
        parent_dict = super()._get_dict_to_log()
        parent_dict['env'] = f'bias for {np.argmax(self.prob_ratio) + 1} training type'
        return parent_dict


