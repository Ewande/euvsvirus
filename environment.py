import json
import logging

import gym
from gym import spaces
import numpy as np

import reporting

MEAN_START_SKILL_LEVEL = 20
STD_START_SKILL_LEVEL = 10


POPULATION_MEAN_SKILL_GAIN = 2
POPULATION_STD_SKILL_GAIN = 1
POPULATION_MIN_SKILL_GAIN = 0.2

STUDENT_SKILL_GAIN_STD_MEAN_RATIO = 0.5

TEST_SCORE_STD = 5

TARGET_SCORE = 90

REVIEW_RATIO = 0.25

REWARD_FOR_ACHIEVING_TARGET_LEVEL = 100
REWARD_FOR_ACHIEVING_ALL_LEVELS = 500

PENALTY_FOR_UNNECESSARY_TEST = -0.5 * REWARD_FOR_ACHIEVING_TARGET_LEVEL
TIME_PENALTY_FOR_TEST = - 5
GAIN_MULTIPLIER_FOR_TEST = 20

NOT_ADAPTED_DIFFICULTY_PENALTY = 0.25


class StudentEnv(gym.Env):
    def __init__(self, subjects_number=4, difficulties_levels=3, learning_units_number=3):
        super(StudentEnv).__init__()
        self.action_space = spaces.MultiDiscrete([2, subjects_number, difficulties_levels,
                                                  learning_units_number, difficulties_levels])
        self.observation_space = spaces.Box(low=0, high=100, shape=(subjects_number, difficulties_levels))
        self.difficulties_levels = difficulties_levels
        self.skills_levels = np.maximum(
            np.random.normal(MEAN_START_SKILL_LEVEL, STD_START_SKILL_LEVEL, size=subjects_number),
            np.zeros(subjects_number)
        )
        self.last_scores = np.zeros(shape=(subjects_number, difficulties_levels))
        self.mean_skill_gains = _get_mean_skills_gains(subjects_number, learning_units_number)
        self.difficulties_thresholds = np.linspace(0, 100, num=difficulties_levels, endpoint=False)
        self.review_ratio = 1 / (difficulties_levels + 1)
        self.cumulative_train_time = 0
        self.episode = 0
        self.last_action = None

    def step(self, action):
        assert self.action_space.contains(action)
        is_test, subject, test_difficulty, learning_units, learning_difficulty = action
        self.last_action = {
            'action': ['train', 'test'][is_test],
            'subject': subject + 1,
            'difficulty': test_difficulty + 1,
        }

        if is_test:
            reward = self._test(subject, test_difficulty)
            self.cumulative_train_time = 0
            if (self.last_scores[:, -1] > TARGET_SCORE).all():
                is_done = 1
                reward += REWARD_FOR_ACHIEVING_ALL_LEVELS
            else:
                is_done = 0
        else:
            reward = self._train(subject, learning_units, learning_difficulty)
            is_done = 0
        self.last_action['reward'] = reward
        return self.last_scores, reward, is_done, {}

    def _test(self, subject, difficulty):
        test_mean = self._get_test_mean(subject, difficulty)
        previous_score = self.last_scores[subject, difficulty]
        sampled_test_score = np.random.normal(test_mean, TEST_SCORE_STD)
        self.last_scores[subject, difficulty] = min(max(sampled_test_score, 0), 100)
        self.last_action['test_score'] = self.last_scores[subject, difficulty]
        if not self.cumulative_train_time:
            return TIME_PENALTY_FOR_TEST
        if self.last_scores[subject, difficulty] >= TARGET_SCORE:
            # if not self.skills_level_achieved[subject] and difficulty+1 == self.difficulties_level_nb:
            if previous_score < TARGET_SCORE:
                return REWARD_FOR_ACHIEVING_TARGET_LEVEL * (difficulty+1)/self.difficulties_levels
            else:
                return PENALTY_FOR_UNNECESSARY_TEST
        return GAIN_MULTIPLIER_FOR_TEST*((self.last_scores[subject, difficulty] - previous_score)
                                         / self.cumulative_train_time) + TIME_PENALTY_FOR_TEST

    def _get_test_mean(self, subject, difficulty):
        proper_difficulty = self._get_proper_difficulty(subject)
        if proper_difficulty < difficulty:
            return self._get_too_hard_test_mean(subject, difficulty, proper_difficulty)
        if proper_difficulty > difficulty:
            return 100
        return self._get_proper_test_mean(subject, difficulty)

    def _get_too_hard_test_mean(self, subject, difficulty, proper_difficulty):
        review_mean = self._get_scaled_mean_score(subject, proper_difficulty)
        return self.review_ratio ** (difficulty - proper_difficulty) * review_mean

    def _get_proper_test_mean(self, subject, difficulty):
        proper_mean = self._get_scaled_mean_score(subject, difficulty)
        review_score = self.review_ratio * 100 if difficulty else 0
        return review_score + proper_mean

    def _get_scaled_mean_score(self, subject, difficulty):
        return (self.skills_levels[subject] - self.difficulties_thresholds[difficulty]) * self.difficulties_levels

    def _train(self, subject, learning_unit, learning_difficulty):
        mean_gain = self.mean_skill_gains[subject, learning_unit]
        std_gain = mean_gain * STUDENT_SKILL_GAIN_STD_MEAN_RATIO
        sampled_gain = np.random.normal(mean_gain, std_gain)
        adjusted_gain = sampled_gain * self._get_not_adapted_learning_penalty(subject, learning_difficulty)
        self.skills_levels[subject] += max(adjusted_gain, 0)
        self.skills_levels[subject] = min(self.skills_levels[subject], 100)
        self.last_action['improvement'] = max(adjusted_gain, 0)
        self.cumulative_train_time += (learning_unit + 1)
        return 0 - (learning_unit + 1)

    def _get_not_adapted_learning_penalty(self, subject, learning_difficulty):
        proper_difficulty = self._get_proper_difficulty(subject)
        return NOT_ADAPTED_DIFFICULTY_PENALTY ** abs(learning_difficulty - proper_difficulty)

    def _get_proper_difficulty(self, subject):
        return sum(self.difficulties_thresholds <= self.skills_levels[subject]) - 1

    def reset(self):
        self.skills_levels = np.maximum(
            np.random.normal(MEAN_START_SKILL_LEVEL, STD_START_SKILL_LEVEL, size=len(self.skills_levels)),
            np.zeros_like(self.skills_levels)
        )
        self.last_scores = np.zeros_like(self.last_scores)
        self.mean_skill_gains = _get_mean_skills_gains(*self.mean_skill_gains.shape)
        self.difficulties_thresholds = np.linspace(0, 100, num=self.difficulties_levels, endpoint=False)
        self.cumulative_train_time = 0
        self.episode += 1
        return self.last_scores

    def render(self, mode='human'):
        action_to_str = ';'.join(f'{k}={v}' for k, v in self.last_action.items())
        print(f'***\n'
              f'Action: {action_to_str}\n'
              f'Test matrix: \n{self.last_scores.round(1)}\n'
              f'Latent skill level: {self.skills_levels.round(1)}\n'
              f'***')
        logging.info(json.dumps({**self.last_action, 'skills': self.skills_levels}, cls=reporting.NpEncoder))
        return self.last_action


def _get_mean_skills_gains(subjects_number, learning_units_number):
    interval_mean_skill_gains = np.maximum(
        np.random.normal(POPULATION_MEAN_SKILL_GAIN, POPULATION_STD_SKILL_GAIN, size=(subjects_number,
                                                                                      learning_units_number)),
        np.full(shape=(subjects_number, learning_units_number), fill_value=POPULATION_MIN_SKILL_GAIN)
    )
    return np.cumsum(interval_mean_skill_gains, axis=1)
