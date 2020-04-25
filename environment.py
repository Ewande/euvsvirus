import json
import logging

import gym
from gym import spaces
import numpy as np

import reporting
from utils import estimate_skills
import sys

MEAN_START_SKILL_LEVEL = 20
STD_START_SKILL_LEVEL = 10

POPULATION_MEAN_SKILL_GAIN = 3
POPULATION_STD_SKILL_GAIN = 2
POPULATION_STD_TYPE_GAIN = 0.2
POPULATION_MIN_SKILL_GAIN = 0.2

STUDENT_SKILL_GAIN_STD = 1

TEST_SCORE_STD = 4

TARGET_SCORE = 90

REVIEW_RATIO = 0.25

REWARD_FOR_ACHIEVING_TARGET_LEVEL = 100
REWARD_FOR_ACHIEVING_ALL_LEVELS = 1000

PENALTY_FOR_UNNECESSARY_TEST = -500
TIME_PENALTY_FOR_TEST = - 5
GAIN_MULTIPLIER_FOR_TEST = 0

NOT_ADAPTED_DIFFICULTY_PENALTY = 0.25

GAIN_REWARD_RATIO = 0.1


class StudentEnv(gym.Env):
    def __init__(self, subjects_number=3, difficulties_levels=3, learning_type_number=3):
        super(StudentEnv).__init__()
        self.action_space = spaces.MultiDiscrete([2, subjects_number, difficulties_levels,
                                                  learning_type_number, difficulties_levels])
        high_bound_observation_space_vector = np.array([100, 100, *[sys.maxsize] * learning_type_number,
                                                        *[100] * learning_type_number])
        self.observation_space = spaces.Box(
            low=np.zeros((subjects_number, difficulties_levels, 2 * learning_type_number + 2))
            ,
            high=np.tile(high_bound_observation_space_vector[None, None, :], (subjects_number, difficulties_levels, 1))
            # ,shape=(subjects_number, difficulties_levels)
        )
        self.difficulties_levels = difficulties_levels
        self.learning_type_number = learning_type_number
        self.skills_levels = np.maximum(
            np.random.normal(MEAN_START_SKILL_LEVEL, STD_START_SKILL_LEVEL, size=subjects_number), 0
        )
        self.last_scores = np.zeros(shape=(subjects_number, difficulties_levels, 2 * learning_type_number + 2))
        self.mean_skill_gains = _get_mean_skills_gains(subjects_number, learning_type_number)
        self.difficulties_thresholds = np.linspace(0, 100, num=difficulties_levels, endpoint=False)
        self.review_ratio = 1 / (difficulties_levels + 1)
        self.cumulative_train_time = np.zeros(subjects_number)
        self.train_counter = np.zeros((subjects_number, difficulties_levels, learning_type_number))
        self.episode = 0
        self.step_num = 0
        self.last_action = None

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
            reward = self._train(subject, learning_types, learning_difficulty)
            is_done = 0
        reward += - np.sqrt(self.step_num)
        self.last_action['reward'] = reward
        self.step_num += 1
        return self.last_scores, reward, is_done, {}

    def _test(self, subject, difficulty):
        test_mean = self._get_test_mean(subject, difficulty)
        previous_score = self.last_scores[subject, difficulty, 0]
        sampled_test_score = min(max(np.random.normal(test_mean, TEST_SCORE_STD), 0), 100)
        self.last_scores[subject, difficulty, 1] = sampled_test_score - previous_score
        self.last_scores[subject, difficulty, 0] = sampled_test_score
        self.last_scores[subject, difficulty, -self.learning_type_number:] = self._get_mean_type_gain(subject,
                                                                                                      difficulty)
        self.last_scores[subject, difficulty, 2:2 + self.learning_type_number] = 0
        self.last_action['test_score'] = self.last_scores[subject, difficulty, 0]
        if not self.cumulative_train_time[subject]:
            return TIME_PENALTY_FOR_TEST
        if self.last_scores[subject, difficulty, 0] >= TARGET_SCORE:
            # if not self.skills_level_achieved[subject] and difficulty+1 == self.difficulties_level_nb:
            if previous_score < TARGET_SCORE:
                return REWARD_FOR_ACHIEVING_TARGET_LEVEL * (difficulty + 1) / self.difficulties_levels
            else:
                return PENALTY_FOR_UNNECESSARY_TEST
        return GAIN_MULTIPLIER_FOR_TEST * ((self.last_scores[subject, difficulty, 0] - previous_score)
                                           / self.cumulative_train_time[subject]) + TIME_PENALTY_FOR_TEST

    def _get_mean_type_gain(self, subject, difficulty):
        num_trainings_since_last_test = self.last_scores[subject, difficulty, 2:2 + self.learning_type_number]
        if np.sum(num_trainings_since_last_test) > 0:
            ratio = num_trainings_since_last_test / np.sum(num_trainings_since_last_test)
            new_gain = ratio * self.last_scores[subject, difficulty, 1]
            result = -np.ones(self.learning_type_number)
            for idx, elem in enumerate(zip(self.last_scores[subject, difficulty, -self.learning_type_number:], new_gain)):
                last_avg, new_avg = elem
                if self.train_counter[subject, difficulty, idx] == 0:
                    result[idx] = 0
                else:
                    result[idx] = np.average([last_avg, new_avg],
                                             weights=[self.train_counter[subject, difficulty, idx],
                                                      num_trainings_since_last_test[idx]])
            self.train_counter[subject, difficulty, :] += num_trainings_since_last_test
            return result
        else:
            return self.last_scores[subject, difficulty, -self.learning_type_number:]

    def _get_test_mean(self, subject, difficulty):
        proper_difficulty = self._get_proper_difficulty(self.skills_levels[subject])
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

    def _train(self, subject, learning_type, learning_difficulty):
        mean_gain = self.mean_skill_gains[subject, learning_type]
        sampled_gain = np.random.normal(mean_gain, STUDENT_SKILL_GAIN_STD)
        adjusted_gain = sampled_gain * self._get_not_adapted_learning_penalty(
            self.skills_levels[subject], learning_difficulty)
        adjusted_gain = max(adjusted_gain, 0)
        self.skills_levels[subject] += adjusted_gain
        self.skills_levels[subject] = min(self.skills_levels[subject], 100)
        self.last_action['improvement'] = max(adjusted_gain, 0)
        self.last_action['learning_type'] = learning_type + 1
        self.cumulative_train_time[subject] += (learning_type + 1)
        self.last_scores[subject, learning_difficulty, 2 + learning_type] += 1
        estimated_skill = estimate_skills(self.last_scores[:, :, 0], REVIEW_RATIO)[subject]
        estimated_penalty = self._get_not_adapted_learning_penalty(estimated_skill, learning_difficulty)
        estimated_gain = POPULATION_MEAN_SKILL_GAIN * learning_type
        adapted_learning_reward = estimated_penalty * estimated_gain * GAIN_REWARD_RATIO
        return 0
        # return 0 - (learning_type + 1) + adapted_learning_reward

    def _get_not_adapted_learning_penalty(self, skill, learning_difficulty):
        proper_difficulty = self._get_proper_difficulty(skill)
        return NOT_ADAPTED_DIFFICULTY_PENALTY ** abs(learning_difficulty - proper_difficulty)

    def _get_proper_difficulty(self, skill):
        return sum(self.difficulties_thresholds <= skill) - 1

    def reset(self):
        self.skills_levels = np.maximum(
            np.random.normal(MEAN_START_SKILL_LEVEL, STD_START_SKILL_LEVEL, size=len(self.skills_levels)),
            np.zeros_like(self.skills_levels)
        )
        self.last_scores = np.zeros_like(self.last_scores)
        self.mean_skill_gains = _get_mean_skills_gains(*self.mean_skill_gains.shape)
        self.difficulties_thresholds = np.linspace(0, 100, num=self.difficulties_levels, endpoint=False)
        self.cumulative_train_time = np.zeros_like(self.cumulative_train_time)
        self.train_counter = np.zeros_like(self.train_counter)
        self.episode += 1
        self.step_num = 0
        return self.last_scores

    def render(self, mode='human'):
        action_to_str = ';'.join(f'{k}={v}' for k, v in self.last_action.items())
        print(f'***\n'
              f'Action: {action_to_str}\n'
              f'Test matrix: \n{self.last_scores.round(1)}\n'
              f'Latent skill level: {self.skills_levels.round(1)}\n'
              f'***')
        logging.info(json.dumps({**self.last_action, 'skills': self.skills_levels, 'step': self.step_num,
                                 'episode': self.episode},
                                cls=reporting.NpEncoder))
        return self.last_action


def _get_mean_skills_gains(subjects_number, learning_types_number):
    skill_gain_matrix = np.tile(np.random.normal(POPULATION_MEAN_SKILL_GAIN, POPULATION_STD_SKILL_GAIN,
                                                 size=(learning_types_number)), (subjects_number, 1))
    skill_gain_matrix += np.random.normal(0, POPULATION_STD_TYPE_GAIN, size=(subjects_number, learning_types_number))
    interval_mean_skill_gains = np.maximum(
        skill_gain_matrix,
        np.full(shape=(subjects_number, learning_types_number), fill_value=POPULATION_MIN_SKILL_GAIN)
    )
    return interval_mean_skill_gains
