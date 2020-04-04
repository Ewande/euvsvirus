import gym
from gym import spaces
import numpy as np


MEAN_START_SKILL_LEVEL = 20
STD_START_SKILL_LEVEL = 10


POPULATION_MEAN_SKILL_GAIN = 2
POPULATION_STD_SKILL_GAIN = 1
POPULATION_MIN_SKILL_GAIN = 0.2

STUDENT_SKILL_GAIN_STD_MEAN_RATIO = 0.5

TEST_SCORE_STD = 5

TARGET_SKILL_LEVEL = 90

REVIEW_RATIO = 0.25

REWARD_FOR_ACHIEVING_TARGET_LEVEL = 100
REWARD_FOR_ACHIEVING_ALL_LEVELS = 500

PENALTY_FOR_UNNECESSARY_TEST = -0.5 * REWARD_FOR_ACHIEVING_TARGET_LEVEL
TIME_PENALTY_FOR_TEST = - 5
GAIN_MULTIPLIER_FOR_TEST = 20


class StudentEnv(gym.Env):
    def __init__(self, subjects_number=4, difficulties_levels=3, learning_units_number=3):
        super(StudentEnv).__init__()
        self.action_space = spaces.MultiDiscrete([2, subjects_number, difficulties_levels, learning_units_number])
        self.observation_space = spaces.Box(low=0, high=100, shape=(subjects_number, difficulties_levels))
        self.difficulties_level_nb = difficulties_levels
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
        self.skills_level_achieved = np.zeros((subjects_number, 1))

    def step(self, action):
        assert self.action_space.contains(action)
        self.last_action = ('test' if action[0] else 'train') + f';subject={action[1] + 1};difficulty={action[2] + 1};'

        if action[0]:
            reward = self._test(action[1], action[2])
            if all(self.skills_level_achieved > 0):
                is_done = 1
                reward += REWARD_FOR_ACHIEVING_ALL_LEVELS
            else:
                is_done = 0
        else:
            reward = self._train(action[1], action[2])
            is_done = 0
        self.last_action += f';reward={reward}'
        return self.last_scores, reward, is_done, {}

    def _test(self, subject, difficulty):
        test_mean = self._get_test_mean(subject, difficulty)
        previous_score = self.last_scores[subject, difficulty]
        sampled_test_score = np.random.normal(test_mean, TEST_SCORE_STD)
        self.last_scores[subject, difficulty] = min(max(sampled_test_score, 0), 100)
        self.last_action += f'test score={self.last_scores[subject, difficulty]:.1f}'
        if not self.cumulative_train_time:
            return TIME_PENALTY_FOR_TEST
        if self.skills_levels[subject] > TARGET_SKILL_LEVEL:
            # if not self.skills_level_achieved[subject] and difficulty+1 == self.difficulties_level_nb:
            if not self.skills_level_achieved[subject]:
                self.skills_level_achieved[subject] = 1
                return REWARD_FOR_ACHIEVING_TARGET_LEVEL * (difficulty+1)/self.difficulties_level_nb
            else:
                return PENALTY_FOR_UNNECESSARY_TEST
        return GAIN_MULTIPLIER_FOR_TEST*((self.last_scores[subject, difficulty] - previous_score)
                                         / self.cumulative_train_time) + TIME_PENALTY_FOR_TEST

    def _get_test_mean(self, subject, difficulty):
        proper_difficulty = sum(self.difficulties_thresholds <= self.skills_levels[subject]) - 1
        if proper_difficulty < difficulty:
            return self._get_too_hard_test_mean(subject, difficulty, proper_difficulty)
        if proper_difficulty > difficulty:
            return 100
        return self._get_proper_test_mean(subject, difficulty)

    def _get_too_hard_test_mean(self, subject, difficulty, proper_difficulty):
        mean_scale = len(self.difficulties_thresholds)
        review_mean = (self.skills_levels[subject] - self.difficulties_thresholds[proper_difficulty]) * mean_scale
        return self.review_ratio ** (difficulty - proper_difficulty) * review_mean

    def _get_proper_test_mean(self, subject, difficulty):
        mean_scale = len(self.difficulties_thresholds)
        proper_mean = (self.skills_levels[subject] - self.difficulties_thresholds[difficulty]) * mean_scale
        review_score = self.review_ratio * 100 if difficulty else 0
        return review_score + proper_mean

    def _train(self, subject, learning_unit):
        mean_gain = self.mean_skill_gains[subject, learning_unit]
        std_gain = mean_gain * STUDENT_SKILL_GAIN_STD_MEAN_RATIO
        sampled_gain = np.random.normal(mean_gain, std_gain)
        self.skills_levels[subject] += max(sampled_gain, 0)
        self.skills_levels[subject] = min(self.skills_levels[subject], 100)
        self.last_action += f'improvement={max(sampled_gain, 0):.1f}'
        self.cumulative_train_time += (learning_unit + 1)
        return 0 - (learning_unit + 1)

    def reset(self):
        self.skills_levels = np.maximum(
            np.random.normal(MEAN_START_SKILL_LEVEL, STD_START_SKILL_LEVEL, size=len(self.skills_levels)),
            np.zeros_like(self.skills_levels)
        )
        self.last_scores = np.zeros_like(self.last_scores)
        self.mean_skill_gains = _get_mean_skills_gains(*self.mean_skill_gains.shape)
        self.difficulties_thresholds = np.linspace(0, 100, num=len(self.difficulties_thresholds), endpoint=False)
        self.cumulative_train_time = 0
        self.episode += 1
        self.skills_level_achieved = np.zeros(self.skills_level_achieved.shape)
        return self.last_scores

    def render(self, mode='human'):
        print(f'***\n'
              f'Action: {self.last_action}\n'
              f'Test matrix: \n{self.last_scores.round(1)}\n'
              f'Latent skill level: {self.skills_levels.round(1)}\n'
              f'***')
        return self.last_action


def _get_mean_skills_gains(subjects_number, learning_units_number):
    interval_mean_skill_gains = np.maximum(
        np.random.normal(POPULATION_MEAN_SKILL_GAIN, POPULATION_STD_SKILL_GAIN, size=(subjects_number,
                                                                                      learning_units_number)),
        np.full(shape=(subjects_number, learning_units_number), fill_value=POPULATION_MIN_SKILL_GAIN)
    )
    return np.cumsum(interval_mean_skill_gains, axis=0)
