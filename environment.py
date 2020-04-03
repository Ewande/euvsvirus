import gym
import numpy as np
from gym import spaces


MEAN_START_SKILL_LEVEL = 20
STD_START_SKILL_LEVEL = 10


POPULATION_MEAN_SKILL_GAIN = 2
POPULATION_STD_SKILL_GAIN = 1
POPULATION_MIN_SKILL_GAIN = 0.2

STUDENT_SKILL_GAIN_STD_MEAN_RATIO = 0.5

TEST_SCORE_STD_MEAN_RATIO = 0.05

TARGET_SKILL_LEVEL = 90


class StudentEnv(gym.Env):
    def __init__(self, subjects_number=4, difficulties_levels=3, learning_units_number=3):
        super(StudentEnv).__init__()
        self.action_space = spaces.Dict({
            'is_test': spaces.Discrete(2),
            'subject': spaces.Discrete(subjects_number),
            'test_difficulty': spaces.Discrete(difficulties_levels),
            'learning_unit': spaces.Discrete(learning_units_number)
        })
        self.observation_space = spaces.Box(low=0, high=100, shape=(subjects_number, difficulties_levels))
        self.skills_levels = np.maximum(
            np.random.normal(MEAN_START_SKILL_LEVEL, STD_START_SKILL_LEVEL, size=subjects_number),
            np.zeros(subjects_number)
        )
        self.last_scores = np.zeros(shape=(subjects_number, difficulties_levels))
        self.mean_skill_gains = _get_mean_skills_gains(subjects_number, learning_units_number)
        self.difficulties_thresholds = np.linspace(0, 100, num=difficulties_levels, endpoint=False)
        self.cumulative_train_time = 0

    def step(self, action):
        assert self.action_space.contains(action)
        if action['is_test']:
            reward = self._test(action['subject'], action['test_difficulty'])
        else:
            reward = self._train(action['subject'], action['learning_unit'])
        is_done = all(self.skills_levels > TARGET_SKILL_LEVEL)
        return self.last_scores, reward, is_done, None

    def _test(self, subject, difficulty):
        skill_level = self.skills_levels[subject]
        proper_difficulty = sum(self.difficulties_thresholds <= skill_level) - 1
        test_mean = self._get_test_mean(difficulty, proper_difficulty, subject)
        test_std = test_mean * TEST_SCORE_STD_MEAN_RATIO
        previous_scores = self.last_scores.copy()
        self.last_scores[subject, difficulty] = min(max(np.random.normal(test_mean, test_std), 0), 100)
        if not self.cumulative_train_time:
            return 0
        return sum(sum(self.last_scores - previous_scores)) / self.cumulative_train_time

    def _get_test_mean(self, difficulty, proper_difficulty, subject):
        if proper_difficulty < difficulty:
            return self._get_too_hard_test_mean(difficulty, subject)
        if proper_difficulty > difficulty:
            return 100
        return self._get_proper_test_mean(subject, difficulty)

    def _get_too_hard_test_mean(self, difficulty, subject):
        penalty = 1 - (self.difficulties_thresholds[difficulty] - self.skills_levels[subject]) / 100
        return self.skills_levels[subject] * penalty

    def _get_proper_test_mean(self, subject, difficulty):
        mean_scale = len(self.difficulties_thresholds)
        mean = (self.skills_levels[subject] - self.difficulties_thresholds[difficulty]) * mean_scale
        return mean

    def _train(self, subject, learning_unit):
        mean_gain = self.mean_skill_gains[subject, learning_unit]
        std_gain = mean_gain * STUDENT_SKILL_GAIN_STD_MEAN_RATIO
        self.skills_levels[subject] += max(np.random.normal(mean_gain, std_gain), 0)
        self.cumulative_train_time += (learning_unit + 1)
        return 0

    def reset(self):
        self.skills_levels = np.maximum(
            np.random.normal(MEAN_START_SKILL_LEVEL, STD_START_SKILL_LEVEL, size=len(self.skills_levels)),
            np.zeros_like(self.skills_levels)
        )
        self.last_scores = np.zeros_like(self.last_scores)
        self.mean_skill_gains = _get_mean_skills_gains(*self.mean_skill_gains.shape)
        self.difficulties_thresholds = np.linspace(0, 100, num=len(self.difficulties_thresholds), endpoint=False)
        self.cumulative_train_time = 0
        return self.last_scores

    def render(self, mode='human'):
        pass


def _get_mean_skills_gains(subjects_number, learning_units_number):
    interval_mean_skill_gains = np.maximum(
        np.random.normal(POPULATION_MEAN_SKILL_GAIN, POPULATION_STD_SKILL_GAIN, size=(subjects_number,
                                                                                      learning_units_number)),
        np.full(shape=(subjects_number, learning_units_number), fill_value=POPULATION_MIN_SKILL_GAIN)
    )
    return np.cumsum(interval_mean_skill_gains, axis=0)
