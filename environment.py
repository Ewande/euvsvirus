import gym
import numpy as np
from gym import spaces


POPULATION_MEAN_SKILL_GAIN = 2
POPULATION_STD_SKILL_GAIN = 1
POPULATION_MIN_SKILL_GAIN = 0.2

STUDENT_SKILL_GAIN_STD_MEAN_RATIO = 0.5


class StudentEnv(gym.Env):
    def __init__(self, subjects_number=4, difficulties_levels=3, learning_units_number=3):
        super(StudentEnv).__init__()
        self.action_space = spaces.Dict({
            'is_test': spaces.Discrete(2),
            'subject': spaces.Discrete(subjects_number),
            'learning_unit': spaces.Discrete(learning_units_number)
        })
        self.observation_space = spaces.Box(low=0, high=100, shape=(subjects_number, difficulties_levels))
        self.skills_levels = np.random.random(subjects_number)
        self.last_scores = np.zeros(subjects_number, difficulties_levels)
        self.mean_skill_gains = _get_mean_skills_gains(subjects_number, learning_units_number)

    def step(self, action):
        assert self.action_space.contains(action)
        if action['is_test']:
            self._test(action['subject'])
        else:
            self._train(action['subject'], action['learning_unit'])
        return self.last_scores

    def _test(self, subject):
        pass

    def _train(self, subject, learning_unit):
        mean_gain = self.mean_skill_gains[subject, learning_unit]
        std_gain = mean_gain * STUDENT_SKILL_GAIN_STD_MEAN_RATIO
        self.skills_levels[subject] += max(np.random.normal(mean_gain, std_gain), 0)

    def reset(self):
        self.skills_levels = np.random.random(len(self.skills_levels))
        self.last_scores = np.zeros_like(self.last_scores)
        return self.last_scores

    def render(self, mode='human'):
        pass


def _get_mean_skills_gains(subjects_number, learning_units_number):
    interval_mean_skill_gains = np.maximum(
        np.random.normal(POPULATION_MEAN_SKILL_GAIN, POPULATION_STD_SKILL_GAIN, size=(subjects_number, learning_units_number)),
        np.full(shape=(subjects_number, learning_units_number), fill_value=POPULATION_MIN_SKILL_GAIN)
    )
    return np.cumsum(interval_mean_skill_gains, axis=0)
