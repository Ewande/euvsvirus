import numpy as np

import environment
import settings


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def predict(self, observation):
        # for compatibility with stable_baselines.common.BaseRLModel.predict
        _ = None
        return self.env.action_space.sample(), _


class SimpleAgent:
    """
    Chooses action for `StudentEnv` in simple static manner - trains subject on
    lowest level for given number of units (consistent difficulty, random type),
    then tests.
    """
    def __init__(self, env, max_sequence_of_trainigs=10):
        self.num_subjects = env.num_subjects
        self.num_difficulty_levels = env.num_difficulty_levels
        self.num_train_types = env.num_train_types
        self.max_sequence_of_trainings = max_sequence_of_trainigs
        self.trainings_counter = 0

    def predict(self, observation):
        observed_state = self._get_observed_state(observation)
        subject, difficulty = self._get_lowest_level_subject(
            observed_state.last_test_scores)

        if self.trainings_counter < self.max_sequence_of_trainings:
            self.trainings_counter += 1
            action = self._get_training_action(subject, difficulty)
        else:
            self.trainings_counter = 0
            action = _get_test_action(subject, difficulty)

        # for compatibility with stable_baselines.common.BaseRLModel.predict
        _ = None
        return action, _

    def _get_observed_state(self, observation):
        return environment.ObservedState.from_observation(
            observation,
            self.num_subjects, self.num_difficulty_levels, self.num_train_types
        )

    def _get_lowest_level_subject(self, last_test_scores):
        for difficulty, scores in enumerate(last_test_scores.T):
            for subject, score in enumerate(scores):
                if score < settings.TARGET_SCORE:
                    return subject, difficulty
        return self.num_subjects - 1, self.num_difficulty_levels - 1

    def _get_training_action(self, subject, difficulty):
        train_type = np.random.randint(self.num_train_types)
        return np.array([
            environment.ActionType.TRAIN.value,
            subject,
            0,  # not used
            train_type,
            difficulty
        ])


def _get_test_action(subject, difficulty):
    return np.array([
        environment.ActionType.TEST.value,
        subject,
        difficulty,
        0,  # not used
        0,  # not used
    ])

