import unittest
from numpy.testing import assert_almost_equal, assert_array_equal
import numpy as np
import os

from environment import StudentEnv
from Settings import TARGET_SCORE


class TestStudentEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.weak_student = StudentEnv(num_subjects=2)
        self.weak_student.skill_levels = np.array([4, 5])
        self.basic_student = StudentEnv(num_subjects=2)
        self.basic_student.skill_levels = np.array([32, 31])
        self.medium_student = StudentEnv(num_subjects=2)
        self.medium_student.skill_levels = np.array([34, 35])
        self.strong_student = StudentEnv(num_subjects=2)
        self.strong_student.skill_levels = np.array([64, 65])
        self.advanced_student = StudentEnv(num_subjects=2)
        self.advanced_student.skill_levels = np.array([80, 90])
        self.master_student = StudentEnv(num_subjects=2)
        self.master_student.skill_levels = np.array([TARGET_SCORE + 1, TARGET_SCORE + 2])
        self.student_pack = [self.weak_student, self.basic_student, self.medium_student, self.strong_student,
                             self.advanced_student, self.master_student]
        self.four_subject_student = StudentEnv(num_subjects=4)
        self.four_level_student = StudentEnv(num_difficulty_levels=4)

    def test_length(self):
        self.assertEqual(len(self.weak_student.skill_levels), 2)
        self.assertEqual(len(self.four_subject_student.skill_levels), 4)
        self.assertEqual(self.four_level_student.num_difficulty_levels, 4)
        assert_array_equal(self.four_level_student.difficulty_thresholds, np.array([0, 25, 50, 75]))

    def test_get_proper_difficulty(self):
        for idx, student in enumerate(self.student_pack):
            self.assertEqual(student._get_proper_difficulty(student.skill_levels[1]), np.floor(idx/2))

if __name__ == '__main__':
    unittest.main()
