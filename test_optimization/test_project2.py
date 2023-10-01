import unittest

import numpy as np

from project2.project2 import OptimizationProblem
from project2.project2 import NewtonMethodWithLineSearch


class TestOptimizationProblem(unittest.TestCase):
    def setUp(self) -> None:
        def objective(x):
            return x[0] ** 2 + x[1] ** 2

        def gradient(x):
            return np.array([2 * x[0], 2 * x[1]])

        self.problem_with_gradient = OptimizationProblem(objective, gradient)
        self.problem_without_gradient = OptimizationProblem(objective)

    def test_problem_initialization_with_gradient(self):
        self.assertEqual(self.problem_with_gradient.objective([1, 2]), 5)
        np.testing.assert_array_equal(self.problem_with_gradient.gradient([1, 2]), [2, 4])

    def test_problem_initialization_without_gradient(self):
        self.assertEqual(self.problem_without_gradient.objective([1, 2]), 5)

    def test_gradient_eval(self):
        np.testing.assert_array_equal(self.problem_with_gradient.evaluate_gradient([1, 2]), [2, 4])

    def test_numerical_gradient(self):
        np.testing.assert_allclose(self.problem_without_gradient._numerical_gradient([1, 2]), np.array([2, 4]),
                                   atol=1e-6)

    def test_newtonmethod_with_line(self):
        test_optimizer = NewtonMethodWithLineSearch(self.problem_without_gradient)
        test_solution = test_optimizer.optimize(np.array([1.0, 1.0]))
        self.assertTrue(np.allclose(self.problem_without_gradient.evaluate(test_solution), [0, 0], atol=1e-6))

    def test_newtonmethod_with_line_and_grad(self):
        test_optimizer = NewtonMethodWithLineSearch(self.problem_with_gradient)
        test_solution = test_optimizer.optimize(np.array([1.0, 1.0]))
        self.assertTrue(np.allclose(self.problem_with_gradient.evaluate(test_solution), [0, 0], atol=1e-6))


if __name__ == '__main__':
    unittest.main()
