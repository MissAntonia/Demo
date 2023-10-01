import project2
import numpy as np


class BroydenUpdate(project2.OptimizationMethod):
    def __init__(self, problem, initial, max_iter=100, tol=1e-6):
        super().__init__(problem)
        self.x = initial
        self.max_iter = max_iter
        self.tol = tol
        self.H = np.eye(len(initial))  # Hessian initialization

    def hessian_updater(self, delta, gamma):
        rho = 1 / np.dot(gamma, delta)
        A = np.eye(len(delta)) - rho * np.outer(delta, gamma)
        B = np.eye(len(delta)) - rho * np.outer(gamma, delta)
        W = np.outer(delta, delta)
        self.H = np.dot(np.dot(A, self.H), B) + rho * W

    def solve(self):
        for i in range(self.max_iter):
            gradient = self.problem.gradient(self.x)
            direction = -np.dot(np.linalg.inv(self.H), gradient)
            step_size = self.exact_line_search(self.x, direction)
            self.x = self.x + step_size * direction
            new_gradient = self.problem.gradient(self.x)
            s = step_size * direction
            y = new_gradient - gradient
            self.hessian_updater(s, y)

            if np.linalg.norm(gradient) < self.tol:
                break

        return self.x, i

    def exact_line_search(self, x, direction):
        alpha = 0.1
        while self.problem.evaluate(x + alpha * direction) > self.problem.evaluate(x):
            alpha *= 0.5
        return alpha


# Example usage:
def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def rosenbrock_gradient(x):
    dfdx0 = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    dfdx1 = 200 * (x[1] - x[0] ** 2)
    return np.array([dfdx0, dfdx1])

def objective_function(x):
    return x[0] ** 2 + 2 * x[1] ** 2


def gradrient_func(x):
    return np.array([2 * x[0], 4 * x[1]])
    return np.array([2, 4])

initial_guess = np.array([1, 1.1])
prob = project2.OptimizationProblem(objective_function, gradrient_func)
newton_solver = BroydenUpdate(prob, initial_guess)
result, itr = newton_solver.solve()

print("Optimal solution:", result)
print("Objective value at optimized point:", prob.evaluate(result))
print(f"Completed in {itr+1} iterations")
