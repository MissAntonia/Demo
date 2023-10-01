import numpy as np


# Define a class to represent an optimization problem.
class OptimizationProblem:
    # Initialization method, taking in the objective function and its gradient (if provided).
    def __init__(self, objective, gradient=None):
        self.objective = objective

        self.gradient = gradient if gradient is not None else self._numerical_gradient

    # Evaluate the value of the objective function at a given point.
    def evaluate(self, x):
        return self.objective(x)

    def evaluate_gradient(self, x):
        return self.gradient(x)

    def _numerical_gradient(self, x, h=1e-6):
        """
      Approximate gradient of the objective function 
      """
        n = len(x)
        grad = [0] * n
        for i in range(n):
            x_minus_h = x.copy()
            x_plus_h = x.copy()
            x_minus_h[i] -= h
            x_plus_h[i] += h
            grad[i] = (self.objective(x_plus_h) - self.objective(x_minus_h)) / (2 * h)
        # return grad
        return np.array(grad)


# Define a general optimization method class.
class OptimizationMethod:
    # Initialization method, taking in an instance of the optimization problem.
    def __init__(self, problem):
        self.problem = problem

    # Placeholder method for the optimization algorithm. It should be implemented by subclasses.
    def optimize(self, initial_guess):
        raise NotImplementedError("This method should be implemented by derived classes.")


# Implement the classical Newton's method for optimization.
class NewtonMethod(OptimizationMethod):
    # Approximate the Hessian matrix using finite differences.
    def approx_hessian(self, x):
        n = len(x)  # 确定x的维度
        hessian = np.zeros((n, n))  # 初始化全0海森矩阵
        delta = 1e-5

        for i in range(n):  # 遍历每一个维度
            for j in range(n):
                x_inc_i = x.copy()
                x_inc_j = x.copy()
                x_inc_i_j = x.copy()

                x_inc_i[i] += delta
                x_inc_j[j] += delta
                x_inc_i_j[i] += delta
                x_inc_i_j[j] += delta

                # central finite difference formula
                hessian[i, j] = (self.problem.evaluate(x_inc_i_j) + self.problem.evaluate(x) -
                                 self.problem.evaluate(x_inc_i) - self.problem.evaluate(x_inc_j)) / (delta ** 2)

        # Ensure the Hessian is symmetric.
        G_bar = hessian
        G = 0.5 * (G_bar + G_bar.T)
        return G

    # Newton's method optimization routine.
    def optimize(self, initial_guess, max_iter=1000, tol=1e-6):
        x = initial_guess
        for _ in range(max_iter):
            gradient = self.problem.evaluate_gradient(x)
            hessian = self.approx_hessian(x)

            try:
                direction = np.linalg.solve(hessian, -gradient)
                # 求取线性方程组，获取下降方向
            except np.linalg.LinAlgError:
                # 海森矩阵奇异，没有逆矩阵
                print("Hessian is singular!")
                return x

            alpha = self.exact_line_search(x, direction)
            x = x + alpha * direction

            if np.linalg.norm(gradient) < tol:
                break

        return x


# Extend the NewtonMethod class to include an exact line search method.
class NewtonMethodWithLineSearch(NewtonMethod):
    # Implement a backtracking line search strategy.
    def exact_line_search(self, x, direction):
        alpha = 0.01
        while self.problem.evaluate(x + alpha * direction) > self.problem.evaluate(x):
            alpha *= 0.9
        return alpha


# #Test the performance of this method on the Rosenbrock function
def instantiate_optimization_problem():
    def rosenbrock(x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    def rosenbrock_gradient(x):
        dfdx0 = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
        dfdx1 = 200 * (x[1] - x[0] ** 2)
        return np.array([dfdx0, dfdx1])

    rosenbrock_problem = OptimizationProblem(rosenbrock, rosenbrock_gradient)
    return rosenbrock_problem


def objective_function(x):
    return x[0] ** 2 + 2 * x[1] ** 2


def gradrient_func(x):
    return np.array([2 * x[0], 4 * x[1]])
    return np.array([2, 4])


# test
if __name__ == "__main__":
    problem = OptimizationProblem(objective_function, gradrient_func)

    optimizer = NewtonMethodWithLineSearch(problem)

    initial_guess = np.array([1.0, 1.0])

    solution = optimizer.optimize(initial_guess)

    # Print the solution.
    print("Optimized point:", solution)
    print("Objective value at optimized point:", problem.evaluate(solution))
