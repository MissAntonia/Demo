import numpy as np
import project2


class QuasiNewton(project2.OptimizationMethod):
    def __init__(self, problem, initial_guess,max_iterations=100, tol=1e-6, alg='bad'):
        """
        This class perfroms both Bad Broyden and Symmetric Broyden update

        :param problem: problem class from project2.py
            Objective problem used for inherited Optimization method class
        :param initial_guess: np array
        :param max_iterations: int
        :param tol: float
            tolerance value
        :param alg: str
            Use 'bad' for Simple Broyden rank-1 update and 'sym' for symmetric Broyden update
        """
        super().__init__(problem)
        self.initial = initial_guess
        self.hessian_inv = None
        self.max_itr = max_iterations
        self.tol = tol
        self.algtm = alg

    def solve(self):
        x = initial_guess
        n = len(x)

        for iter in range(self.max_itr):
            gradient = self.problem.gradient(x)
            if self.hessian_inv is None:
                self.hessian_inv = np.eye(n)  # Initialize Hessian inverse as identity matrix
            direction = -np.dot(self.hessian_inv, gradient)
            step_size = self.exact_line_search(x, direction)
            x_new = x + step_size * direction
            delta = x_new - x
            gamma = self.problem.gradient(x_new) - gradient
            x = x_new
            self.hessian_updater(delta,gamma,n,x_new)
            if np.linalg.norm(gradient) < self.tol:
                break

        return x, iter

    def hessian_updater(self,delta,gamma,n,x_new):
        if self.algtm == 'bad':
            # Broyden's rank-1 update to approximate the Hessian inverse
            rho = 1.0 / np.dot(gamma, delta)
            self.hessian_inv = (np.eye(n) - rho * np.outer(delta, gamma)) @ self.hessian_inv @ (
                        np.eye(n) - rho * np.outer(gamma, delta)) + rho * np.outer(delta, delta)

        elif self.algtm == 'sym':
            delta_gamma_T = np.outer(delta, gamma).T
            delta_delta_T = np.outer(delta, delta).T
            gamma_T_gamma = np.dot(gamma, gamma)

            self.hessian_inv = self.hessian_inv + (1.0 / gamma_T_gamma) * (
                        delta_gamma_T - np.dot(self.hessian_inv, gamma) @ delta_gamma_T) + (
                                       1.0 / gamma_T_gamma ** 2) * (delta_gamma_T @ np.dot(self.hessian_inv,
                                                                                           gamma) @ delta_gamma_T - gamma_T_gamma * delta_delta_T)

        elif self.algtm == 'dfp':
            w1 = delta
            w2 = np.dot(self.hessian_inv,gamma)
            w1T = w1.T
            w2T = w2.T
            sigma1 = 1 / (w1T.dot(gamma))  # See line 20 of the algorithm
            sigma2 = -1 / (w2T.dot(gamma))  # See line 21 of the algorithm
            W1 = np.outer(w1, w1)
            W2 = np.outer(w2, w2)
            Delta = sigma1 * W1 + sigma2 * W2  # See line 22 of the algorithm
            self.hessian_inv += Delta  # See line 23 of the algorithm

        elif self.algtm == 'bfgs2':
            den = delta.dot(gamma)
            num = np.dot(self.hessian_inv,gamma)
            term1 = 1 + np.dot(gamma, num) / den
            term2 = np.outer(delta, delta) / den
            term3 = np.outer(delta, num) / den
            term4 = np.outer(num, delta) / den

            Delta = term1*term2 -term3 -term4
            self.hessian_inv += Delta
        else:
            raise Exception("Unknown algorithm requested")

    def exact_line_search(self, x, direction):
        alpha = 0.01
        while self.problem.evaluate(x + alpha * direction) > self.problem.evaluate(x):
            alpha *= 0.5
        return alpha

# Example usage:
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

initial_guess = np.array([1.0, 2.0])
obj_fnc =project2.OptimizationProblem(rosenbrock)
broyden_newton = QuasiNewton(obj_fnc, initial_guess, 10000, alg='dfp')
result,itr = broyden_newton.solve()
print("Optimal solution:", result)
print("Objective value at optimized point:", obj_fnc.evaluate(result))
print(f"Completed in {itr+1} iterations")
def objective_function(x):
    return x[0] ** 2 + 2 * x[1] ** 2


def gradrient_func(x):
    return np.array([2 * x[0], 4 * x[1]])

initialguess = np.array([1, 2])
prob = project2.OptimizationProblem(objective_function, gradrient_func)
newton_solver = QuasiNewton(prob, initialguess, 10000, alg='bfgs2')
result, itr = newton_solver.solve()

print("Optimal solution:", result)
print("Objective value at optimized point:", prob.evaluate(result))
print(f"Completed in {itr+1} iterations")