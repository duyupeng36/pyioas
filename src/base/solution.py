"""

"""

__all__ = ['Solution']

class Solution:
    def __init__(self):
        self.problem = None
        self.start_time = None
        self.end_time = None
        self.best_position = None
        self.best_fitness = float("inf")
        self.iteration_curve = None

    def __str__(self):
        return f"""
        Solution<problem={self.problem.name}, 
                 best_position={self.best_position}, 
                 best_fitness={self.best_fitness}, 
                 start_time={self.start_time},
                 end_time={self.end_time}>
        """

    def __repr__(self):
        return self.__str__()
