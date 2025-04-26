from causallearn.utils.cit import CIT_Base, register_ci_test
from causallearn.score.LocalScoreFunction import local_score_BIC
import numpy as np

def register_ci_tests():
    register_ci_test("gaussbic", BIC_CIT)

class BIC_CIT(CIT_Base):
    def __init__(self, data, penalty_discount: float = None, **kwargs):
        """
        Linear Gaussian BIC based CIT using causal-learn's score function.
        Inspired by Ramsey et al. 2018: https://arxiv.org/pdf/1805.03108
        """
        super().__init__(data, **kwargs)
        self.data = data
        if penalty_discount is None:
            self.penalty_discount = 1.0
        else:
            self.penalty_discount = penalty_discount

    def __call__(self, X, Y, condition_set=None):
        """
        Perform an independence test.

        Parameters
        ----------
        X, Y: column indices of data
        condition_set: conditioning variables, default None

        Returns
        -------
        b: the difference of bic value without X and with X where a positive score indicates independence
        """
        cond_set = list(condition_set)
        score_without_X = local_score_BIC(self.data, Y, cond_set, parameters={"lambda_value": self.penalty_discount})
        score_with_X = local_score_BIC(self.data, Y, cond_set + [X], parameters={"lambda_value": self.penalty_discount})

        # ensure that scores are scalars and not ndarrays
        if isinstance(score_without_X, np.ndarray):
            score_without_X = score_without_X.item()
        if isinstance(score_with_X, np.ndarray):
            score_with_X = score_with_X.item()

        return score_without_X - score_with_X

    def is_ci(self, X, Y, condition_set=None):
        return self.__call__(X, Y, condition_set) >= 0
