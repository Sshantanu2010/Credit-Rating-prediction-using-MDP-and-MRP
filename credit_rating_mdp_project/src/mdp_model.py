"""
mdp_model.py — Markov Decision Process for exposure / capital allocation.

Extends the MRP to a controlled decision setting:

  State  : (credit_rating, exposure_level)
           credit_rating ∈ {AAA, AA, A, BBB, BB, B, C, D}
           exposure_level ∈ {Low, Medium, High}

  Action : {Increase, Hold, Reduce, Exit}

  Reward : coupon_income − expected_default_loss − capital_penalty
           scaled by an exposure multiplier.

  Solved via  Value Iteration  and  Policy Iteration  (from scratch).
"""

import numpy as np
import pandas as pd
from itertools import product
from src.utils import BUCKETED_RATINGS, EXPOSURE_LEVELS, MDP_ACTIONS


# ═══════════════════════════════════════════════════════════════
# REWARD PARAMETERS
# ═══════════════════════════════════════════════════════════════

# Coupon income per unit exposure by rating (₹ Cr equivalent)
COUPON_INCOME = {
    'AAA': 7.5, 'AA': 8.0, 'A': 8.5, 'BBB': 9.5,
    'BB': 11.0, 'B': 13.0, 'C': 15.0, 'D': 0.0,
}

# Expected default loss rate (probability × loss-given-default)
DEFAULT_LOSS_RATE = {
    'AAA': 0.001, 'AA': 0.005, 'A': 0.015, 'BBB': 0.04,
    'BB': 0.10,  'B': 0.20,  'C': 0.40,  'D': 1.0,
}

# Capital / risk charge per unit exposure
CAPITAL_CHARGE = {
    'AAA': 0.5, 'AA': 1.0, 'A': 2.0, 'BBB': 4.0,
    'BB': 8.0,  'B': 12.0, 'C': 18.0, 'D': 25.0,
}

# Exposure multiplier
EXPOSURE_MULTIPLIER = {'Low': 0.5, 'Medium': 1.0, 'High': 2.0}

# Action-to-exposure transition (deterministic component)
ACTION_EXPOSURE_MAP = {
    'Increase': {'Low': 'Medium', 'Medium': 'High', 'High': 'High'},
    'Hold':     {'Low': 'Low',    'Medium': 'Medium', 'High': 'High'},
    'Reduce':   {'Low': 'Low',    'Medium': 'Low',    'High': 'Medium'},
    'Exit':     {'Low': 'Low',    'Medium': 'Low',    'High': 'Low'},
    # Exit always results in Low (minimal residual)
}

# Loss-given-default for the D state
LGD = 0.60  # 60 % loss on default


class MDP:
    """
    Markov Decision Process for capital-allocation optimisation.

    Parameters
    ----------
    credit_transition : np.ndarray
        8×8 rating transition matrix (same as MRP).
    gamma : float
        Discount factor.
    """

    def __init__(self, credit_transition: np.ndarray,
                 gamma: float = 0.95,
                 rating_states=None, exposure_states=None,
                 actions=None):

        self.ratings   = rating_states  or list(BUCKETED_RATINGS)
        self.exposures = exposure_states or list(EXPOSURE_LEVELS)
        self.actions   = actions or list(MDP_ACTIONS)
        self.gamma     = gamma

        # Credit transition matrix (only the rating dimension)
        self.P_credit = np.array(credit_transition, dtype=float)
        n = self.P_credit.shape[0]
        rs = self.P_credit.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        self.P_credit = self.P_credit / rs

        # Build composite state space
        self.states = list(product(self.ratings, self.exposures))
        self.n_states = len(self.states)
        self.state_idx = {s: i for i, s in enumerate(self.states)}

        # Pre-compute transition & reward tensors
        self._build_models()

    # ───────────────────────────────────────────────────────────
    def _build_models(self):
        """Build T[a] and R[a] for every action."""
        nS = self.n_states
        nA = len(self.actions)

        # T[a][s, s'] = Pr(s' | s, a)
        self.T = np.zeros((nA, nS, nS))
        # R[a][s]     = immediate reward
        self.R = np.zeros((nA, nS))

        for ai, action in enumerate(self.actions):
            for si, (rating, exposure) in enumerate(self.states):
                # Determine next exposure level (deterministic)
                next_exp = ACTION_EXPOSURE_MAP[action][exposure]

                # Reward = coupon × multiplier − default_loss × multiplier × LGD − capital_charge × multiplier
                mult = EXPOSURE_MULTIPLIER[exposure]
                coupon = COUPON_INCOME.get(rating, 0.0) * mult
                loss   = DEFAULT_LOSS_RATE.get(rating, 0.0) * mult * LGD * 100
                cap    = CAPITAL_CHARGE.get(rating, 0.0) * mult

                # Exit penalty: forgo future income, but also eliminate risk
                if action == 'Exit' and exposure != 'Low':
                    coupon *= 0.1  # residual
                    loss   *= 0.1
                    cap    *= 0.1

                self.R[ai, si] = coupon - loss - cap

                # Transition: rating changes stochastically, exposure is deterministic
                ri = self.ratings.index(rating)
                for rj, next_rating in enumerate(self.ratings):
                    sj = self.state_idx[(next_rating, next_exp)]
                    self.T[ai, si, sj] += self.P_credit[ri, rj]

    # ───────────────────────────────────────────────────────────
    # VALUE ITERATION
    # ───────────────────────────────────────────────────────────

    def value_iteration(self, tol: float = 1e-6, max_iter: int = 1000):
        """
        Standard Value Iteration.

        Returns:
            V (ndarray), policy (ndarray of action indices),
            convergence_history (list of max Bellman residuals)
        """
        nS, nA = self.n_states, len(self.actions)
        V = np.zeros(nS)
        history = []

        for it in range(max_iter):
            Q = np.zeros((nA, nS))
            for a in range(nA):
                Q[a] = self.R[a] + self.gamma * self.T[a] @ V
            V_new = Q.max(axis=0)
            delta = np.max(np.abs(V_new - V))
            history.append(float(delta))
            V = V_new
            if delta < tol:
                break

        policy = np.zeros(nS, dtype=int)
        Q = np.zeros((nA, nS))
        for a in range(nA):
            Q[a] = self.R[a] + self.gamma * self.T[a] @ V
        policy = Q.argmax(axis=0)

        self.V_vi = V
        self.policy_vi = policy
        self.convergence_vi = history
        return V, policy, history

    # ───────────────────────────────────────────────────────────
    # POLICY ITERATION
    # ───────────────────────────────────────────────────────────

    def policy_iteration(self, max_iter: int = 100):
        """
        Standard Policy Iteration.

        Returns:
            V (ndarray), policy (ndarray of action indices),
            convergence_history (list of policy-change counts)
        """
        nS, nA = self.n_states, len(self.actions)
        policy = np.zeros(nS, dtype=int)
        history = []

        for it in range(max_iter):
            # --- policy evaluation ---
            # Build P_pi and R_pi under current policy
            P_pi = np.zeros((nS, nS))
            R_pi = np.zeros(nS)
            for s in range(nS):
                a = policy[s]
                P_pi[s] = self.T[a, s]
                R_pi[s] = self.R[a, s]
            I = np.eye(nS)
            V = np.linalg.solve(I - self.gamma * P_pi, R_pi)

            # --- policy improvement ---
            Q = np.zeros((nA, nS))
            for a in range(nA):
                Q[a] = self.R[a] + self.gamma * self.T[a] @ V
            new_policy = Q.argmax(axis=0)
            changes = int(np.sum(new_policy != policy))
            history.append(changes)
            policy = new_policy
            if changes == 0:
                break

        self.V_pi = V
        self.policy_pi = policy
        self.convergence_pi = history
        return V, policy, history

    # ───────────────────────────────────────────────────────────
    # REPORTING HELPERS
    # ───────────────────────────────────────────────────────────

    def get_policy_table(self, policy: np.ndarray = None) -> pd.DataFrame:
        """
        Return a human-readable table: Rating × Exposure → Action.
        """
        if policy is None:
            if hasattr(self, 'policy_vi'):
                policy = self.policy_vi
            else:
                _, policy, _ = self.value_iteration()

        records = []
        for si, (rating, exposure) in enumerate(self.states):
            records.append({
                'Rating': rating,
                'Exposure': exposure,
                'Action': self.actions[policy[si]],
            })
        df = pd.DataFrame(records)
        return df.pivot(index='Rating', columns='Exposure', values='Action')

    def get_value_table(self, V: np.ndarray = None) -> pd.DataFrame:
        """State values as a pivot table."""
        if V is None:
            V = self.V_vi if hasattr(self, 'V_vi') else self.value_iteration()[0]
        records = []
        for si, (rating, exposure) in enumerate(self.states):
            records.append({
                'Rating': rating,
                'Exposure': exposure,
                'Value': round(float(V[si]), 2),
            })
        df = pd.DataFrame(records)
        return df.pivot(index='Rating', columns='Exposure', values='Value')

    def get_policy_for_rating(self, rating: str,
                              policy: np.ndarray = None) -> dict:
        """Return {exposure: action} for a given rating."""
        if policy is None:
            policy = self.policy_vi if hasattr(self, 'policy_vi') else self.value_iteration()[1]
        out = {}
        for exp in self.exposures:
            si = self.state_idx[(rating, exp)]
            out[exp] = self.actions[policy[si]]
        return out

    def interpret_action(self, action: str) -> str:
        """Friendly description of each action."""
        interp = {
            'Increase': "📈 Increase exposure — the risk-reward favours growth",
            'Hold':     "📊 Maintain exposure — current allocation is optimal",
            'Reduce':   "📉 Reduce exposure — deterioration risk outweighs return",
            'Exit':     "🚫 Exit exposure — unacceptable risk; wind down position",
        }
        return interp.get(action, action)

    # ───────────────────────────────────────────────────────────
    # GAMMA SENSITIVITY
    # ───────────────────────────────────────────────────────────

    def gamma_sensitivity(self, gammas=None) -> pd.DataFrame:
        """
        Run value iteration for different γ and report optimal actions.
        """
        if gammas is None:
            gammas = [0.80, 0.85, 0.90, 0.95, 0.99]
        records = []
        for g in gammas:
            saved = self.gamma
            self.gamma = g
            self._build_models()
            _, policy, _ = self.value_iteration()
            for si, (rating, exposure) in enumerate(self.states):
                records.append({
                    'gamma': g,
                    'Rating': rating,
                    'Exposure': exposure,
                    'Action': self.actions[policy[si]],
                })
            self.gamma = saved
        self._build_models()  # restore
        return pd.DataFrame(records)
