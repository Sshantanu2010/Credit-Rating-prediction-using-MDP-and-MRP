"""
mrp_model.py — Markov Reward Process for credit-rating transitions.

Implements the MRP from first principles:
  • State space  S = {AAA, AA, A, BBB, BB, B, C, D}
  • Transition matrix  P  (D is absorbing)
  • Reward vector  R  (economic coupon / loss / penalty)
  • Bellman equation  V = R + γ P V  ⟹  V = (I − γP)⁻¹ R
  • Multi-step transition  P^k
  • Cumulative default probability over time
"""

import numpy as np
import pandas as pd
from src.utils import BUCKETED_RATINGS


# ═══════════════════════════════════════════════════════════════
# REWARD SPECIFICATION
# ═══════════════════════════════════════════════════════════════

# Annual economic reward per rating bucket (₹ Cr per unit exposure).
# Positive = coupon income net of expected loss; negative = net loss.
# Calibrated loosely to Indian corporate bond yields minus expected-loss.
DEFAULT_REWARDS = {
    'AAA':  8.5,   # ~7.5 % yield, near-zero loss
    'AA':   7.8,   # ~8 % yield, low loss
    'A':    6.5,   # ~8.5 % yield, moderate loss
    'BBB':  4.0,   # ~9.5 % yield, non-trivial loss
    'BB':   0.5,   # ~11 % yield, high expected loss
    'B':   -3.0,   # coupon barely covers expected loss
    'C':   -8.0,   # heavy provisioning
    'D':  -40.0,   # recovery-adjusted loss-given-default
}


class MRP:
    """
    Markov Reward Process for credit-rating dynamics.

    Parameters
    ----------
    transition_matrix : pd.DataFrame or np.ndarray
        8×8 matrix, rows = from-state, cols = to-state.
    rewards : dict or np.ndarray, optional
        Reward per state.  Defaults to DEFAULT_REWARDS.
    gamma : float
        Discount factor (0, 1].
    states : list[str]
        Ordered state labels.
    """

    def __init__(self, transition_matrix, rewards=None,
                 gamma: float = 0.95, states=None):
        # --- states ---
        if states is None:
            if isinstance(transition_matrix, pd.DataFrame):
                self.states = list(transition_matrix.index)
            else:
                self.states = list(BUCKETED_RATINGS)
        else:
            self.states = list(states)

        self.n = len(self.states)
        self.gamma = gamma

        # --- transition matrix ---
        if isinstance(transition_matrix, pd.DataFrame):
            self.P = transition_matrix.values.astype(float)
        else:
            self.P = np.array(transition_matrix, dtype=float)
        # Ensure rows sum to 1
        row_sums = self.P.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.P = self.P / row_sums

        # --- rewards ---
        if rewards is None:
            self.R = np.array([DEFAULT_REWARDS.get(s, 0.0) for s in self.states])
        elif isinstance(rewards, dict):
            self.R = np.array([rewards.get(s, 0.0) for s in self.states])
        else:
            self.R = np.array(rewards, dtype=float)

    # ───────────────────────────────────────────────────────────
    # Value function
    # ───────────────────────────────────────────────────────────

    def compute_state_values(self) -> np.ndarray:
        """
        Solve V = (I − γP)⁻¹ R  (closed-form Bellman solution).
        """
        I = np.eye(self.n)
        self.V = np.linalg.solve(I - self.gamma * self.P, self.R)
        return self.V

    def get_state_value_df(self) -> pd.DataFrame:
        """Return state values as a tidy DataFrame."""
        if not hasattr(self, 'V'):
            self.compute_state_values()
        return pd.DataFrame({
            'State': self.states,
            'Value': self.V.round(4),
        })

    # ───────────────────────────────────────────────────────────
    # Multi-step transitions
    # ───────────────────────────────────────────────────────────

    def multi_step_transition(self, k: int) -> np.ndarray:
        """Compute P^k — the k-step transition matrix."""
        return np.linalg.matrix_power(self.P, k)

    def multi_step_transition_df(self, k: int) -> pd.DataFrame:
        Pk = self.multi_step_transition(k)
        return pd.DataFrame(Pk, index=self.states, columns=self.states)

    # ───────────────────────────────────────────────────────────
    # Default probability over time
    # ───────────────────────────────────────────────────────────

    def default_probability_over_time(self, max_years: int = 20) -> pd.DataFrame:
        """
        Compute cumulative probability of reaching state D from every
        starting state, for each year up to max_years.
        """
        d_idx = self.states.index('D') if 'D' in self.states else self.n - 1
        records = []
        Pk = np.eye(self.n)
        for year in range(1, max_years + 1):
            Pk = Pk @ self.P
            for i, s in enumerate(self.states):
                records.append({
                    'Year': year,
                    'Starting_State': s,
                    'Default_Probability': round(float(Pk[i, d_idx]), 6),
                })
        return pd.DataFrame(records)

    # ───────────────────────────────────────────────────────────
    # Rating migration summary
    # ───────────────────────────────────────────────────────────

    def migration_summary(self, state: str, horizon: int = 1) -> dict:
        """
        For a given starting state, return the probability distribution
        over all states after `horizon` years plus a textual interpretation.
        """
        idx = self.states.index(state)
        Pk = self.multi_step_transition(horizon)
        probs = Pk[idx]

        summary = {
            'starting_state': state,
            'horizon_years': horizon,
            'distribution': {s: round(float(p), 4) for s, p in zip(self.states, probs)},
        }

        # Textual interpretation
        stay_prob = probs[idx]
        upgrade_prob = float(np.sum(probs[:idx]))
        downgrade_prob = float(np.sum(probs[idx + 1:]))
        d_prob = probs[self.states.index('D')] if 'D' in self.states else 0.0

        summary['interpretation'] = {
            'stay_probability': round(stay_prob, 4),
            'upgrade_probability': round(upgrade_prob, 4),
            'downgrade_probability': round(downgrade_prob, 4),
            'default_probability': round(float(d_prob), 4),
        }

        return summary

    # ───────────────────────────────────────────────────────────
    # Gamma sensitivity analysis
    # ───────────────────────────────────────────────────────────

    def gamma_sensitivity(self, gammas=None) -> pd.DataFrame:
        """
        Compute state values for a range of discount factors.
        """
        if gammas is None:
            gammas = [0.80, 0.85, 0.90, 0.95, 0.99]
        records = []
        for g in gammas:
            I = np.eye(self.n)
            V = np.linalg.solve(I - g * self.P, self.R)
            for s, v in zip(self.states, V):
                records.append({'gamma': g, 'State': s, 'Value': round(float(v), 4)})
        return pd.DataFrame(records)
