"""Tie-accounted statistics from Appendix G.1 (rates are fractions)."""
import math
from statistics import NormalDist, mean, stdev


def pairwise_statistics(wins, ties, losses):
    n = wins+ties+losses
    if not n:
        raise ValueError('No valid pairwise observations')
    p = (wins+.5*ties)/n
    variance = max(0., (wins+.25*ties)/n-p*p)
    se = math.sqrt(variance/n)
    half_width = NormalDist().inv_cdf(.975)*se
    return dict(n=n, wins=wins, ties=ties, losses=losses, win_rate=p,
                standard_error=se, ci95=[p-half_width,p+half_width], ci95_half_width=half_width)


def satisfaction_statistics(rates):
    if not rates:
        raise ValueError('No valid checklist observations')
    return dict(n=len(rates), mean=mean(rates), standard_error=stdev(rates)/math.sqrt(len(rates)) if len(rates)>1 else 0.)
