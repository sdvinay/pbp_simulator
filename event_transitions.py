import pandas as pd

transitions = pd.read_csv('~/temp/transitions.csv')
x = list(zip(*[transitions[c] for c in ['event_cd', 'start_bases_cd', 'outs_ct']]))
outcomes = list(zip(*[transitions[c] for c in ['event_outs_ct', 'event_runs_ct', 'end_bases_cd']]))
event_transition_map = dict(zip(x, outcomes))


