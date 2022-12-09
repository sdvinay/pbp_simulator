import pandas as pd

lookup_cols = ['event_cd', 'start_bases_cd', 'outs_ct']
outcome_cols = ['bat_dest_id', 'run1_dest_id', 'run2_dest_id', 'run3_dest_id']

transitions = pd.read_csv('transitions.csv')
x = list(zip(*[transitions[c] for c in lookup_cols]))
outcomes = list(zip(*[transitions[c] for c in outcome_cols]))
event_transition_map = dict(zip(x, outcomes))


