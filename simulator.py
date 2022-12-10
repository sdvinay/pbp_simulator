# ### Simulate a game, play-by-play

# #### Algorithm:
# * Define a game state dataclass
# * Begin with initial game state
# * While game is not over:
#     * Compute/retrieve a probability distribution of events
#     * Choose an event from the distribution
#     * Apply the event to the current game state to get a new game state
#     * Emit some pbp data
# * Emit game results

# All of these can start out simply and evolve.  E.g. start with a single inning, and only two events (K and HR) and static probabilities, and simple rules for ending the game.

from typing import List, Tuple
from enum import Enum, IntEnum
from dataclasses import dataclass
import dataclasses
import random
import pandas as pd
import numpy as np


import lineups
import event_transitions
from game_state import GameState
import game_state as gs

class EventType(IntEnum):
    OUT = 2
    K = 3
    BB = 14
    HBP = 16
    ROE = 18
    FC = 19
    S = 20
    D = 21
    T = 22
    HR = 23

event_mapper = {
    '1B' : EventType.S,
    '2B' : EventType.D,
    '3B' : EventType.T,
    'HR' : EventType.HR,
    'BB' : EventType.BB,
    'SO' : EventType.K,
    'K' : EventType.K,
    'Out' : EventType.OUT,
    'HBP' : EventType.HBP,
}


def get_event_dist(g: GameState) -> dict:
    return lineups.get_event_dist(g.get_current_batter())


def select_event(event_dist: dict) -> EventType:
    choice = random.choices(list(event_dist.keys()), weights=event_dist.values(), k=1)[0]
    return event_mapper.get(choice, choice)

def get_transition_from_event(g: GameState, ev: EventType):
    transitions = event_transitions.event_transition_map.get((ev, g.get_bases_as_num(), g.outs))
    return transitions

def sim_game(g: GameState = GameState()) -> List[Tuple[GameState, EventType]]:
    game_states = []
    while not gs.is_game_over(g):
        event_dist = get_event_dist(g)
        event = select_event(event_dist)
        transitions = get_transition_from_event(g, event)
        game_states.append((g, event, transitions))
        g = gs.apply_transitions_to_GS(g, transitions)
        g = gs.increment_batter(g)
        if g.outs>=3:
            g = gs.advance_inning(g)

    game_states.append((g, None, None))
    return game_states


def summarize_game(results: List[Tuple[GameState, EventType]], game_id: int = 0):
    def process_row(play_id, g, e, t):
        d = dataclasses.asdict(g)
        d['event'] = EventType(e).name if e else e
        d['transition'] = t
        d['play_id'] = play_id
        return d
    plays = pd.DataFrame([ process_row(i, g, e, t) for (i, (g, e, t)) in enumerate(results)])
    plays['game_id'] = game_id
    plays['batter_id'] = np.where(plays['inning_half_bottom'], plays['batter_t2'], plays['batter_t1'])

    summary = plays.groupby(['inning_half_bottom'])['event'].value_counts().unstack().fillna(0).astype(int)
    final = results[-1][0]
    summary['R']= pd.Series({False: final.scores[0], True: final.scores[1]})
    summary['RA']= pd.Series({True: final.scores[0], False: final.scores[1]})
    summary['W'] = summary['R'] > summary['RA']
    summary['L'] = summary['RA'] > summary['R']
    summary['game_id'] = game_id
    return (plays, summary)


def sim_games(num_games: int, g: GameState = GameState()):
    results = [summarize_game(sim_game(g), i) for i in range(num_games)]

    p, s = zip(*results)
    plays = pd.concat(p).set_index(['play_id', 'game_id'])
    summaries = pd.concat(s).reset_index().rename(columns={'inning_half_bottom':'home'}).set_index(['game_id', 'home'])
    event_cols = [ev.name for ev in EventType if ev.name in summaries.columns]
    for col in event_cols:
        summaries[col] = summaries[col].fillna(0).astype(int)
    cols = event_cols + ['R', 'RA', 'W', 'L']

    return (plays, summaries[cols])