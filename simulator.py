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

@dataclass
class GameState:
    scores: tuple[int] = dataclasses.field(default_factory=lambda: (0, 0))
    inning_num: int = 1
    inning_half_bottom: bool = False
    outs: int = 0
    bases: list[int] = dataclasses.field(default_factory=lambda : [])
    batter_t1: int = 0
    batter_t2: int = 0

    def get_current_batter(self):
        return self.batter_t2 if self.inning_half_bottom else self.batter_t1

    def __str__(self) -> str:
        score = f'{self.scores[0]}-{self.scores[1]}'
        inn = f'{"b" if self.inning_half_bottom else "t"}{self.inning_num}'
        bases = "".join([str(b) if b in self.bases else "-" for b in (1, 2, 3)])
        batter = self.get_current_batter()
        return f'{score} {inn}({batter+1}) {self.outs}o {bases}'


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

def is_game_over(g: GameState) -> bool:
    return g.inning_num>9


def add_out(g: GameState) -> GameState:
    return dataclasses.replace(g, outs=g.outs+1)


def advance_inning(g: GameState) -> GameState:
    inning_num = g.inning_num + int(g.inning_half_bottom)
    return dataclasses.replace(g, outs=0, bases=[], inning_num=inning_num, inning_half_bottom=(not g.inning_half_bottom))


def add_runs(g: GameState, runs: int = 1) -> GameState:
    new_score = (g.scores[0], g.scores[1]+runs) if g.inning_half_bottom else (g.scores[0]+runs, g.scores[1])
    return dataclasses.replace(g, scores=new_score)


def increment_batter(g: GameState) -> GameState:
    if g.inning_half_bottom:
        return dataclasses.replace(g, batter_t2=(g.batter_t2+1)%9)
    else:
        return dataclasses.replace(g, batter_t1=(g.batter_t1+1)%9)

def get_event_dist(g: GameState) -> dict:
    return lineups.get_event_dist(g.get_current_batter())


def select_event(event_dist: dict) -> EventType:
    choice = random.choices(list(event_dist.keys()), weights=event_dist.values(), k=1)[0]
    return event_mapper.get(choice, choice)


def advance_runners(g: GameState, num: int = 1) -> GameState:
    bases_new = [b+num for b in [0]+g.bases if b+num <=3]
    num_runs = sum([b+num >3 for b in [0]+g.bases])
    g = add_runs(g, num_runs)
    g = dataclasses.replace(g, bases=bases_new)
    return g


def force_runners(g: GameState) -> GameState:
    if len(g.bases)==3:
        return add_runs(g, 1)
    b = [1]
    match g.bases:
        case [1] | [2]: b = [1, 2]
        case [3]: b = [1, 3]
        case [_, _]: b= [1, 2, 3]

    return dataclasses.replace(g, bases=b)


advance = {ev: bases for bases, ev in enumerate([EventType.S, EventType.D, EventType.T, EventType.HR], 1)}
def apply_event_to_GS(g: GameState, ev: EventType) -> GameState:
    match ev:
        case EventType.K | EventType.OUT:
            return add_out(g)
        case EventType.BB | EventType.HBP:
            return force_runners(g)
        case EventType.S | EventType.D | EventType.T | EventType.HR:
            return advance_runners(g, advance[ev])
        case _:
            raise KeyError(f'Unknown event "{ev}"')


def sim_game(g: GameState = GameState()) -> List[Tuple[GameState, EventType]]:
    game_states = []
    while not is_game_over(g):
        event_dist = get_event_dist(g)
        event = select_event(event_dist)
        game_states.append((g, event))
        g = apply_event_to_GS(g, event)
        g = increment_batter(g)
        if g.outs>=3:
            g = advance_inning(g)

    game_states.append((g, None))
    return game_states


def summarize_game(results: List[Tuple[GameState, EventType]], game_id: int = 0):
    def process_row(play_id, g, e):
        d = dataclasses.asdict(g)
        d['event'] = EventType(e).name if e else e
        d['play_id'] = play_id
        return d
    plays = pd.json_normalize([ process_row(i, g, e) for (i, (g, e)) in enumerate(results)])
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