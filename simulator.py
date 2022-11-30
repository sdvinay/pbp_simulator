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

from dataclasses import dataclass
import dataclasses
import random
import pandas as pd

@dataclass
class GameState:
    score_t1: int = 0
    score_t2: int = 0
    inning_num: int = 1
    inning_half_bottom: bool = False
    outs: int = 0
    bases: list[int] = dataclasses.field(default_factory=lambda : [])

    def __str__(self) -> str:
        score = f'{self.score_t1}-{self.score_t2}'
        inn = f'{"b" if self.inning_half_bottom else "t"}{self.inning_num}'
        bases = "".join([str(b) if b in self.bases else "-" for b in (1, 2, 3)])
        return f'{score} {inn} {self.outs}o {bases}'


def is_game_over(g: GameState) -> bool:
    return g.inning_num>9


def add_out(g: GameState) -> GameState:
    if g.outs < 2:
        return dataclasses.replace(g, outs=g.outs+1)
    else:
        if g.inning_half_bottom:
            return dataclasses.replace(g, outs=0, bases=[], inning_num=g.inning_num+1, inning_half_bottom=False)
        else:
            return dataclasses.replace(g, outs=0, bases=[], inning_half_bottom=True)


def add_runs(g: GameState, runs: int = 1) -> GameState:
    if g.inning_half_bottom:
        return dataclasses.replace(g, score_t2=g.score_t2+runs)
    else:
        return dataclasses.replace(g, score_t1=g.score_t1+runs)


def get_event_dist():
    return {'1B': 8, '2B': 2, '3B': 1, 'HR': 1, 'K': 27, 'BB': 4}


def select_event(event_dist):
    return random.choices(list(event_dist.keys()), weights=event_dist.values(), k=1)[0]


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


def apply_event_to_GS(g: GameState, ev) -> GameState:
    match ev:
        case 'K':
            return add_out(g)
        case 'BB':
            return force_runners(g)
        case '1B':
            return advance_runners(g, 1)
        case '2B':
            return advance_runners(g, 2)
        case '3B':
            return advance_runners(g, 3)
        case 'HR':
            return advance_runners(g, 4)
        case _:
            raise KeyError(f'Unknown event "{ev}"')


def sim_game():
    # initial state
    g = GameState()
    event_dist = get_event_dist()

    game_states = []
    while not is_game_over(g):
        event = select_event(event_dist)
        game_states.append((g, event))
        g = apply_event_to_GS(g, event)

    game_states.append((g, None))
    return game_states


def summarize_game(results, game_id: int = 0):
    def process_row(play_id, g, e):
        d = dataclasses.asdict(g)
        d['event'] = e
        d['play_id'] = play_id
        return d
    plays = pd.json_normalize([ process_row(i, g, e) for (i, (g, e)) in enumerate(results)])
    plays['game_id'] = game_id
    summary = plays.groupby(['inning_half_bottom'])['event'].value_counts().unstack().fillna(0).astype(int)
    summary['R']= pd.Series({False: results[-1][0].score_t1,
                            True: results[-1][0].score_t2})
    summary['RA']= pd.Series({True: results[-1][0].score_t1,
                            False: results[-1][0].score_t2})
    summary['W'] = summary['R'] > summary['RA']
    summary['L'] = summary['RA'] > summary['R']
    summary['game_id'] = game_id
    return (plays, summary)


def sim_games(num_games: int):
    results = [summarize_game(sim_game(), i) for i in range(num_games)]

    p, s = zip(*results)
    plays = pd.concat(p).set_index(['play_id', 'game_id'])
    summaries = pd.concat(s).reset_index().rename(columns={'inning_half_bottom':'home'}).set_index(['game_id', 'home'])

    event_cols=['1B', '2B', '3B', 'BB', 'HR', 'K']
    for col in event_cols:
        summaries[col] = summaries[col].fillna(0).astype(int)

    return (plays, summaries)