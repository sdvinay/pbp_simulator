from dataclasses import dataclass
import dataclasses

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

    def get_bases_as_num(self):
        return sum(2**(b-1) for b in self.bases)

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