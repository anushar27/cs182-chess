"""Exact Hamming-radius lookup using a pigeonhole index, with compact states."""
from collections import defaultdict
import chess
from .state import State


def geometry(board):
    return board.turn, bytes(ord(board.piece_at(s).symbol()) if board.piece_at(s) else 46 for s in chess.SQUARES)


def game_geometries(game):
    b=State.from_dict(game['initial']).board()
    yield geometry(b)
    for uci in game['moves']:
        b.push_uci(uci)
        yield geometry(b)


class NearIndex:
    def __init__(self, radius=2):
        if not 0 <= radius < 64:raise ValueError('Radius must be in [0,63]')
        self.radius=radius;self.parts=radius+1
        self.items=[];self.buckets=defaultdict(list)

    def add(self,item):
        side,squares=item;i=len(self.items);self.items.append(item)
        for part in range(self.parts):
            self.buckets[(side,part,squares[part::self.parts])].append(i)

    def has_near(self,item):
        side,squares=item;seen=set()
        # <=r changes leave at least one of r+1 disjoint square subsets unchanged.
        for part in range(self.parts):
            for i in self.buckets.get((side,part,squares[part::self.parts]),()):
                if i in seen:continue
                seen.add(i);other=self.items[i][1];count=0
                for a,b in zip(squares,other):
                    count+=a!=b
                    if count>self.radius:break
                if count<=self.radius:return True
        return False
