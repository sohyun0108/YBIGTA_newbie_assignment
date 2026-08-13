from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Iterable


"""
TODO:
- Trie.push 구현하기
- (필요할 경우) Trie에 추가 method 구현하기
"""


T = TypeVar("T")


@dataclass
class TrieNode(Generic[T]):
    body: Optional[T] = None
    children: list[int] = field(default_factory=lambda: [])
    is_end: bool = False


class Trie(list[TrieNode[T]]):
    def __init__(self) -> None:
        super().__init__()
        self.append(TrieNode(body=None))

    def push(self, seq: Iterable[T]) -> None:
        """
        seq: T의 열 (list[int]일 수도 있고 str일 수도 있고 등등...)

        action: trie에 seq을 저장하기
        """
        pointer = 0
        for element in seq:
            found = -1
            # 현재 노드의 자식들 중 같은 글자가 있는지 인덱스를 탐색합니다
            for child_idx in self[pointer].children:
                if self[child_idx].body == element:
                    found = child_idx
                    break
            
            # 이미 존재하면 해당 노드로 이동합니다
            if found != -1:
                pointer = found
            # 존재하지 않으면 새로운 노드를 flat list 맨 뒤에 추가하고 연결합니다
            else:
                new_idx = len(self)
                self.append(TrieNode(body=element))
                self[pointer].children.append(new_idx)
                pointer = new_idx
                
        # 문자열이 끝나는 지점에 표시합니다
        self[pointer].is_end = True