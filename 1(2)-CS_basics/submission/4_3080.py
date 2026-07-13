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


import sys


"""
TODO:
- 일단 lib.py의 Trie Class부터 구현하기
- main 구현하기

힌트: 한 글자짜리 자료에도 그냥 str을 쓰기에는 메모리가 아깝다...
=> int로 변환해서 넣어주면 메모리 절약이 가능하다
"""


def main() -> None:
    """
    아름다운 이름 문제를 해결하는 메인 함수입니다
    Trie의 각 노드에서 자식 노드들과 단어의 끝을 배치할 수 있는 경우의 수(팩토리얼)를 구하여 누적 곱합니다
    """
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    names = input_data[1:N+1]
    
    trie = Trie()
    for name in names:
        # 힌트를 반영하여 문자열 대신 bytes(Iterable[int]) 형태로 주입해 메모리를 절약합니다
        trie.push(name.encode('ascii'))
        
    MOD = 1_000_000_007
    ans = 1
    
    # 알파벳 개수가 최대 26개이므로 필요한 팩토리얼 값(최대 27)을 미리 계산해둡니다
    fact = [1] * 28
    for i in range(1, 28):
        fact[i] = (fact[i-1] * i) % MOD
        
    # 트라이의 모든 노드를 순회하며 경우의 수를 곱해줍니다
    for node in trie:
        cnt = len(node.children)
        if node.is_end:
            cnt += 1
        ans = (ans * fact[cnt]) % MOD
        
    print(ans)


if __name__ == "__main__":
    main()