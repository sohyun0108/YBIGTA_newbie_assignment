from __future__ import annotations
from lib import SegmentTree
import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


class Pair(tuple[int, int]):
    """
    힌트: 2243, 3653에서 int에 대한 세그먼트 트리를 만들었다면 여기서는 Pair에 대한 세그먼트 트리를 만들 수 있을지도...?
    """
    def __new__(cls, a: int, b: int) -> 'Pair':
        return super().__new__(cls, (a, b))

    @staticmethod
    def default() -> 'Pair':
        """
        기본값
        이게 왜 필요할까...?
        """
        return Pair(0, 0)

    @staticmethod
    def f_conv(w: int) -> 'Pair':
        """
        원본 수열의 값을 대응되는 Pair 값으로 변환하는 연산
        이게 왜 필요할까...?
        """
        return Pair(w, 0)

    @staticmethod
    def f_merge(a: Pair, b: Pair) -> 'Pair':
        """
        두 Pair를 하나의 Pair로 합치는 연산
        이게 왜 필요할까...?
        """
        return Pair(*sorted([*a, *b], reverse=True)[:2])

    def sum(self) -> int:
        return self[0] + self[1]


def main() -> None:
    """
    BOJ 17408 수열과 쿼리 24 문제를 해결하는 메인 함수입니다
    제공된 Pair 클래스와 f_merge 연산을 활용해 세그먼트 트리를 구축합니다
    """
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    a_elements = [int(x) for x in input_data[1:n+1]]
    m = int(input_data[n+1])
    
    # 원본 Pair 구조를 그대로 받아 사용할 수 있도록 트리 생성
    tree: SegmentTree[Pair] = SegmentTree(n, Pair.default(), Pair.f_merge)
    
    # 초기 수열 값 세팅
    for i in range(1, n + 1):
        tree.update(i, Pair.f_conv(a_elements[i-1]))
        
    idx = n + 2
    out = []
    for _ in range(m):
        q_type = int(input_data[idx])
        if q_type == 1:
            i = int(input_data[idx+1])
            v = int(input_data[idx+2])
            tree.update(i, Pair.f_conv(v))
            idx += 3
        else:
            l = int(input_data[idx+1])
            r = int(input_data[idx+2])
            res = tree.query(l, r)
            out.append(str(res.sum()))
            idx += 3
            
    print('\n'.join(out))


if __name__ == "__main__":
    main()