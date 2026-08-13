from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Callable, cast


"""
TODO:
- SegmentTree 구현하기
"""


T = TypeVar("T")
U = TypeVar("U")


class SegmentTree(Generic[T, U]):
    def __init__(self, size: int, default_val: T, merge_fn: Callable[[T, T], T]) -> None:
        """
        세그먼트 트리 생성자
        size: 관리할 데이터의 최대 크기
        default_val: 구간을 벗어났을 때 반환할 기본값
        merge_fn: 두 자식 노드의 데이터를 하나로 합치는 함수
        """
        self.size = size
        self.default_val = default_val
        self.merge_fn = merge_fn
        # 1-indexed 표현을 위해 4 * size 크기로 트리 배열 초기화
        self.tree: list[T] = [default_val] * (4 * size)

    def update(self, idx: int, val: T) -> None:
        """
        특정 인덱스의 값을 갱신하고 상위 노드들을 재계산합니다.
        idx: 갱신할 원소의 위치 (1-indexed)
        val: 새롭게 갱신할 노드 데이터
        """
        def _update(node: int, start: int, end: int, target_idx: int, new_val: T) -> None:
            if start == end:
                self.tree[node] = new_val
                return
            mid = (start + end) // 2
            if target_idx <= mid:
                _update(node * 2, start, mid, target_idx, new_val)
            else:
                _update(node * 2 + 1, mid + 1, end, target_idx, new_val)
            self.tree[node] = self.merge_fn(self.tree[node * 2], self.tree[node * 2 + 1])

        _update(1, 1, self.size, idx, val)

    def query(self, left: int, right: int) -> T:
        """
        주어진 범위 [left, right]의 구간 연산 결과를 구합니다.
        left: 시작 구간 인덱스
        right: 끝 구간 인덱스
        """
        def _query(node: int, start: int, end: int, l: int, r: int) -> T:
            if r < start or end < l:
                return self.default_val
            if l <= start and end <= r:
                return self.tree[node]
            mid = (start + end) // 2
            p1 = _query(node * 2, start, mid, l, r)
            p2 = _query(node * 2 + 1, mid + 1, end, l, r)
            return self.merge_fn(p1, p2)

        return _query(1, 1, self.size, left, right)

    def find_kth(self, k: int) -> int:
        """
        트리를 이진 탐색하여 누적 합이 k번째에 도달하는 원소의 인덱스를 찾습니다 (BOJ 2243용).
        k: 찾고자 하는 사탕의 순위
        """
        node = 1
        start = 1
        end = self.size
        while start != end:
            mid = (start + end) // 2
            left_val = cast(int, self.tree[node * 2])
            if k <= left_val:
                node = node * 2
                end = mid
            else:
                k -= left_val
                node = node * 2 + 1
                start = mid + 1
        return start