from __future__ import annotations
import copy
from collections import deque
from collections import defaultdict
from typing import DefaultDict, List


"""
TODO:
- __init__ 구현하기
- add_edge 구현하기
- dfs 구현하기 (재귀 또는 스택 방식 선택)
- bfs 구현하기
"""


class Graph:
    def __init__(self, n: int) -> None:
        """
        그래프 초기화
        n: 정점의 개수 (1번부터 n번까지)
        """
        self.n = n
        # 1번부터 n번 정점까지 인덱스로 직접 접근할 수 있도록 (n + 1) 크기로 인접 리스트 초기화
        self.adj: list[list[int]] = [[] for _ in range(n + 1)]
    
    def add_edge(self, u: int, v: int) -> None:
        """
        양방향 간선 추가
        u: 정점 1, v: 정점 2
        """
        self.adj[u].append(v)
        self.adj[v].append(u)
    
    def dfs(self, start: int) -> list[int]:
        """
        깊이 우선 탐색 (DFS)
        
        구현 방법 선택:
        깊이 우선 탐색 (DFS)을 수행하고 방문 순서대로 정점 리스트를 반환합니다.
        재귀(Recursion) 방식을 사용하여 구현했습니다.
        """
        visited = [False] * (self.n + 1)
        result: list[int] = []
        
        def _dfs(curr: int) -> None:
            visited[curr] = True
            result.append(curr)
            
            # 번호가 작은 정점부터 방문하기 위해 오름차순으로 정렬하여 순회합니다.
            for neighbor in sorted(self.adj[curr]):
                if not visited[neighbor]:
                    _dfs(neighbor)
                    
        _dfs(start)
        return result
    
    def bfs(self, start: int) -> list[int]:
        """
        너비 우선 탐색 (BFS)
        큐를 사용하여 구현
        """
        visited = [False] * (self.n + 1)
        result: list[int] = []
        queue = deque([start])
        visited[start] = True

        while queue:
            curr = queue.popleft()
            result.append(curr)

            # 번호가 작은 정점부터 방문하기 위해 오름차순으로 정렬하여 순회합니다.
            for neighbor in sorted(self.adj[curr]):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        return result

    def search_and_print(self, start: int) -> None:
        """
        DFS와 BFS 결과를 출력
        """
        dfs_result = self.dfs(start)
        bfs_result = self.bfs(start)
        
        print(' '.join(map(str, dfs_result)))
        print(' '.join(map(str, bfs_result)))
