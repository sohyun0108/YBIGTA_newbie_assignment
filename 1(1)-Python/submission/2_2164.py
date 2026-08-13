from __future__ import annotations
from collections import deque


"""
TODO:
- rotate_and_remove 구현하기 
"""


def create_circular_queue(n: int) -> deque[int]:
    """1부터 n까지의 숫자로 deque를 생성합니다."""
    return deque(range(1, n + 1))

def rotate_and_remove(queue: deque[int], k: int) -> int:
    """
    큐에서 k번째 원소를 제거하고 반환합니다.
    """
    # k-1번 왼쪽으로 회전하여 k번째 원소를 맨 앞으로 가져옵니다.
    queue.rotate(-(k - 1))
    # 맨 앞으로 온 k번째 원소를 제거하고 반환합니다.
    return queue.popleft()




"""
TODO:
- simulate_card_game 구현하기
    # 카드 게임 시뮬레이션 구현
        # 1. 큐 생성
        # 2. 카드가 1장 남을 때까지 반복
        # 3. 마지막 남은 카드 반환
"""


def simulate_card_game(n: int) -> int:
    """
    카드2 문제의 시뮬레이션
    맨 위 카드를 버리고, 그 다음 카드를 맨 아래로 이동
    """
   # 1. 1부터 n까지의 카드가 담긴 큐를 생성합니다.
    queue = create_circular_queue(n)
    
    # 2. 카드가 1장 남을 때까지 반복합니다.
    while len(queue) > 1:
        queue.popleft()  # 제일 위의 카드를 바닥에 버립니다.
        if queue:
            queue.append(queue.popleft())  # 그 다음 위의 카드를 맨 아래로 이동 시킵니다.
            
    # 3. 마지막 남은 카드 반환
    return queue[0]

def solve_card2() -> None:
    """입, 출력 format"""
    n: int = int(input())
    result: int = simulate_card_game(n)
    print(result)

if __name__ == "__main__":
    solve_card2()