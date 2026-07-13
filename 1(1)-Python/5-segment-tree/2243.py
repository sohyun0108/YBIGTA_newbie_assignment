from lib import SegmentTree
import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


def main() -> None:
    """
    BOJ 2243 사탕상자 문제를 해결하는 메인 함수입니다
    맛의 등급을 인덱스로 삼아 사탕 개수를 세그먼트 트리로 관리하고 k번째 사탕을 꺼냅니다
    """
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    MAX_TASTE = 1000000
    
    # 사탕 개수 합을 관리하는 트리를 생성합니다
    tree: SegmentTree[int, int] = SegmentTree(MAX_TASTE, 0, lambda x, y: x + y)
    counts = [0] * (MAX_TASTE + 1)
    
    idx = 1
    out = []
    for _ in range(n):
        a = int(input_data[idx])
        if a == 1:
            b = int(input_data[idx+1])
            # k번째 맛있는 사탕을 추적합니다
            taste = tree.find_kth(b)
            out.append(str(taste))
            counts[taste] -= 1
            tree.update(taste, counts[taste])
            idx += 2
        else:
            b = int(input_data[idx+1])
            c = int(input_data[idx+2])
            counts[b] += c
            tree.update(b, counts[b])
            idx += 3
            
    print('\n'.join(out))


if __name__ == "__main__":
    main()