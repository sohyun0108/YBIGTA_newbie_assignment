from lib import SegmentTree
import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


def main() -> None:
    """
    BOJ 3653 영화 수집 문제를 해결하는 메인 함수입니다
    세그먼트 트리를 이용해 특정 DVD 왼쪽에 위치한(위에 쌓인) DVD 개수를 카운트합니다
    """
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    num_test_cases = int(input_data[0])
    idx = 1
    out = []
    
    for _ in range(num_test_cases):
        n = int(input_data[idx])
        m = int(input_data[idx+1])
        idx += 2
        
        max_size = n + m
        tree: SegmentTree[int, int] = SegmentTree(max_size, 0, lambda x, y: x + y)
        pos = [0] * (n + 1)
        
        # 최초 배치: 상단 공간 M개를 비워두고 M+1번 위치부터 DVD 배치합니다
        for i in range(1, n + 1):
            pos[i] = m + i
            tree.update(m + i, 1)
            
        top = m
        case_res = []
        for _ in range(m):
            movie = int(input_data[idx])
            idx += 1
            
            curr_pos = pos[movie]
            # 내 위치보다 앞(위)에 존재하는 DVD 개수 합 쿼리
            ans = tree.query(1, curr_pos - 1)
            case_res.append(str(ans))
            
            # 본 영화를 가장 위(top)로 이동하는 업데이트
            tree.update(curr_pos, 0)
            pos[movie] = top
            tree.update(top, 1)
            top -= 1
            
        out.append(' '.join(case_res))
        
    print('\n'.join(out))


if __name__ == "__main__":
    main()