from lib import Trie
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