from lib import Trie
import sys


"""
TODO:
- 일단 Trie부터 구현하기
- count 구현하기
- main 구현하기
"""


def count(trie: Trie, query_seq: str) -> int:
    """
    trie - 이름 그대로 trie
    query_seq - 단어 ("hello", "goodbye", "structures" 등)

    returns: query_seq의 단어를 입력하기 위해 버튼을 눌러야 하는 횟수
    """
    pointer = 0
    cnt = 0

    for element in query_seq:
        # 갈래길이 있거나, 여기서 끝나는 다른 단어가 있다면 타이핑 횟수를 추가합니다.
        if len(trie[pointer].children) > 1 or trie[pointer].is_end:
            cnt += 1

        new_index = 0
        # 일치하는 자식 노드가 있는지 확인하고, 있으면 그 노드로 이동합니다
        for child_index in trie[pointer].children:
            if trie[child_index].body == element:
                new_index = child_index
                break

        pointer = new_index

    # 첫 글자는 무조건 눌러야 하므로 루트 자식이 1개뿐이었을 때 예외를 보정합니다
    return cnt + int(len(trie[0].children) == 1)


def main() -> None:
    """
    휴대폰 자판 문제를 해결하는 메인 함수입니다
    여러 개의 테스트 케이스를 순차적으로 파싱하여 처리합니다
    """
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    iterator = iter(input_data)
    while True:
        try:
            N_str = next(iterator)
        except StopIteration:
            break
        
        N = int(N_str)
        words = []
        trie = Trie()
        
        for _ in range(N):
            word = next(iterator)
            words.append(word)
            trie.push(word)
            
        total_presses = 0
        for word in words:
            total_presses += count(trie, word)
            
        avg = total_presses / N
        # 소수점 둘째 자리까지 반올림하여 출력합니다
        print(f"{avg:.2f}")


if __name__ == "__main__":
    main()