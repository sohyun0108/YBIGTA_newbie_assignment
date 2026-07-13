from __future__ import annotations
import copy


"""
TODO:
- __setitem__ 구현하기
- __pow__ 구현하기 (__matmul__을 활용해봅시다)
- __repr__ 구현하기
"""


class Matrix:
    MOD = 1000

    def __init__(self, matrix: list[list[int]]) -> None:
        self.matrix = matrix

    @staticmethod
    def full(n: int, shape: tuple[int, int]) -> Matrix:
        return Matrix([[n] * shape[1] for _ in range(shape[0])])

    @staticmethod
    def zeros(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(0, shape)

    @staticmethod
    def ones(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(1, shape)

    @staticmethod
    def eye(n: int) -> Matrix:
        matrix = Matrix.zeros((n, n))
        for i in range(n):
            matrix[i, i] = 1
        return matrix

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.matrix), len(self.matrix[0]))

    def clone(self) -> Matrix:
        return Matrix(copy.deepcopy(self.matrix))

    def __getitem__(self, key: tuple[int, int]) -> int:
        return self.matrix[key[0]][key[1]]

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        """행렬의 특정 위치(row, col)에 값을 할당합니다."""
        self.matrix[key[0]][key[1]] = value

    def __matmul__(self, matrix: Matrix) -> Matrix:
        x, m = self.shape
        m1, y = matrix.shape
        assert m == m1

        result = self.zeros((x, y))

        for i in range(x):
            for j in range(y):
                for k in range(m):
                    result[i, j] += self[i, k] * matrix[k, j]

        return result

    def __pow__(self, n: int) -> Matrix:
        """
        분할 정복을 이용한 행렬의 거듭제곱을 구합니다.
        __matmul__ 연산(@)을 활용하며, 연산 도중 숫자가 비대해지는 것을 방지하기 위해 
        매 곱셈 직후 각 원소를 Matrix.MOD로 나눈 나머지를 취합니다.
        """
        size = self.shape[0]
        result = Matrix.eye(size)
        base = self.clone()

        # 초기 행렬 원소 중 MOD(1000) 이상인 값이 있을 수 있으므로 먼저 나머지 연산을 수행합니다.
        for i in range(size):
            for j in range(size):
                base.matrix[i][j] %= self.MOD

        while n > 0:
            if n % 2 == 1:
                result = result @ base
                for i in range(size):
                    for j in range(size):
                        result.matrix[i][j] %= self.MOD
            base = base @ base
            for i in range(size):
                for j in range(size):
                    base.matrix[i][j] %= self.MOD
            n //= 2

        return result

    def __repr__(self) -> str:
        # 행렬을 백준 출력 형식(행 내 원소는 공백 구분, 행끼리는 줄바꿈)에 맞게 문자열로 변환합니다
        return "\n".join(" ".join(map(str, row)) for row in self.matrix)


from typing import Callable
import sys


"""
-아무것도 수정하지 마세요!
"""


def main() -> None:
    intify: Callable[[str], list[int]] = lambda l: [*map(int, l.split())]

    lines: list[str] = sys.stdin.readlines()

    N, B = intify(lines[0])
    matrix: list[list[int]] = [*map(intify, lines[1:])]

    Matrix.MOD = 1000
    modmat = Matrix(matrix)

    print(modmat ** B)


if __name__ == "__main__":
    main()