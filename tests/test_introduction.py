"""Introductory file to get users familiar with tests and pull requests.

Perform the following steps:
- Add your username to the user_list
- Add a test to the file below and name it test_<username>
- Assert something trivial.
- Run the bashs script ./scripts/test-check.sh
- Fix any and all errors that occur, rerun untill all are passing
- Create a pull request to add your changes.
"""

import pytest


class TestIntroductory:
    @pytest.fixture
    def user_list(self):
        return ["Zain-Mahmoud", "kurbydoo", "mouftz"]

    def test_kurbydoo(self, user_list):
        assert 1 + 1 == 2, "Somthing went really wrong"

        n = 10
        counter = 0
        for i in range(1, n):
            counter += i

        assert counter == n * (n - 1) // 2, "Check arithmetic sum"

        assert len(user_list) == 3

    ### Add your tests below, follow the format above

    def test_Zain_Mahmoud(self, user_list):

        sorted_list = sorted(user_list)
        assert all(sorted_list[0] <= x for x in user_list), "Check sorted"

        num_list = range(15)
        assert all((2 * x) % 2 == 0 for x in num_list), "Check even"

    def test_mouftz(self, user_list):
        assert "racecar" == "racecar"[::-1], "Check palindrome"
