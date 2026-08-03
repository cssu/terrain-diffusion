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
        return [
            # add your usernames here
            "kurbydoo",
            "DanisLol",
        ]

    def test_kurbydoo(self, user_list):
        assert 1 + 1 == 2, "Somthing went really wrong"

        n = 10
        counter = 0
        for i in range(1, n):
            counter += i

        assert counter == n * (n - 1) // 2, "Check arithmetic sum"

        assert len(user_list) == 2

    ### Add your tests below, follow the format above
    def test_danislol(self, user_list):
        assert 2 + 2 == 4, "basic math"

        assert True is True
        
