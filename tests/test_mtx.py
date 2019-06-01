import unittest

from udsp.core import mtx


class MatrixTestCase(unittest.TestCase):

    def __do_test(self, fun, test_data):
        """
        Private refactored method to execute the test cases

        Parameters
        ----------
        fun: callable
            The test case to be executed
        test_data: dict
            A table with the test data for the case

        """
        for i, inp_exp in test_data.items():

            args = inp_exp["args"]
            expected = inp_exp["expect"]
            assrt = inp_exp["assert"] if "assert" in inp_exp else ""
            emsg = "{} failed".format(i)

            try:
                if expected is "exception":
                    with self.assertRaises(ValueError, msg=emsg):
                        fun(*args)
                elif assrt and assrt is "equal":
                    results = fun(*args)
                    self.assertEqual(results, expected, msg=emsg)
                else:
                    results = fun(*args)
                    self.assertListEqual(results, expected, msg=emsg)
            # Catch exceptions not covered by the tests and print
            # info to pinpoint the failing case, otherwise a bit obscure
            # since this code is not inline with the tests.
            except Exception as exc:
                if not isinstance(exc, AssertionError):
                    print("\n\nError in {}\nmsg: {}\nargs: {}\n".
                          format(i, exc, args))
                raise exc

# ---------------------------------------------------------
#                         Test cases
# ---------------------------------------------------------

    def test_mat_submat_copy(self):

        inplace = False
        a1 = [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]]
        a2 = [[0, 0, 0]]
        a3 = [[0], [0], [0]]
        a4 = [[0]]
        a5 = None
        b1 = [[1, 2],
              [3, 4]]
        b2 = []
        b3 = None
        b4 = [[1]]
        # Test data
        test_data = {
            "Input 1": {
                "args": [a1, b1, (1, 1), inplace],
                "expect": [[0, 0, 0],
                           [0, 1, 2],
                           [0, 3, 4]]
            },
            "Input 2": {
                "args": [a1, b1, (2, 2), inplace],
                "expect": [[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 1]]
            },
            "Input 3": {
                "args": [a1, b1, (2, 0), inplace],
                "expect": [[0, 0, 0],
                           [0, 0, 0],
                           [1, 2, 0]]
            },
            "Input 4": {
                "args": [a1, b1, (0, -1), inplace],
                "expect": "exception"
            },
            "Input 5": {
                "args": [a2, b1, (0, 2), inplace],
                "expect": [[0, 0, 1]]
            },
            "Input 6": {
                "args": [a3, b1, (0, 0), inplace],
                "expect": [[1], [3], [0]]
            },
            "Input 7": {
                "args": [a3, b1, (2, 0), inplace],
                "expect": [[0], [0], [1]]
            },
            "Input 8": {
                "args": [a4, b1, (0, 0), inplace],
                "expect": [[1]]
            },
            "Input 9": {
                "args": [a4, b1, (1, 1), inplace],
                "expect": a4
            },
            "Input 10": {
                "args": [a5, b1, (1, 1), inplace],
                "expect": []
            },
            "Input 11": {
                "args": [a1, b2, (1, 1), inplace],
                "expect": a1
            },
            "Input 12": {
                "args": [a1, b3, (1, 1), inplace],
                "expect": a1
            },
            "Input 13": {
                "args": [a4, b4, (0, 0), inplace],
                "expect": [[1]]
            },
            "Input 14": {
                "args": [a1, b4, (2, 2), inplace],
                "expect": [[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 1]]
            },
            "Input 15": {
                "args": [a1, b4, (1, 1), inplace],
                "expect": [[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]]
            },
            "Input 16": {
                "args": [a1, b4, (), inplace],
                "expect": "exception"
            }
        }
        self.__do_test(mtx.mat_submat_copy, test_data)

    def test_mat_flatten(self):

        a1 = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]
        a2 = [[1, 2, 3]]
        a3 = [[1], [2], [3]]
        a4 = [[1]]
        a5 = []
        # Test data
        test_data = {
            "Input 1": {
                "args": [a1, (1, False)],
                "expect": [1, 2, 3, 4, 5, 6, 7, 8, 9]
            },
            "Input 2": {
                "args": [a1, (1, True)],
                "expect": [7, 8, 9, 4, 5, 6, 1, 2, 3]
            },
            "Input 3": {
                "args": [a1, (2, False)],
                "expect": [1, 4, 7, 2, 5, 8, 3, 6, 9]
            },
            "Input 4": {
                "args": [a1, (2, True)],
                "expect": [3, 6, 9, 2, 5, 8, 1, 4, 7]
            },
            "Input 5": {
                "args": [a2, (1, False)],
                "expect": [1, 2, 3]
            },
            "Input 6": {
                "args": [a2, (1, True)],
                "expect": [1, 2, 3]
            },
            "Input 7": {
                "args": [a2, (2, False)],
                "expect": [1, 2, 3]
            },
            "Input 8": {
                "args": [a2, (2, True)],
                "expect": [3, 2, 1]
            },
            "Input 9": {
                "args": [a3, (1, False)],
                "expect": [1, 2, 3]
            },
            "Input 10": {
                "args": [a3, (1, True)],
                "expect": [3, 2, 1]
            },
            "Input 11": {
                "args": [a3, (2, False)],
                "expect": [1, 2, 3]
            },
            "Input 12": {
                "args": [a3, (2, True)],
                "expect": [1, 2, 3]
            },
            "Input 13": {
                "args": [a4, (1, True)],
                "expect": [1]
            },
            "Input 14": {
                "args": [a4, (1, False)],
                "expect": [1]
            },
            "Input 15": {
                "args": [a4, (2, True)],
                "expect": [1]
            },
            "Input 16": {
                "args": [a4, (2, False)],
                "expect": [1]
            },
            "Input 17": {
                "args": [a5, (1, False)],
                "expect": []
            },
            "Input 18": {
                "args": [a5, (1, True)],
                "expect": []
            },
            "Input 19": {
                "args": [a5, (2, False)],
                "expect": []
            },
            "Input 20": {
                "args": [a5, (2, True)],
                "expect": []
            },
            "Input 21": {
                "args": [a1, (3, False)],
                "expect": "exception"
            },
            "Input 22": {
                "args": [a1, (3,)],
                "expect": "exception"
            }
        }
        self.__do_test(mtx.mat_flatten, test_data)

    def test_mat_unflatten(self):

        def make_args(m, mode):
            return mtx.mat_flatten(m, mode), mtx.mat_dim(m), mode

        a1 = [[1, 2, 3],
              [4, 5, 6]]
        a2 = [[1, 2, 3]]
        a3 = [[1], [2], [3]]
        a4 = [[1]]
        a5 = []
        # Test data
        test_data = {
            "Input 1": {
                "args": [*make_args(a1, (1, False))],
                "expect": a1
            },
            "Input 2": {
                "args": [*make_args(a1, (1, True))],
                "expect": a1
            },
            "Input 3": {
                "args": [*make_args(a1, (2, False))],
                "expect": a1
            },
            "Input 4": {
                "args": [*make_args(a1, (2, True))],
                "expect": a1
            },
            "Input 5": {
                "args": [*make_args(a2, (1, False))],
                "expect": a2
            },
            "Input 6": {
                "args": [*make_args(a2, (1, True))],
                "expect": a2
            },
            "Input 7": {
                "args": [*make_args(a2, (2, False))],
                "expect": a2
            },
            "Input 8": {
                "args": [*make_args(a2, (2, True))],
                "expect": a2
            },
            "Input 9": {
                "args": [*make_args(a3, (1, False))],
                "expect": a3
            },
            "Input 10": {
                "args": [*make_args(a3, (1, True))],
                "expect": a3
            },
            "Input 11": {
                "args": [*make_args(a3, (2, False))],
                "expect": a3
            },
            "Input 12": {
                "args": [*make_args(a3, (2, True))],
                "expect": a3
            },
            "Input 13": {
                "args": [*make_args(a4, (1, True))],
                "expect": a4
            },
            "Input 14": {
                "args": [*make_args(a4, (1, False))],
                "expect": a4
            },
            "Input 15": {
                "args": [*make_args(a4, (2, True))],
                "expect": a4
            },
            "Input 16": {
                "args": [*make_args(a4, (2, False))],
                "expect": a4
            },
            "Input 17": {
                "args": [*make_args(a5, (1, False))],
                "expect": a5
            },
            "Input 18": {
                "args": [*make_args(a5, (1, True))],
                "expect": a5
            },
            "Input 19": {
                "args": [*make_args(a5, (2, False))],
                "expect": a5
            },
            "Input 20": {
                "args": [*make_args(a5, (2, True))],
                "expect": a5
            },
            "Input 21": {
                "args": [[1, 2, 3], (2, 3)],
                "expect": "exception"
            },
            "Input 22": {
                "args": [[1, 2, 3], (3,)],
                "expect": "exception"
            },
            "Input 23": {
                "args": [[1, 2, 3], (3, 0)],
                "expect": "exception"
            },
            "Input 24": {
                "args": [[1, 2, 3], (3, 1), (3, True)],
                "expect": "exception"
            }
        }
        self.__do_test(mtx.mat_unflatten, test_data)

    def test_mat_pad(self):

        a1 = [[1, 2],
              [3, 4]]
        a2 = [[1, 2]]
        a3 = [[1], [2]]
        a4 = [[1]]
        a5 = []
        # Test data
        test_data = {
            "Input 1": {
                "args": [a1, (1, 1, 1, 1)],
                "expect": [[0, 0, 0, 0],
                           [0, 1, 2, 0],
                           [0, 3, 4, 0],
                           [0, 0, 0, 0]]
            },
            "Input 2": {
                "args": [a1, (1, 1, 1, 1), 3],
                "expect": [[3, 3, 3, 3],
                           [3, 1, 2, 3],
                           [3, 3, 4, 3],
                           [3, 3, 3, 3]]
            },
            "Input 3": {
                "args": [a1, (0, 1, 1, 1)],
                "expect": [[0, 1, 2, 0],
                           [0, 3, 4, 0],
                           [0, 0, 0, 0]]
            },
            "Input 4": {
                "args": [a1, (1, 0, 1, 1)],
                "expect": [[0, 0, 0, 0],
                           [0, 1, 2, 0],
                           [0, 3, 4, 0]]
            },
            "Input 5": {
                "args": [a1, (1, 1, 0, 1)],
                "expect": [[0, 0, 0],
                           [1, 2, 0],
                           [3, 4, 0],
                           [0, 0, 0]]
            },
            "Input 6": {
                "args": [a1, (1, 1, 1, 0)],
                "expect": [[0, 0, 0],
                           [0, 1, 2],
                           [0, 3, 4],
                           [0, 0, 0]]
            },
            "Input 7": {
                "args": [a1, (0, 0, 0, 0)],
                "expect": a1
            },
            "Input 8": {
                "args": [a2, (1, 1, 1, 1)],
                "expect": [[0, 0, 0, 0],
                           [0, 1, 2, 0],
                           [0, 0, 0, 0]]
            },
            "Input 9": {
                "args": [a2, (2, 0, 1, 0)],
                "expect": [[0, 0, 0],
                           [0, 0, 0],
                           [0, 1, 2]]
            },
            "Input 10": {
                "args": [a3, (1, 1, 1, 1)],
                "expect": [[0, 0, 0],
                           [0, 1, 0],
                           [0, 2, 0],
                           [0, 0, 0]]
            },
            "Input 11": {
                "args": [a3, (0, 0, 0, 0)],
                "expect": a3
            },
            "Input 12": {
                "args": [a4, (1, 1, 1, 1)],
                "expect": [[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]]
            },
            "Input 13": {
                "args": [a4, (0, 1, 0, 1)],
                "expect": [[1, 0],
                           [0, 0]]
            },
            "Input 14": {
                "args": [a4, (1, 0, 1, 0)],
                "expect": [[0, 0],
                           [0, 1]]
            },
            "Input 15": {
                "args": [a4, (0, 1, 1, 0)],
                "expect": [[0, 1],
                           [0, 0]]
            },
            "Input 16": {
                "args": [a5, (1, 0, 1, 0)],
                "expect": []
            },
            "Input 17": {
                "args": [a4, (1, 0, -1, 0)],
                "expect": "exception"
            },
            "Input 18": {
                "args": [a4, ()],
                "expect": "exception"
            },
        }
        self.__do_test(mtx.mat_pad, test_data)
