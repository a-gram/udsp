import unittest

from udsp.core import mtx


class MatrixTestCase(unittest.TestCase):

    def __do_test(self, test, test_data):
        """
        Private refactored method to execute the test cases

        Parameters
        ----------
        test: callable
            The test case function to be executed
        test_data: dict
            A dictionary with the test data for the case. Each entry
            in the dictionary represents an input to the test case and
            has the following layout:

            "<input_name>": {
                                "args": <list> | <dict>,
                                "expect": <object>,
                                ["assert"]: <function>,
                                ["assert_params"]: <dict>
                            }

            where "<input_name>" is a string indicating the name of the
            input, "args" is a list or dictionary holding the input
            values to the test (the arguments to the tested function/
            method), "expect" is an object indicating the expected
            results, "assert" an optional assertion function to be
            executed to check the results and "assert_params" an
            optional sequence of parameters (in a dict) to be injected
            into the assertion function.
            Note that not all assertion functions can be used with this
            method, but the most common can. Particularly, only those
            that require at least two positional arguments of the type
            (results, expected) are acceptable. If none is specified
            than a default one is used.

        """
        def is_except_type(t):
            """Checks whether a type is an exception"""
            return type(t) is type and issubclass(t, BaseException)

        def test_case():
            """Executes the case with appropriately expanded args"""
            if isinstance(args, list):
                return test(*args)
            elif isinstance(args, dict):
                return test(**args)

        for i, inp in test_data.items():

            args = inp["args"]
            expected = inp["expect"]
            assrt = inp["assert"] if "assert" in inp else None
            assrt_pms = (inp["assert_params"]
                         if "assert_params" in inp else {})
            emsg = "{} failed".format(i)

            # Default assertion
            assertion = assrt or self.assertListEqual

            try:
                if is_except_type(expected):
                    with self.assertRaises(expected, msg=emsg):
                        test_case()
                else:
                    results = test_case()
                    assertion(results, expected, msg=emsg, **assrt_pms)
            # Catch exceptions not covered by the tests and print
            # info to pinpoint the failing case, otherwise a bit obscure
            # since this code is not inline with the tests.
            except Exception as exc:
                if not isinstance(exc, AssertionError):
                    # print("\n\nError in {}\nmsg: {}\nargs: {}\n".
                    print("\n\nError in test case '{}'\n"
                          "  test: {}\n"
                          "  input: {}\n"
                          "  msg: {}\n"
                          "  args: {}\n".
                          format(self.__class__.__name__,
                                 test.__name__,
                                 i, exc, args))
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
                "expect": ValueError
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
                "expect": ValueError
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
                "expect": ValueError
            },
            "Input 22": {
                "args": [a1, (3,)],
                "expect": ValueError
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
                "expect": ValueError
            },
            "Input 22": {
                "args": [[1, 2, 3], (3,)],
                "expect": ValueError
            },
            "Input 23": {
                "args": [[1, 2, 3], (3, 0)],
                "expect": ValueError
            },
            "Input 24": {
                "args": [[1, 2, 3], (3, 1), (3, True)],
                "expect": ValueError
            }
        }
        self.__do_test(mtx.mat_unflatten, test_data)

    def test_mat_extend(self):

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
                "expect": ValueError
            },
            "Input 18": {
                "args": [a4, ()],
                "expect": ValueError
            },
        }
        self.__do_test(mtx.mat_extend, test_data)

    def test_mat_bin(self):

        a1 = [[1, -3,  0],
              [2, -1,  5],
              [0,  4, -3]]
        a2 = [[1, 3, 1],
              [1, 2, 3],
              [2, 2, 2]]

        def sum_all(e):
            return 0, e

        def sum_positive_negative(e):
            return e < 0, e

        def sum_2_ranges(e):
            return e > 2, e

        def sum_3_ranges(e):
            return (0 if -5 <= e < -2 else
                   (1 if -2 <= e < 2 else 2), e)

        def count_zeroes(e):
            return 0 if e == 0 else None, 1

        def histo(e):
            return e - 1, 1

        # Test data
        test_data = {
            "Input 1": {
                "args": [a1, 1, sum_all],
                "expect": [5]
            },
            "Input 2": {
                "args": [a1, 2, sum_positive_negative],
                "expect": [12, -7]
            },
            "Input 3": {
                "args": [a1, 2, sum_2_ranges],
                "expect": [-4, 9]
            },
            "Input 4": {
                "args": [a1, 3, sum_3_ranges],
                "expect": [-6, 0, 11]
            },
            "Input 5": {
                "args": [a1, 1, count_zeroes],
                "expect": [2]
            },
            "Input 6": {
                "args": [a2, 3, histo],
                "expect": [3, 4, 2]
            },
            "Input 7": {
                "args": [a1, 1, sum_all, 5],
                "expect": [10]
            },
            "Input 8": {
                "args": [[], 2, sum_all],
                "expect": [0, 0]
            }
        }
        self.__do_test(mtx.mat_bin, test_data)

    def test_vec_extend(self):

        a1 = [1, 2, 3, 4]
        a2 = [1]
        a3 = []
        a4 = [0.25, 0.50, 0.75, 1]
        # Test data
        test_data = {
            "Input 1": {
                "args": [a1, (1, 1)],
                "expect": [0] + a1 + [0]
            },
            "Input 2": {
                "args": [a1, (1, 1), 3],
                "expect": [3] + a1 + [3]
            },
            "Input 3": {
                "args": [a1, (0, 1)],
                "expect": a1 + [0]
            },
            "Input 4": {
                "args": [a1, (1, 0)],
                "expect": [0] + a1
            },
            "Input 5": {
                "args": [a1, (3, 2)],
                "expect": [0, 0, 0] + a1 + [0, 0]
            },
            "Input 6": {
                "args": [a1, (0, 0)],
                "expect": a1
            },
            "Input 7": {
                "args": [a2, (1, 1)],
                "expect": [0] + a2 + [0]
            },
            "Input 8": {
                "args": [a2, (2, 0)],
                "expect":  [0, 0] + a2
            },
            "Input 9": {
                "args": [a2, (0, 1)],
                "expect":  a2 + [0]
            },
            "Input 10": {
                "args": [a2, (2, 3)],
                "expect":  [0, 0] + a2 + [0, 0, 0]
            },
            "Input 11": {
                "args": [a2, (0, 0)],
                "expect": a2
            },
            "Input 12": {
                "args": [a3, (1, 1)],
                "expect": a3
            },
            "Input 13": {
                "args": {"v": a4,
                         "ext": (0, 0),
                         "mode": "range",
                         "ds": 0.25},
                "expect": a4
            },
            "Input 14": {
                "args": {"v": a4,
                         "ext": (0, 1),
                         "mode": "range",
                         "ds": 0.25},
                "expect": a4 + [1.25]
            },
            "Input 15": {
                "args": {"v": a4,
                         "ext": (1, 0),
                         "mode": "range",
                         "ds": 0.25},
                "expect": [0] + a4
            },
            "Input 16": {
                "args": {"v": a4,
                         "ext": (3, 2),
                         "mode": "range",
                         "ds": 0.25},
                "expect": [-0.5, -0.25, 0] + a4 + [1.25, 1.5]
            },
            "Input 17": {
                "args": {"v": a4,
                         "ext": (3, 2),
                         "mode": "strange",
                         "ds": 0.25},
                "expect": ValueError
            },
            "Input 18": {
                "args": [a4, (3, 2), None, "range"],
                "expect": KeyError
            },
            "Input 19": {
                "args": [a4, (-1, 0)],
                "expect": ValueError
            },
            "Input 20": {
                "args": [a4, ()],
                "expect": ValueError
            },
        }
        self.__do_test(mtx.vec_extend, test_data)

    def test_vec_bin(self):

        a1 = [1, -3, 0, 2, -1, 5, 0, 4, -3]
        a2 = [1, 3, 1, 1, 2, 3, 2, 2, 2]

        def sum_all(e):
            return 0, e

        def sum_positive_negative(e):
            return e < 0, e

        def sum_2_ranges(e):
            return e > 2, e

        def sum_3_ranges(e):
            return (0 if -5 <= e < -2 else
                   (1 if -2 <= e < 2 else 2), e)

        def count_zeroes(e):
            return 0 if e == 0 else None, 1

        def histo(e):
            return e - 1, 1

        # Test data
        test_data = {
            "Input 1": {
                "args": [a1, 1, sum_all],
                "expect": [5]
            },
            "Input 2": {
                "args": [a1, 2, sum_positive_negative],
                "expect": [12, -7]
            },
            "Input 3": {
                "args": [a1, 2, sum_2_ranges],
                "expect": [-4, 9]
            },
            "Input 4": {
                "args": [a1, 3, sum_3_ranges],
                "expect": [-6, 0, 11]
            },
            "Input 5": {
                "args": [a1, 1, count_zeroes],
                "expect": [2]
            },
            "Input 6": {
                "args": [a2, 3, histo],
                "expect": [3, 4, 2]
            },
            "Input 7": {
                "args": [a1, 1, sum_all, 5],
                "expect": [10]
            },
            "Input 8": {
                "args": [[], 2, sum_all],
                "expect": [0, 0]
            }
        }
        self.__do_test(mtx.vec_bin, test_data)
