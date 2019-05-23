import unittest

from udsp.zombies import mtx


class MatrixTestCase(unittest.TestCase):

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
            }
        }
        for i, inp_exp in test_data.items():
            args = inp_exp["args"]
            expected = inp_exp["expect"]
            emsg = "{} failed".format(i)
            if expected is "exception":
                with self.assertRaises(ValueError):
                    mtx.mat_submat_copy(*args)
            else:
                results = mtx.mat_submat_copy(*args)
                self.assertListEqual(results, expected, msg=emsg)

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
                "args": [a1, 1, False],
                "expect": [1, 2, 3, 4, 5, 6, 7, 8, 9]
            },
            "Input 2": {
                "args": [a1, 1, True],
                "expect": [7, 8, 9, 4, 5, 6, 1, 2, 3]
            },
            "Input 3": {
                "args": [a1, 2, False],
                "expect": [1, 4, 7, 2, 5, 8, 3, 6, 9]
            },
            "Input 4": {
                "args": [a1, 2, True],
                "expect": [3, 6, 9, 2, 5, 8, 1, 4, 7]
            },
            "Input 5": {
                "args": [a2, 1, False],
                "expect": [1, 2, 3]
            },
            "Input 6": {
                "args": [a2, 1, True],
                "expect": [1, 2, 3]
            },
            "Input 7": {
                "args": [a2, 2, False],
                "expect": [1, 2, 3]
            },
            "Input 8": {
                "args": [a2, 2, True],
                "expect": [3, 2, 1]
            },
            "Input 9": {
                "args": [a3, 1, False],
                "expect": [1, 2, 3]
            },
            "Input 10": {
                "args": [a3, 1, True],
                "expect": [3, 2, 1]
            },
            "Input 11": {
                "args": [a3, 2, False],
                "expect": [1, 2, 3]
            },
            "Input 12": {
                "args": [a3, 2, True],
                "expect": [1, 2, 3]
            },
            "Input 13": {
                "args": [a4, 1, True],
                "expect": [1]
            },
            "Input 14": {
                "args": [a4, 1, False],
                "expect": [1]
            },
            "Input 15": {
                "args": [a4, 2, True],
                "expect": [1]
            },
            "Input 16": {
                "args": [a4, 2, False],
                "expect": [1]
            },
            "Input 17": {
                "args": [a5, 1, False],
                "expect": []
            },
            "Input 18": {
                "args": [a5, 1, True],
                "expect": []
            },
            "Input 19": {
                "args": [a5, 2, False],
                "expect": []
            },
            "Input 20": {
                "args": [a5, 2, True],
                "expect": []
            },
            "Input 21": {
                "args": [a1, 3, False],
                "expect": "exception"
            }
        }
        for i, inp_exp in test_data.items():
            args = inp_exp["args"]
            expected = inp_exp["expect"]
            emsg = "{} failed".format(i)
            if expected is "exception":
                with self.assertRaises(ValueError):
                    mtx.mat_flatten(*args)
            else:
                results = mtx.mat_flatten(*args)
                self.assertListEqual(results, expected, msg=emsg)
