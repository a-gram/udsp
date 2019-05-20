import unittest

# Run from the root dir with >python ./tests/run.py
if __name__ == "__main__":
    tests = unittest.defaultTestLoader.discover("./tests")
    runner = unittest.TextTestRunner()
    runner.run(tests)
