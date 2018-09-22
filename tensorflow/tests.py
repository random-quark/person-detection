import unittest


def fun(a):
    return 4


class MyTest(unittest.TestCase):
    def test(self):
        self.assertEqual(fun(3), 4)


if __name__ == '__main__':
    unittest.main()
