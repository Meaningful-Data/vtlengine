from development.tests.test_binary_comparison import *
from development.tests.test_binary_numeric import *
from development.tests.test_binary_general import *
from development.tests.test_unary_numeric import *

if __name__ == '__main__':
    print("Starting tests...")

    #Binary
    #Comparison
    test_equal()
    test_not_equal()
    test_greater()
    test_greater_equal()
    test_less()
    test_less_equal()

    #General
    test_membership()

    #Numeric
    test_binplus()
    test_binminus()
    test_mult()
    test_div()

    #Unary
    test_unplus()
    test_unminus()
    test_absolute()

    print("All tests passed!")