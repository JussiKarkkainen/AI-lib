from tests.test_backprop import BackPropTest
from tests.test_forward import ForwardTest

a = ForwardTest()
b = BackPropTest()

def test_main():
    a.test_binary_op()
    a.test_unary_op()
    a.test_tensor_op()
    b.test_backprop_simple()
#    b.test_backprop_harder()
#    b.test_backprop_diamond()

