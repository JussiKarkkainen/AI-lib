from test_backprop import BackPropTest
from test_forward import ForwardTest

a = ForwardTest()
b = BackPropTest()

if __name__ == "__main__":
    print("Starting forward op testing\n")
    a.test_forward_all()
    print("Finished test\n")

    print("Starting backprop testing\n")
    b.test_backprop_simple()
#    b.test_backprop_harder()
#    b.test_backprop_diamond()
    print("Finished test\n")
