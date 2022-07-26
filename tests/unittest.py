from test_backprop import BackPropTest


a = BackPropTest()

if __name__ == "__main__":
    print("Starting backprop testing")
    a.test_backprop_simple()
    a.test_backprop_harder()
    a.test_backprop_diamond()
    print("Finished test")
