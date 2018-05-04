from misc.singleton import SingletonMetaClass


class AppState(metaclass=SingletonMetaClass):
    def __init__(self):
        self.visualize = False
