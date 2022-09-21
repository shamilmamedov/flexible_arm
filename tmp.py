from abc import ABC, abstractmethod


class Base(ABC):
    def __init__(self):
        a = 1

    @abstractmethod
    def run(self, a):
        pass


class A(Base):
    def run(self, a):
        print("A:" + a)


class B(Base):
    def run(self, a):
        print("B:" + a)


class Safe:
    def post(self):
        print("safe")


def get_safe_controller(base_controller_class, safety_filter):
    class ControllerSafetyWrapper(base_controller_class):
        def __init__(self):
            super(ControllerSafetyWrapper, self).__init__()

        def run(self, a):
            super(ControllerSafetyWrapper, self).run(a)
            safety_filter.post()

    return ControllerSafetyWrapper


if __name__ == "__main__":
    class_a = A
    class_safe = get_safe_controller(A,Safe())
    obj_safe = class_safe()
    obj_safe.run("rudi")
