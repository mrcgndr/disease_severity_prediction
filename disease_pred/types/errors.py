from typing import Any, List, Type


class NoImageDataAvaliableError(Exception):
    def __init__(self):
        Exception.__init__(
            self,
            "The evaluator output object does not contain image data. Run the evaluation again with argument `save_raw_image` set to True.",
        )


class ValueNotUnderstoodError(Exception):
    def __init__(self, value_name: str, value: Any, all_values: List[Any]):
        Exception.__init__(self, f"{value_name} '{value}' not understood. Choose one of {all_values}.")


class OutOfTimePeriodError(Exception):
    def __init__(self):
        Exception.__init__(self, "The given date range is out of the availiable data time window.")


class NonNumericCoordinate(Exception):
    def __init__(self, coordinate_name: str, coordinate_value: Any):
        Exception.__init__(self, f"Coordinate '{coordinate_name}' has a non-numeric value '{coordinate_value}'.")


class WrongModuleTypeError(Exception):
    def __init__(self, name: str, wrong_type: Type, correct_type: Type):
        Exception.__init__(self, f"Module '{name}' has to be of tyoe '{correct_type}' but is of type '{wrong_type}'.")
