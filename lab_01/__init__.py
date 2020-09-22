from .funcs import constant_function
from .funcs import sum_function
from .funcs import prod_function
from .funcs import naive_poly_function
from .funcs import horner_method_function
from .funcs import bubble_sort
from .funcs import quick_sort
from .funcs import timsort
from .funcs import prod_matrix
from .funcs import vinograd_mod
from numpy import matmul

__all__ = [
    "constant_function", "sum_function", "prod_function",
    "naive_poly_function", "horner_method_function",
    "bubble_sort", "quick_sort", "timsort",
    "prod_matrix", "vinograd_mod", "matmul"
]
