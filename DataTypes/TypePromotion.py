# implicit_type_promotion_dict = {
#     String: {String},
#     Number: {String, Number},
#     Integer: {String, Number, Integer},
#     TimeInterval: {String, TimeInterval},
#     Date: {String, TimeInterval, Date},
#     TimePeriod: {String, TimeInterval, TimePeriod},
#     Duration: {String, Duration},
#     Boolean: {String, Boolean},
#     Item: {String, Item},
#     Subcategory: {String, Subcategory},
#     Null: {String, Number, Integer, TimeInterval, Date, TimePeriod, Duration, Boolean, Item, Subcategory, Null}
# }
# TODO: 2 parameters come from the operator (we have to include in cls or something), 2 parameters come from the operands
# def binary_implicit_type_promotion(
#         left: ScalarType, right: ScalarType, op_type_to_check: ScalarType = None, return_type: ScalarType = None,
#         interval_allowed: bool = True, error_info: dict = None):
#     """
#     """
#     left_implicities = implicit_type_promotion_dict[left.__class__]
#     right_implicities = implicit_type_promotion_dict[right.__class__]

#     if left and right:
#         warning_raising = not (isinstance(left, type(right)) or isinstance(right, type(left)))
#     else:
#         warning_raising = False

#     if op_type_to_check:

#         if op_type_to_check.is_included(
#                 left_implicities.intersection(right_implicities)):  # general case and date->str and boolean-> str in the str operator

#             if warning_raising:
#                 warnings.warn(
#                     f"Implicit promotion between {left} and {right} and op_type={op_type_to_check}.")
#             if return_type:
#                 binary_check_interval(result_operand=return_type, left_operand=left, right_operand=right, op_type_to_check=op_type_to_check,
#                                       return_type=return_type, interval_allowed=interval_allowed, error_info=error_info)
#                 return return_type

#             if not left.is_null_type() and left.is_included(right_implicities):
#                 binary_check_interval(result_operand=left, left_operand=left, right_operand=right, op_type_to_check=op_type_to_check,
#                                       return_type=return_type, interval_allowed=interval_allowed, error_info=error_info)
#                 return left
#             elif not right.is_null_type() and right.is_included(left_implicities):
#                 binary_check_interval(result_operand=right, left_operand=left, right_operand=right, op_type_to_check=op_type_to_check,
#                                       return_type=return_type, interval_allowed=interval_allowed, error_info=error_info)
#                 return right
#             else:
#                 if isinstance(op_type_to_check, Number):
#                     binary_check_interval(result_operand=op_type_to_check, left_operand=left, right_operand=right,
#                                           op_type_to_check=op_type_to_check, return_type=return_type, interval_allowed=interval_allowed,
#                                           error_info=error_info)
#                 return op_type_to_check

#         else:
#             origin = None if error_info is None else "operator={operator} {left} {right}".format(left=error_info['left_name'],
#                                                                                                  operator=error_info['op'],
#                                                                                                  right=error_info['right_name'])
#             raise SemanticError("3-2", type_1=left, type_2=right, type_op=op_type_to_check, origin=origin)
#     else:
#         if warning_raising:
#             warnings.warn(f"Implicit promotion between {left} and {right}.")
#         if return_type and (left.is_included(right_implicities) or right.is_included(left_implicities)):
#             return return_type
#         elif left.is_included(right_implicities):
#             return left
#         elif right.is_included(left_implicities):
#             return right
#         else:
#             origin = None if error_info is None else "operator={operator} {left} {right}".format(left=error_info['left_name'],
#                                                                                                  operator=error_info['op'],
#                                                                                                  right=error_info['right_name'])
#             raise SemanticError("3-1", type_1=left, type_2=right, origin=origin)

# def unary_implicit_type_promotion(operand: ScalarType, op_type_to_check: ScalarType = None, return_type: ScalarType = None,
#                                   interval_allowed: bool = True, error_info: dict = None):
#     """
#     """

#     operand_implicities = implicit_type_promotion_dict[operand.__class__]

#     unary_check_interval(operand=operand, op_type_to_check=op_type_to_check, return_type=return_type, interval_allowed=interval_allowed,
#                          error_info=error_info)

#     if op_type_to_check:
#         if not op_type_to_check.is_included(operand_implicities):
#             origin = None if error_info is None else "{}({})".format(error_info['op'], error_info['operand_name'])
#             raise SemanticError("3-3", type_1=operand, type_op=op_type_to_check, origin=origin)

#     if return_type:
#         return return_type
#     if op_type_to_check and not operand.is_subtype(op_type_to_check):
#         return op_type_to_check
#     return operand