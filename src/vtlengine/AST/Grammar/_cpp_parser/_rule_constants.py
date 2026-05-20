"""Constants mapping (rule_index, alt_index) tuples for use with the C++ parser.

Each constant is a tuple of (rule_index, alt_index) where:
- rule_index corresponds to Vtl::Rule* enum values
- alt_index is the labeled alternative index, or -1 for rules without alternatives

Generated from the type map in bindings.cpp and rule indices from Vtl.h.
"""

from typing import Final

RuleConstant = tuple[int, int]

# =============================================================================
# Rules WITHOUT labeled alternatives (alt_index = -1)
# =============================================================================

# RuleStart = 0
START: Final[RuleConstant] = (0, -1)

# RuleDatasetClause = 6
DATASET_CLAUSE: Final[RuleConstant] = (6, -1)

# RuleRenameClause = 7
RENAME_CLAUSE: Final[RuleConstant] = (7, -1)

# RuleAggrClause = 8
AGGR_CLAUSE: Final[RuleConstant] = (8, -1)

# RuleFilterClause = 9
FILTER_CLAUSE: Final[RuleConstant] = (9, -1)

# RuleCalcClause = 10
CALC_CLAUSE: Final[RuleConstant] = (10, -1)

# RuleKeepOrDropClause = 11
KEEP_OR_DROP_CLAUSE: Final[RuleConstant] = (11, -1)

# RulePivotOrUnpivotClause = 12
PIVOT_OR_UNPIVOT_CLAUSE: Final[RuleConstant] = (12, -1)

# RuleCustomPivotClause = 13
CUSTOM_PIVOT_CLAUSE: Final[RuleConstant] = (13, -1)

# RuleSubspaceClause = 14
SUBSPACE_CLAUSE: Final[RuleConstant] = (14, -1)

# RuleVpSignature = 17
VP_SIGNATURE: Final[RuleConstant] = (17, -1)

# RuleVpBody = 18
VP_BODY: Final[RuleConstant] = (18, -1)

# RuleVpCondition = 20
VP_CONDITION: Final[RuleConstant] = (20, -1)

# RuleParameterComponent = 23
PARAMETER_COMPONENT: Final[RuleConstant] = (23, -1)

# RuleParameter = 24
PARAMETER: Final[RuleConstant] = (24, -1)

# RuleStringDistanceMethods = 25
STRING_DISTANCE_METHODS: Final[RuleConstant] = (25, -1)

# RuleHierarchyOperators = 35
HIERARCHY_OPERATORS: Final[RuleConstant] = (35, -1)

# RuleRenameClauseItem = 43
RENAME_CLAUSE_ITEM: Final[RuleConstant] = (43, -1)

# RuleAggregateClause = 44
AGGREGATE_CLAUSE: Final[RuleConstant] = (44, -1)

# RuleAggrFunctionClause = 45
AGGR_FUNCTION_CLAUSE: Final[RuleConstant] = (45, -1)

# RuleCalcClauseItem = 46
CALC_CLAUSE_ITEM: Final[RuleConstant] = (46, -1)

# RuleSubspaceClauseItem = 47
SUBSPACE_CLAUSE_ITEM: Final[RuleConstant] = (47, -1)

# RuleJoinClause = 49
JOIN_CLAUSE: Final[RuleConstant] = (49, -1)

# RuleJoinClauseItem = 50
JOIN_CLAUSE_ITEM: Final[RuleConstant] = (50, -1)

# RuleUsingClause = 51
USING_CLAUSE: Final[RuleConstant] = (51, -1)

# RuleNvlJoinClause = 52
NVL_JOIN_CLAUSE: Final[RuleConstant] = (52, -1)

# RuleJoinBody = 53
JOIN_BODY: Final[RuleConstant] = (53, -1)

# RuleJoinApplyClause = 54
JOIN_APPLY_CLAUSE: Final[RuleConstant] = (54, -1)

# RuleOrderByClause = 56
ORDER_BY_CLAUSE: Final[RuleConstant] = (56, -1)

# RuleOrderByItem = 57
ORDER_BY_ITEM: Final[RuleConstant] = (57, -1)

# RuleWindowingClause = 58
WINDOWING_CLAUSE: Final[RuleConstant] = (58, -1)

# RuleSignedInteger = 59
SIGNED_INTEGER: Final[RuleConstant] = (59, -1)

# RuleSignedNumber = 60
SIGNED_NUMBER: Final[RuleConstant] = (60, -1)

# RuleLimitClauseItem = 61
LIMIT_CLAUSE_ITEM: Final[RuleConstant] = (61, -1)

# RuleHavingClause = 63
HAVING_CLAUSE: Final[RuleConstant] = (63, -1)

# RuleParameterItem = 64
PARAMETER_ITEM: Final[RuleConstant] = (64, -1)

# RuleOutputParameterType = 65
OUTPUT_PARAMETER_TYPE: Final[RuleConstant] = (65, -1)

# RuleOutputParameterTypeComponent = 66
OUTPUT_PARAMETER_TYPE_COMPONENT: Final[RuleConstant] = (66, -1)

# RuleInputParameterType = 67
INPUT_PARAMETER_TYPE: Final[RuleConstant] = (67, -1)

# RuleRulesetType = 68
RULESET_TYPE: Final[RuleConstant] = (68, -1)

# RuleScalarType = 69
SCALAR_TYPE: Final[RuleConstant] = (69, -1)

# RuleComponentType = 70
COMPONENT_TYPE: Final[RuleConstant] = (70, -1)

# RuleDatasetType = 71
DATASET_TYPE: Final[RuleConstant] = (71, -1)

# RuleEvalDatasetType = 72
EVAL_DATASET_TYPE: Final[RuleConstant] = (72, -1)

# RuleScalarSetType = 73
SCALAR_SET_TYPE: Final[RuleConstant] = (73, -1)

# RuleValueDomainName = 76
VALUE_DOMAIN_NAME: Final[RuleConstant] = (76, -1)

# RuleRulesetID = 77
RULESET_ID: Final[RuleConstant] = (77, -1)

# RuleRulesetSignature = 78
RULESET_SIGNATURE: Final[RuleConstant] = (78, -1)

# RuleSignature = 79
SIGNATURE: Final[RuleConstant] = (79, -1)

# RuleRuleClauseDatapoint = 80
RULE_CLAUSE_DATAPOINT: Final[RuleConstant] = (80, -1)

# RuleRuleItemDatapoint = 81
RULE_ITEM_DATAPOINT: Final[RuleConstant] = (81, -1)

# RuleRuleClauseHierarchical = 82
RULE_CLAUSE_HIERARCHICAL: Final[RuleConstant] = (82, -1)

# RuleRuleItemHierarchical = 83
RULE_ITEM_HIERARCHICAL: Final[RuleConstant] = (83, -1)

# RuleHierRuleSignature = 84
HIER_RULE_SIGNATURE: Final[RuleConstant] = (84, -1)

# RuleValueDomainSignature = 85
VALUE_DOMAIN_SIGNATURE: Final[RuleConstant] = (85, -1)

# RuleCodeItemRelation = 86
CODE_ITEM_RELATION: Final[RuleConstant] = (86, -1)

# RuleCodeItemRelationClause = 87
CODE_ITEM_RELATION_CLAUSE: Final[RuleConstant] = (87, -1)

# RuleValueDomainValue = 88
VALUE_DOMAIN_VALUE: Final[RuleConstant] = (88, -1)

# RuleCompConstraint = 90
COMP_CONSTRAINT: Final[RuleConstant] = (90, -1)

# RuleMultModifier = 91
MULT_MODIFIER: Final[RuleConstant] = (91, -1)

# RuleValidationOutput = 92
VALIDATION_OUTPUT: Final[RuleConstant] = (92, -1)

# RuleValidationMode = 93
VALIDATION_MODE: Final[RuleConstant] = (93, -1)

# RuleConditionClause = 94
CONDITION_CLAUSE: Final[RuleConstant] = (94, -1)

# RuleInputMode = 95
INPUT_MODE: Final[RuleConstant] = (95, -1)

# RuleImbalanceExpr = 96
IMBALANCE_EXPR: Final[RuleConstant] = (96, -1)

# RuleInputModeHierarchy = 97
INPUT_MODE_HIERARCHY: Final[RuleConstant] = (97, -1)

# RuleOutputModeHierarchy = 98
OUTPUT_MODE_HIERARCHY: Final[RuleConstant] = (98, -1)

# RuleAlias = 99
ALIAS: Final[RuleConstant] = (99, -1)

# RuleVarID = 100
VAR_ID: Final[RuleConstant] = (100, -1)

# RuleSimpleComponentId = 101
SIMPLE_COMPONENT_ID: Final[RuleConstant] = (101, -1)

# RuleComponentID = 102
COMPONENT_ID: Final[RuleConstant] = (102, -1)

# RuleLists = 103
LISTS: Final[RuleConstant] = (103, -1)

# RuleErCode = 104
ER_CODE: Final[RuleConstant] = (104, -1)

# RuleErLevel = 105
ER_LEVEL: Final[RuleConstant] = (105, -1)

# RuleComparisonOperand = 106
COMPARISON_OPERAND: Final[RuleConstant] = (106, -1)

# RuleOptionalExpr = 107
OPTIONAL_EXPR: Final[RuleConstant] = (107, -1)

# RuleOptionalExprComponent = 108
OPTIONAL_EXPR_COMPONENT: Final[RuleConstant] = (108, -1)

# RuleComponentRole = 109
COMPONENT_ROLE: Final[RuleConstant] = (109, -1)

# RuleViralAttribute = 110
VIRAL_ATTRIBUTE: Final[RuleConstant] = (110, -1)

# RuleValueDomainID = 111
VALUE_DOMAIN_ID: Final[RuleConstant] = (111, -1)

# RuleOperatorID = 112
OPERATOR_ID: Final[RuleConstant] = (112, -1)

# RuleRoutineName = 113
ROUTINE_NAME: Final[RuleConstant] = (113, -1)

# RuleBasicScalarType = 115
BASIC_SCALAR_TYPE: Final[RuleConstant] = (115, -1)

# RuleRetainType = 116
RETAIN_TYPE: Final[RuleConstant] = (116, -1)

# =============================================================================
# Rules WITH labeled alternatives
# =============================================================================

# --- RuleStatement = 1 ---
STATEMENT: Final[RuleConstant] = (1, -1)
TEMPORARY_ASSIGNMENT: Final[RuleConstant] = (1, 0)
PERSIST_ASSIGNMENT: Final[RuleConstant] = (1, 1)
DEFINE_EXPRESSION: Final[RuleConstant] = (1, 2)

# --- RuleExpr = 2 ---
EXPR: Final[RuleConstant] = (2, -1)
PARENTHESIS_EXPR: Final[RuleConstant] = (2, 0)
FUNCTIONS_EXPRESSION: Final[RuleConstant] = (2, 1)
CLAUSE_EXPR: Final[RuleConstant] = (2, 2)
MEMBERSHIP_EXPR: Final[RuleConstant] = (2, 3)
UNARY_EXPR: Final[RuleConstant] = (2, 4)
ARITHMETIC_EXPR: Final[RuleConstant] = (2, 5)
ARITHMETIC_EXPR_OR_CONCAT: Final[RuleConstant] = (2, 6)
COMPARISON_EXPR: Final[RuleConstant] = (2, 7)
IN_NOT_IN_EXPR: Final[RuleConstant] = (2, 8)
BOOLEAN_EXPR: Final[RuleConstant] = (2, 9)
IF_EXPR: Final[RuleConstant] = (2, 10)
CASE_EXPR: Final[RuleConstant] = (2, 11)
CONSTANT_EXPR: Final[RuleConstant] = (2, 12)
VAR_ID_EXPR: Final[RuleConstant] = (2, 13)

# --- RuleExprComponent = 3 ---
EXPR_COMPONENT: Final[RuleConstant] = (3, -1)
PARENTHESIS_EXPR_COMP: Final[RuleConstant] = (3, 0)
FUNCTIONS_EXPRESSION_COMP: Final[RuleConstant] = (3, 1)
UNARY_EXPR_COMP: Final[RuleConstant] = (3, 2)
ARITHMETIC_EXPR_COMP: Final[RuleConstant] = (3, 3)
ARITHMETIC_EXPR_OR_CONCAT_COMP: Final[RuleConstant] = (3, 4)
COMPARISON_EXPR_COMP: Final[RuleConstant] = (3, 5)
IN_NOT_IN_EXPR_COMP: Final[RuleConstant] = (3, 6)
BOOLEAN_EXPR_COMP: Final[RuleConstant] = (3, 7)
IF_EXPR_COMP: Final[RuleConstant] = (3, 8)
CASE_EXPR_COMP: Final[RuleConstant] = (3, 9)
CONSTANT_EXPR_COMP: Final[RuleConstant] = (3, 10)
COMP_ID: Final[RuleConstant] = (3, 11)

# --- RuleFunctionsComponents = 4 ---
GENERIC_FUNCTIONS_COMPONENTS: Final[RuleConstant] = (4, 0)
STRING_FUNCTIONS_COMPONENTS: Final[RuleConstant] = (4, 1)
NUMERIC_FUNCTIONS_COMPONENTS: Final[RuleConstant] = (4, 2)
COMPARISON_FUNCTIONS_COMPONENTS: Final[RuleConstant] = (4, 3)
TIME_FUNCTIONS_COMPONENTS: Final[RuleConstant] = (4, 4)
CONDITIONAL_FUNCTIONS_COMPONENTS: Final[RuleConstant] = (4, 5)
AGGREGATE_FUNCTIONS_COMPONENTS: Final[RuleConstant] = (4, 6)
ANALYTIC_FUNCTIONS_COMPONENTS: Final[RuleConstant] = (4, 7)

# --- RuleFunctions = 5 ---
JOIN_FUNCTIONS: Final[RuleConstant] = (5, 0)
GENERIC_FUNCTIONS: Final[RuleConstant] = (5, 1)
STRING_FUNCTIONS: Final[RuleConstant] = (5, 2)
NUMERIC_FUNCTIONS: Final[RuleConstant] = (5, 3)
COMPARISON_FUNCTIONS: Final[RuleConstant] = (5, 4)
TIME_FUNCTIONS: Final[RuleConstant] = (5, 5)
SET_FUNCTIONS: Final[RuleConstant] = (5, 6)
HIERARCHY_FUNCTIONS: Final[RuleConstant] = (5, 7)
VALIDATION_FUNCTIONS: Final[RuleConstant] = (5, 8)
CONDITIONAL_FUNCTIONS: Final[RuleConstant] = (5, 9)
AGGREGATE_FUNCTIONS: Final[RuleConstant] = (5, 10)
ANALYTIC_FUNCTIONS: Final[RuleConstant] = (5, 11)

# --- RuleJoinOperators = 15 ---
INNER_JOIN_EXPR: Final[RuleConstant] = (15, 0)
LEFT_JOIN_EXPR: Final[RuleConstant] = (15, 1)
FULL_JOIN_EXPR: Final[RuleConstant] = (15, 2)
CROSS_JOIN_EXPR: Final[RuleConstant] = (15, 3)

# --- RuleDefOperators = 16 ---
DEF_OPERATOR: Final[RuleConstant] = (16, 0)
DEF_DATAPOINT_RULESET: Final[RuleConstant] = (16, 1)
DEF_HIERARCHICAL: Final[RuleConstant] = (16, 2)
DEF_VIRAL_PROPAGATION: Final[RuleConstant] = (16, 3)

# --- RuleVpClause = 19 ---
ENUMERATED_VP_CLAUSE: Final[RuleConstant] = (19, 0)
AGGREGATION_VP_CLAUSE: Final[RuleConstant] = (19, 1)
DEFAULT_VP_CLAUSE: Final[RuleConstant] = (19, 2)

# --- RuleGenericOperators = 21 ---
CALL_DATASET: Final[RuleConstant] = (21, 0)
EVAL_ATOM: Final[RuleConstant] = (21, 1)
CAST_EXPR_DATASET: Final[RuleConstant] = (21, 2)

# --- RuleGenericOperatorsComponent = 22 ---
CALL_COMPONENT: Final[RuleConstant] = (22, 0)
CAST_EXPR_COMPONENT: Final[RuleConstant] = (22, 1)
EVAL_ATOM_COMPONENT: Final[RuleConstant] = (22, 2)

# --- RuleStringOperators = 26 ---
UNARY_STRING_FUNCTION: Final[RuleConstant] = (26, 0)
SUBSTR_ATOM: Final[RuleConstant] = (26, 1)
REPLACE_ATOM: Final[RuleConstant] = (26, 2)
INSTR_ATOM: Final[RuleConstant] = (26, 3)
STRING_DISTANCE_ATOM: Final[RuleConstant] = (26, 4)

# --- RuleStringOperatorsComponent = 27 ---
UNARY_STRING_FUNCTION_COMPONENT: Final[RuleConstant] = (27, 0)
SUBSTR_ATOM_COMPONENT: Final[RuleConstant] = (27, 1)
REPLACE_ATOM_COMPONENT: Final[RuleConstant] = (27, 2)
INSTR_ATOM_COMPONENT: Final[RuleConstant] = (27, 3)
STRING_DISTANCE_ATOM_COMPONENT: Final[RuleConstant] = (27, 4)

# --- RuleNumericOperators = 28 ---
UNARY_NUMERIC: Final[RuleConstant] = (28, 0)
UNARY_WITH_OPTIONAL_NUMERIC: Final[RuleConstant] = (28, 1)
BINARY_NUMERIC: Final[RuleConstant] = (28, 2)

# --- RuleNumericOperatorsComponent = 29 ---
UNARY_NUMERIC_COMPONENT: Final[RuleConstant] = (29, 0)
UNARY_WITH_OPTIONAL_NUMERIC_COMPONENT: Final[RuleConstant] = (29, 1)
BINARY_NUMERIC_COMPONENT: Final[RuleConstant] = (29, 2)

# --- RuleComparisonOperators = 30 ---
BETWEEN_ATOM: Final[RuleConstant] = (30, 0)
CHARSET_MATCH_ATOM: Final[RuleConstant] = (30, 1)
IS_NULL_ATOM: Final[RuleConstant] = (30, 2)
EXIST_IN_ATOM: Final[RuleConstant] = (30, 3)

# --- RuleComparisonOperatorsComponent = 31 ---
BETWEEN_ATOM_COMPONENT: Final[RuleConstant] = (31, 0)
CHARSET_MATCH_ATOM_COMPONENT: Final[RuleConstant] = (31, 1)
IS_NULL_ATOM_COMPONENT: Final[RuleConstant] = (31, 2)

# --- RuleTimeOperators = 32 ---
PERIOD_ATOM: Final[RuleConstant] = (32, 0)
FILL_TIME_ATOM: Final[RuleConstant] = (32, 1)
FLOW_ATOM: Final[RuleConstant] = (32, 2)
TIME_SHIFT_ATOM: Final[RuleConstant] = (32, 3)
TIME_AGG_ATOM: Final[RuleConstant] = (32, 4)
CURRENT_DATE_ATOM: Final[RuleConstant] = (32, 5)
DATE_DIFF_ATOM: Final[RuleConstant] = (32, 6)
DATE_ADD_ATOM: Final[RuleConstant] = (32, 7)
YEAR_ATOM: Final[RuleConstant] = (32, 8)
MONTH_ATOM: Final[RuleConstant] = (32, 9)
DAY_OF_MONTH_ATOM: Final[RuleConstant] = (32, 10)
DAY_OF_YEAR_ATOM: Final[RuleConstant] = (32, 11)
DAY_TO_YEAR_ATOM: Final[RuleConstant] = (32, 12)
DAY_TO_MONTH_ATOM: Final[RuleConstant] = (32, 13)
YEAR_TODAY_ATOM: Final[RuleConstant] = (32, 14)
MONTH_TODAY_ATOM: Final[RuleConstant] = (32, 15)

# --- RuleTimeOperatorsComponent = 33 ---
PERIOD_ATOM_COMPONENT: Final[RuleConstant] = (33, 0)
FILL_TIME_ATOM_COMPONENT: Final[RuleConstant] = (33, 1)
FLOW_ATOM_COMPONENT: Final[RuleConstant] = (33, 2)
TIME_SHIFT_ATOM_COMPONENT: Final[RuleConstant] = (33, 3)
TIME_AGG_ATOM_COMPONENT: Final[RuleConstant] = (33, 4)
CURRENT_DATE_ATOM_COMPONENT: Final[RuleConstant] = (33, 5)
DATE_DIFF_ATOM_COMPONENT: Final[RuleConstant] = (33, 6)
DATE_ADD_ATOM_COMPONENT: Final[RuleConstant] = (33, 7)
YEAR_ATOM_COMPONENT: Final[RuleConstant] = (33, 8)
MONTH_ATOM_COMPONENT: Final[RuleConstant] = (33, 9)
DAY_OF_MONTH_ATOM_COMPONENT: Final[RuleConstant] = (33, 10)
DAY_OF_YEAR_ATOM_COMPONENT: Final[RuleConstant] = (33, 11)
DAY_TO_YEAR_ATOM_COMPONENT: Final[RuleConstant] = (33, 12)
DAY_TO_MONTH_ATOM_COMPONENT: Final[RuleConstant] = (33, 13)
YEAR_TODAY_ATOM_COMPONENT: Final[RuleConstant] = (33, 14)
MONTH_TODAY_ATOM_COMPONENT: Final[RuleConstant] = (33, 15)

# --- RuleSetOperators = 34 ---
UNION_ATOM: Final[RuleConstant] = (34, 0)
INTERSECT_ATOM: Final[RuleConstant] = (34, 1)
SET_OR_SYM_DIFF_ATOM: Final[RuleConstant] = (34, 2)

# --- RuleValidationOperators = 36 ---
VALIDATE_D_PRULESET: Final[RuleConstant] = (36, 0)
VALIDATE_HR_RULESET: Final[RuleConstant] = (36, 1)
VALIDATION_SIMPLE: Final[RuleConstant] = (36, 2)

# --- RuleConditionalOperators = 37 ---
NVL_ATOM: Final[RuleConstant] = (37, 0)

# --- RuleConditionalOperatorsComponent = 38 ---
NVL_ATOM_COMPONENT: Final[RuleConstant] = (38, 0)

# --- RuleAggrOperators = 39 ---
AGGR_COMP: Final[RuleConstant] = (39, 0)
COUNT_AGGR_COMP: Final[RuleConstant] = (39, 1)

# --- RuleAggrOperatorsGrouping = 40 ---
AGGR_DATASET: Final[RuleConstant] = (40, 0)

# --- RuleAnFunction = 41 ---
AN_SIMPLE_FUNCTION: Final[RuleConstant] = (41, 0)
LAG_OR_LEAD_AN: Final[RuleConstant] = (41, 1)
RATIO_TO_REPORT_AN: Final[RuleConstant] = (41, 2)

# --- RuleAnFunctionComponent = 42 ---
AN_SIMPLE_FUNCTION_COMPONENT: Final[RuleConstant] = (42, 0)
LAG_OR_LEAD_AN_COMPONENT: Final[RuleConstant] = (42, 1)
RANK_AN_COMPONENT: Final[RuleConstant] = (42, 2)
RATIO_TO_REPORT_AN_COMPONENT: Final[RuleConstant] = (42, 3)

# --- RuleScalarItem = 48 ---
SCALAR_ITEM: Final[RuleConstant] = (48, -1)
SIMPLE_SCALAR: Final[RuleConstant] = (48, 0)
SCALAR_WITH_CAST: Final[RuleConstant] = (48, 1)

# --- RulePartitionByClause = 55 ---
PARTITION_BY_CLAUSE: Final[RuleConstant] = (55, -1)
PARTITION_LISTED: Final[RuleConstant] = (55, 0)
PARTITION_EXCEPT_ALL: Final[RuleConstant] = (55, 1)

# --- RuleGroupingClause = 62 ---
GROUP_BY_OR_EXCEPT: Final[RuleConstant] = (62, 0)
GROUP_ALL: Final[RuleConstant] = (62, 1)

# --- RuleDpRuleset = 74 ---
DATA_POINT: Final[RuleConstant] = (74, 0)
DATA_POINT_VD: Final[RuleConstant] = (74, 1)
DATA_POINT_VAR: Final[RuleConstant] = (74, 2)

# --- RuleHrRuleset = 75 ---
HR_RULESET_TYPE: Final[RuleConstant] = (75, 0)
HR_RULESET_VD_TYPE: Final[RuleConstant] = (75, 1)
HR_RULESET_VAR_TYPE: Final[RuleConstant] = (75, 2)

# --- RuleScalarTypeConstraint = 89 ---
CONDITION_CONSTRAINT: Final[RuleConstant] = (89, 0)
RANGE_CONSTRAINT: Final[RuleConstant] = (89, 1)

# --- RuleConstant = 114 ---
CONSTANT: Final[RuleConstant] = (114, -1)
INTEGER_LITERAL: Final[RuleConstant] = (114, 0)
NUMBER_LITERAL: Final[RuleConstant] = (114, 1)
BOOLEAN_LITERAL: Final[RuleConstant] = (114, 2)
STRING_LITERAL: Final[RuleConstant] = (114, 3)
NULL_LITERAL: Final[RuleConstant] = (114, 4)


class RC:
    """Namespace for rule constants, for convenient RC.XXX access."""

    # Rules WITHOUT labeled alternatives
    START = START
    DATASET_CLAUSE = DATASET_CLAUSE
    RENAME_CLAUSE = RENAME_CLAUSE
    AGGR_CLAUSE = AGGR_CLAUSE
    FILTER_CLAUSE = FILTER_CLAUSE
    CALC_CLAUSE = CALC_CLAUSE
    KEEP_OR_DROP_CLAUSE = KEEP_OR_DROP_CLAUSE
    PIVOT_OR_UNPIVOT_CLAUSE = PIVOT_OR_UNPIVOT_CLAUSE
    CUSTOM_PIVOT_CLAUSE = CUSTOM_PIVOT_CLAUSE
    SUBSPACE_CLAUSE = SUBSPACE_CLAUSE
    VP_SIGNATURE = VP_SIGNATURE
    VP_BODY = VP_BODY
    VP_CONDITION = VP_CONDITION
    PARAMETER_COMPONENT = PARAMETER_COMPONENT
    PARAMETER = PARAMETER
    STRING_DISTANCE_METHODS = STRING_DISTANCE_METHODS
    HIERARCHY_OPERATORS = HIERARCHY_OPERATORS
    RENAME_CLAUSE_ITEM = RENAME_CLAUSE_ITEM
    AGGREGATE_CLAUSE = AGGREGATE_CLAUSE
    AGGR_FUNCTION_CLAUSE = AGGR_FUNCTION_CLAUSE
    CALC_CLAUSE_ITEM = CALC_CLAUSE_ITEM
    SUBSPACE_CLAUSE_ITEM = SUBSPACE_CLAUSE_ITEM
    JOIN_CLAUSE = JOIN_CLAUSE
    JOIN_CLAUSE_ITEM = JOIN_CLAUSE_ITEM
    USING_CLAUSE = USING_CLAUSE
    NVL_JOIN_CLAUSE = NVL_JOIN_CLAUSE
    JOIN_BODY = JOIN_BODY
    JOIN_APPLY_CLAUSE = JOIN_APPLY_CLAUSE
    ORDER_BY_CLAUSE = ORDER_BY_CLAUSE
    ORDER_BY_ITEM = ORDER_BY_ITEM
    WINDOWING_CLAUSE = WINDOWING_CLAUSE
    SIGNED_INTEGER = SIGNED_INTEGER
    SIGNED_NUMBER = SIGNED_NUMBER
    LIMIT_CLAUSE_ITEM = LIMIT_CLAUSE_ITEM
    HAVING_CLAUSE = HAVING_CLAUSE
    PARAMETER_ITEM = PARAMETER_ITEM
    OUTPUT_PARAMETER_TYPE = OUTPUT_PARAMETER_TYPE
    OUTPUT_PARAMETER_TYPE_COMPONENT = OUTPUT_PARAMETER_TYPE_COMPONENT
    INPUT_PARAMETER_TYPE = INPUT_PARAMETER_TYPE
    RULESET_TYPE = RULESET_TYPE
    SCALAR_TYPE = SCALAR_TYPE
    COMPONENT_TYPE = COMPONENT_TYPE
    DATASET_TYPE = DATASET_TYPE
    EVAL_DATASET_TYPE = EVAL_DATASET_TYPE
    SCALAR_SET_TYPE = SCALAR_SET_TYPE
    VALUE_DOMAIN_NAME = VALUE_DOMAIN_NAME
    RULESET_ID = RULESET_ID
    RULESET_SIGNATURE = RULESET_SIGNATURE
    SIGNATURE = SIGNATURE
    RULE_CLAUSE_DATAPOINT = RULE_CLAUSE_DATAPOINT
    RULE_ITEM_DATAPOINT = RULE_ITEM_DATAPOINT
    RULE_CLAUSE_HIERARCHICAL = RULE_CLAUSE_HIERARCHICAL
    RULE_ITEM_HIERARCHICAL = RULE_ITEM_HIERARCHICAL
    HIER_RULE_SIGNATURE = HIER_RULE_SIGNATURE
    VALUE_DOMAIN_SIGNATURE = VALUE_DOMAIN_SIGNATURE
    CODE_ITEM_RELATION = CODE_ITEM_RELATION
    CODE_ITEM_RELATION_CLAUSE = CODE_ITEM_RELATION_CLAUSE
    VALUE_DOMAIN_VALUE = VALUE_DOMAIN_VALUE
    COMP_CONSTRAINT = COMP_CONSTRAINT
    MULT_MODIFIER = MULT_MODIFIER
    VALIDATION_OUTPUT = VALIDATION_OUTPUT
    VALIDATION_MODE = VALIDATION_MODE
    CONDITION_CLAUSE = CONDITION_CLAUSE
    INPUT_MODE = INPUT_MODE
    IMBALANCE_EXPR = IMBALANCE_EXPR
    INPUT_MODE_HIERARCHY = INPUT_MODE_HIERARCHY
    OUTPUT_MODE_HIERARCHY = OUTPUT_MODE_HIERARCHY
    ALIAS = ALIAS
    VAR_ID = VAR_ID
    SIMPLE_COMPONENT_ID = SIMPLE_COMPONENT_ID
    COMPONENT_ID = COMPONENT_ID
    LISTS = LISTS
    ER_CODE = ER_CODE
    ER_LEVEL = ER_LEVEL
    COMPARISON_OPERAND = COMPARISON_OPERAND
    OPTIONAL_EXPR = OPTIONAL_EXPR
    OPTIONAL_EXPR_COMPONENT = OPTIONAL_EXPR_COMPONENT
    COMPONENT_ROLE = COMPONENT_ROLE
    VIRAL_ATTRIBUTE = VIRAL_ATTRIBUTE
    VALUE_DOMAIN_ID = VALUE_DOMAIN_ID
    OPERATOR_ID = OPERATOR_ID
    ROUTINE_NAME = ROUTINE_NAME
    BASIC_SCALAR_TYPE = BASIC_SCALAR_TYPE
    RETAIN_TYPE = RETAIN_TYPE

    # Rules WITH labeled alternatives
    STATEMENT = STATEMENT
    TEMPORARY_ASSIGNMENT = TEMPORARY_ASSIGNMENT
    PERSIST_ASSIGNMENT = PERSIST_ASSIGNMENT
    DEFINE_EXPRESSION = DEFINE_EXPRESSION
    EXPR = EXPR
    PARENTHESIS_EXPR = PARENTHESIS_EXPR
    FUNCTIONS_EXPRESSION = FUNCTIONS_EXPRESSION
    CLAUSE_EXPR = CLAUSE_EXPR
    MEMBERSHIP_EXPR = MEMBERSHIP_EXPR
    UNARY_EXPR = UNARY_EXPR
    ARITHMETIC_EXPR = ARITHMETIC_EXPR
    ARITHMETIC_EXPR_OR_CONCAT = ARITHMETIC_EXPR_OR_CONCAT
    COMPARISON_EXPR = COMPARISON_EXPR
    IN_NOT_IN_EXPR = IN_NOT_IN_EXPR
    BOOLEAN_EXPR = BOOLEAN_EXPR
    IF_EXPR = IF_EXPR
    CASE_EXPR = CASE_EXPR
    CONSTANT_EXPR = CONSTANT_EXPR
    VAR_ID_EXPR = VAR_ID_EXPR
    EXPR_COMPONENT = EXPR_COMPONENT
    PARENTHESIS_EXPR_COMP = PARENTHESIS_EXPR_COMP
    FUNCTIONS_EXPRESSION_COMP = FUNCTIONS_EXPRESSION_COMP
    UNARY_EXPR_COMP = UNARY_EXPR_COMP
    ARITHMETIC_EXPR_COMP = ARITHMETIC_EXPR_COMP
    ARITHMETIC_EXPR_OR_CONCAT_COMP = ARITHMETIC_EXPR_OR_CONCAT_COMP
    COMPARISON_EXPR_COMP = COMPARISON_EXPR_COMP
    IN_NOT_IN_EXPR_COMP = IN_NOT_IN_EXPR_COMP
    BOOLEAN_EXPR_COMP = BOOLEAN_EXPR_COMP
    IF_EXPR_COMP = IF_EXPR_COMP
    CASE_EXPR_COMP = CASE_EXPR_COMP
    CONSTANT_EXPR_COMP = CONSTANT_EXPR_COMP
    COMP_ID = COMP_ID
    GENERIC_FUNCTIONS_COMPONENTS = GENERIC_FUNCTIONS_COMPONENTS
    STRING_FUNCTIONS_COMPONENTS = STRING_FUNCTIONS_COMPONENTS
    NUMERIC_FUNCTIONS_COMPONENTS = NUMERIC_FUNCTIONS_COMPONENTS
    COMPARISON_FUNCTIONS_COMPONENTS = COMPARISON_FUNCTIONS_COMPONENTS
    TIME_FUNCTIONS_COMPONENTS = TIME_FUNCTIONS_COMPONENTS
    CONDITIONAL_FUNCTIONS_COMPONENTS = CONDITIONAL_FUNCTIONS_COMPONENTS
    AGGREGATE_FUNCTIONS_COMPONENTS = AGGREGATE_FUNCTIONS_COMPONENTS
    ANALYTIC_FUNCTIONS_COMPONENTS = ANALYTIC_FUNCTIONS_COMPONENTS
    JOIN_FUNCTIONS = JOIN_FUNCTIONS
    GENERIC_FUNCTIONS = GENERIC_FUNCTIONS
    STRING_FUNCTIONS = STRING_FUNCTIONS
    NUMERIC_FUNCTIONS = NUMERIC_FUNCTIONS
    COMPARISON_FUNCTIONS = COMPARISON_FUNCTIONS
    TIME_FUNCTIONS = TIME_FUNCTIONS
    SET_FUNCTIONS = SET_FUNCTIONS
    HIERARCHY_FUNCTIONS = HIERARCHY_FUNCTIONS
    VALIDATION_FUNCTIONS = VALIDATION_FUNCTIONS
    CONDITIONAL_FUNCTIONS = CONDITIONAL_FUNCTIONS
    AGGREGATE_FUNCTIONS = AGGREGATE_FUNCTIONS
    ANALYTIC_FUNCTIONS = ANALYTIC_FUNCTIONS
    INNER_JOIN_EXPR = INNER_JOIN_EXPR
    LEFT_JOIN_EXPR = LEFT_JOIN_EXPR
    FULL_JOIN_EXPR = FULL_JOIN_EXPR
    CROSS_JOIN_EXPR = CROSS_JOIN_EXPR
    DEF_OPERATOR = DEF_OPERATOR
    DEF_DATAPOINT_RULESET = DEF_DATAPOINT_RULESET
    DEF_HIERARCHICAL = DEF_HIERARCHICAL
    DEF_VIRAL_PROPAGATION = DEF_VIRAL_PROPAGATION
    ENUMERATED_VP_CLAUSE = ENUMERATED_VP_CLAUSE
    AGGREGATION_VP_CLAUSE = AGGREGATION_VP_CLAUSE
    DEFAULT_VP_CLAUSE = DEFAULT_VP_CLAUSE
    CALL_DATASET = CALL_DATASET
    EVAL_ATOM = EVAL_ATOM
    CAST_EXPR_DATASET = CAST_EXPR_DATASET
    CALL_COMPONENT = CALL_COMPONENT
    CAST_EXPR_COMPONENT = CAST_EXPR_COMPONENT
    EVAL_ATOM_COMPONENT = EVAL_ATOM_COMPONENT
    UNARY_STRING_FUNCTION = UNARY_STRING_FUNCTION
    SUBSTR_ATOM = SUBSTR_ATOM
    REPLACE_ATOM = REPLACE_ATOM
    INSTR_ATOM = INSTR_ATOM
    STRING_DISTANCE_ATOM = STRING_DISTANCE_ATOM
    UNARY_STRING_FUNCTION_COMPONENT = UNARY_STRING_FUNCTION_COMPONENT
    SUBSTR_ATOM_COMPONENT = SUBSTR_ATOM_COMPONENT
    REPLACE_ATOM_COMPONENT = REPLACE_ATOM_COMPONENT
    INSTR_ATOM_COMPONENT = INSTR_ATOM_COMPONENT
    STRING_DISTANCE_ATOM_COMPONENT = STRING_DISTANCE_ATOM_COMPONENT
    UNARY_NUMERIC = UNARY_NUMERIC
    UNARY_WITH_OPTIONAL_NUMERIC = UNARY_WITH_OPTIONAL_NUMERIC
    BINARY_NUMERIC = BINARY_NUMERIC
    UNARY_NUMERIC_COMPONENT = UNARY_NUMERIC_COMPONENT
    UNARY_WITH_OPTIONAL_NUMERIC_COMPONENT = UNARY_WITH_OPTIONAL_NUMERIC_COMPONENT
    BINARY_NUMERIC_COMPONENT = BINARY_NUMERIC_COMPONENT
    BETWEEN_ATOM = BETWEEN_ATOM
    CHARSET_MATCH_ATOM = CHARSET_MATCH_ATOM
    IS_NULL_ATOM = IS_NULL_ATOM
    EXIST_IN_ATOM = EXIST_IN_ATOM
    BETWEEN_ATOM_COMPONENT = BETWEEN_ATOM_COMPONENT
    CHARSET_MATCH_ATOM_COMPONENT = CHARSET_MATCH_ATOM_COMPONENT
    IS_NULL_ATOM_COMPONENT = IS_NULL_ATOM_COMPONENT
    PERIOD_ATOM = PERIOD_ATOM
    FILL_TIME_ATOM = FILL_TIME_ATOM
    FLOW_ATOM = FLOW_ATOM
    TIME_SHIFT_ATOM = TIME_SHIFT_ATOM
    TIME_AGG_ATOM = TIME_AGG_ATOM
    CURRENT_DATE_ATOM = CURRENT_DATE_ATOM
    DATE_DIFF_ATOM = DATE_DIFF_ATOM
    DATE_ADD_ATOM = DATE_ADD_ATOM
    YEAR_ATOM = YEAR_ATOM
    MONTH_ATOM = MONTH_ATOM
    DAY_OF_MONTH_ATOM = DAY_OF_MONTH_ATOM
    DAY_OF_YEAR_ATOM = DAY_OF_YEAR_ATOM
    DAY_TO_YEAR_ATOM = DAY_TO_YEAR_ATOM
    DAY_TO_MONTH_ATOM = DAY_TO_MONTH_ATOM
    YEAR_TODAY_ATOM = YEAR_TODAY_ATOM
    MONTH_TODAY_ATOM = MONTH_TODAY_ATOM
    PERIOD_ATOM_COMPONENT = PERIOD_ATOM_COMPONENT
    FILL_TIME_ATOM_COMPONENT = FILL_TIME_ATOM_COMPONENT
    FLOW_ATOM_COMPONENT = FLOW_ATOM_COMPONENT
    TIME_SHIFT_ATOM_COMPONENT = TIME_SHIFT_ATOM_COMPONENT
    TIME_AGG_ATOM_COMPONENT = TIME_AGG_ATOM_COMPONENT
    CURRENT_DATE_ATOM_COMPONENT = CURRENT_DATE_ATOM_COMPONENT
    DATE_DIFF_ATOM_COMPONENT = DATE_DIFF_ATOM_COMPONENT
    DATE_ADD_ATOM_COMPONENT = DATE_ADD_ATOM_COMPONENT
    YEAR_ATOM_COMPONENT = YEAR_ATOM_COMPONENT
    MONTH_ATOM_COMPONENT = MONTH_ATOM_COMPONENT
    DAY_OF_MONTH_ATOM_COMPONENT = DAY_OF_MONTH_ATOM_COMPONENT
    DAY_OF_YEAR_ATOM_COMPONENT = DAY_OF_YEAR_ATOM_COMPONENT
    DAY_TO_YEAR_ATOM_COMPONENT = DAY_TO_YEAR_ATOM_COMPONENT
    DAY_TO_MONTH_ATOM_COMPONENT = DAY_TO_MONTH_ATOM_COMPONENT
    YEAR_TODAY_ATOM_COMPONENT = YEAR_TODAY_ATOM_COMPONENT
    MONTH_TODAY_ATOM_COMPONENT = MONTH_TODAY_ATOM_COMPONENT
    UNION_ATOM = UNION_ATOM
    INTERSECT_ATOM = INTERSECT_ATOM
    SET_OR_SYM_DIFF_ATOM = SET_OR_SYM_DIFF_ATOM
    VALIDATE_D_PRULESET = VALIDATE_D_PRULESET
    VALIDATE_HR_RULESET = VALIDATE_HR_RULESET
    VALIDATION_SIMPLE = VALIDATION_SIMPLE
    NVL_ATOM = NVL_ATOM
    NVL_ATOM_COMPONENT = NVL_ATOM_COMPONENT
    AGGR_COMP = AGGR_COMP
    COUNT_AGGR_COMP = COUNT_AGGR_COMP
    AGGR_DATASET = AGGR_DATASET
    AN_SIMPLE_FUNCTION = AN_SIMPLE_FUNCTION
    LAG_OR_LEAD_AN = LAG_OR_LEAD_AN
    RATIO_TO_REPORT_AN = RATIO_TO_REPORT_AN
    AN_SIMPLE_FUNCTION_COMPONENT = AN_SIMPLE_FUNCTION_COMPONENT
    LAG_OR_LEAD_AN_COMPONENT = LAG_OR_LEAD_AN_COMPONENT
    RANK_AN_COMPONENT = RANK_AN_COMPONENT
    RATIO_TO_REPORT_AN_COMPONENT = RATIO_TO_REPORT_AN_COMPONENT
    SCALAR_ITEM = SCALAR_ITEM
    SIMPLE_SCALAR = SIMPLE_SCALAR
    SCALAR_WITH_CAST = SCALAR_WITH_CAST
    PARTITION_BY_CLAUSE = PARTITION_BY_CLAUSE
    PARTITION_LISTED = PARTITION_LISTED
    PARTITION_EXCEPT_ALL = PARTITION_EXCEPT_ALL
    GROUP_BY_OR_EXCEPT = GROUP_BY_OR_EXCEPT
    GROUP_ALL = GROUP_ALL
    DATA_POINT = DATA_POINT
    DATA_POINT_VD = DATA_POINT_VD
    DATA_POINT_VAR = DATA_POINT_VAR
    HR_RULESET_TYPE = HR_RULESET_TYPE
    HR_RULESET_VD_TYPE = HR_RULESET_VD_TYPE
    HR_RULESET_VAR_TYPE = HR_RULESET_VAR_TYPE
    CONDITION_CONSTRAINT = CONDITION_CONSTRAINT
    RANGE_CONSTRAINT = RANGE_CONSTRAINT
    CONSTANT = CONSTANT
    INTEGER_LITERAL = INTEGER_LITERAL
    NUMBER_LITERAL = NUMBER_LITERAL
    BOOLEAN_LITERAL = BOOLEAN_LITERAL
    STRING_LITERAL = STRING_LITERAL
    NULL_LITERAL = NULL_LITERAL
