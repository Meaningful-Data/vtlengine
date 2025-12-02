"""
Exceptions.messages.py
======================

Description
-----------
All exceptions exposed by the Vtl engine.
"""
# -------------- Codification --------------

# -------------- INPUT ERRORS --------------
# 0-1-X-X = Input Validation Errors
# 0-3-X-X = DataLoad Errors

# -------------- SEMANTIC ERRORS --------------
# 1-1-X-X = Operators Semantic Errors
# 1-3-X-X = Semantic Analyzer Errors
# 1-4-X-X = AST Errors

# -------------- RUNTIME ERRORS --------------
# 2-X-X-X = RunTime Operator Errors

centralised_messages = {
    # Input Validation errors
    "0-1-0-1": {"message": 'Trying to redefine input datasets {dataset}.', "description": 'TEST'},
    "0-1-1-1": {
        "message": 'invalid script format type: {format_}. Input must be a string, TransformationScheme or Path object',
        "description": ''},
    "0-1-1-2": {"message": 'The provided input {input} can not be used in this instance.', "description": ''},
    "0-1-1-3": {"message": 'Invalid file extension: expected {expected_ext}, got {ext}', "description": ''},
    "0-1-1-6": {"message": 'Duplicated records. Combination of identifiers are repeated.', "description": ''},
    "0-1-1-7": {"message": 'G1 - The provided CSV file is empty.', "description": ''},
    "0-1-1-8": {"message": 'The following identifiers {ids} were not found , review file {file}.', "description": ''},
    "0-1-1-9": {"message": 'You have a problem related with commas, review rfc4180 standard, review file {file}.',
                "description": ''},
    "0-1-1-11": {"message": 'Wrong data in the file for this scalar/dataset {name}.', "description": ''},
    "0-1-1-13": {"message": 'Invalid key on {field} field: {key}{closest_key}.', "description": ''},
    "0-1-1-14": {"message": 'Empty datasets {dataset1} and {dataset2} shape missmatch.', "description": ''},
    "0-1-2-3": {"message": "{element_type} '{element}' is/are duplicated.", "description": ''},
    "0-1-2-5": {"message": 'File {file} must be encoded in utf-8 (without BOM).', "description": ''},
    "0-1-2-7": {"message": "Invalid value '{value}' for type {type_} {op_type} {name}.", "description": ''},
    "0-1-2-8": {
        "message": 'Cannot pass as inputs datasets/scalars defined as outputs of transformations in the script, please check: {names}',
        "description": ''},
    "0-1-2-9": {"message": 'The provided JSON does not follow the required JSON Schema', "description": ''},
    # Run SDMX errors
    "0-1-3-1": {"message": 'Expected exactly one input dataset in the whole script, found: {number_datasets}',
                "description": ''},
    "0-1-3-2": {"message": 'SDMX Dataset {schema} requires to have a Schema object defined as structure',
                "description": ''},
    "0-1-3-3": {"message": 'If no mappings are provided, only one dataset is allowed.', "description": ''},
    "0-1-3-4": {"message": 'Dataset {short_urn} not found in mapping dictionary.', "description": ''},
    "0-1-3-5": {"message": 'Dataset {dataset_name} not found in the input datasets.', "description": ''},
    "0-1-3-6": {"message": 'Input name {missing} not found in the input datasets.', "description": ''},
    "0-1-3-7": {"message": 'Invalid input datasets type: {type_}. Expected a sequence of PandasDataset.',
                "description": ''},
    # DataLoad errors
    "0-3-1-1": {"message": 'Dataset {dataset} is not valid according to JSON schema', "description": ''},
    "0-3-1-2": {
        "message": '{file} file not found. Please verify that the file exists and the provided path is correct.',
        "description": ''},
    "0-3-1-3": {"message": 'Output folder {folder} not found or invalid. Must be a valid Path or S3 directory.',
                "description": ''},
    "0-3-1-4": {
        "message": 'On Dataset {name} loading:  An identifier cannot have null values, found null values on {null_identifier}.',
        "description": ''},
    "0-3-1-5": {"message": 'On Dataset {name} loading: Datasets without identifiers must have 0 or 1 datapoints.',
                "description": ''},
    "0-3-1-6": {"message": 'On Dataset {name} loading: Component {comp_name} is missing in Datapoints.',
                "description": ''},
    "0-3-1-7": {"message": 'On Dataset {name} loading: not possible to cast column {column} to {type}.',
                "description": ''},
    "0-3-1-8": {
        "message": 'On Dataset {name} loading: Duplicated identifiers are not allowed, found on row {row_index}',
        "description": ''},
    # ------------Operators-------------
    # General Semantic errors
    "1-1-1-1": {"message": 'Invalid implicit cast from {type_1} to {type_2}.', "description": ''},
    "1-1-1-2": {"message": 'Invalid implicit cast from {type_1} and {type_2} to {type_check}.', "description": ''},
    "1-1-1-3": {"message": 'At op {op}: {entity} {name} cannot be promoted to {target_type}.', "description": ''},
    "1-1-1-4": {"message": 'At op {op}: Operation not allowed for multimeasure datasets.', "description": ''},
    "1-1-1-5": {"message": 'At op {op}: Invalid type {type}.', "description": ''},
    "1-1-1-8": {"message": 'At op {op}: Invalid Dataset {name}, no measures defined.', "description": ''},
    "1-1-1-9": {"message": 'At op {op}: Invalid Dataset {name}, all measures must have the same type: {type}.',
                "description": ''},
    "1-1-1-10": {"message": 'Component {comp_name} not found in Dataset {dataset_name}.', "description": ''},
    "1-1-1-13": {"message": "At op {op}: Component {comp_name} role must be '{role_1}', found '{role_2}'.",
                 "description": ''},
    "1-1-1-15": {"message": 'At op {op}: Datasets {name_1} and {name_2} does not contain the same number of {type}.',
                 "description": ''},
    "1-1-1-16": {"message": 'Found structure not nullable and null values.', "description": ''},
    "1-1-1-20": {"message": 'At op {op}: Only applies to datasets, instead of this a Scalar was provided.',
                 "description": ''},
    # Aggregate errors
    "1-1-2-2": {"message": 'At op {op}: Only Identifiers are allowed for grouping, found {id_name} - {id_type}.',
                "description": ''},
    "1-1-2-3": {"message": 'Having component output type must be boolean, found {type}.', "description": ''},
    # Analytic errors
    "1-1-3-2": {"message": 'At op {op}: Only Identifiers are allowed for partitioning, found {id_name} - {id_type}.',
                "description": ''},
    # Cast errors
    "1-1-5-1": {"message": 'Type {type_1}, cannot be cast to {type_2}.', "description": ''},
    "1-1-5-3": {"message": 'Impossible to cast from type {type_1} to {type_2}, without providing a mask.',
                "description": ''},
    "1-1-5-4": {"message": 'Invalid mask to cast from type {type_1} to {type_2}.', "description": ''},
    "1-1-5-5": {
        "message": "A mask can't be provided to cast from type {type_1} to {type_2}. Mask provided: {mask_value}.",
        "description": ''},
    "2-1-5-1": {"message": 'Impossible to cast {value} from type {type_1} to {type_2}.', "description": ''},
    "2-1-5-2": {"message": 'Value {value} has decimals, cannot cast to integer', "description": ''},
    # Clause errors
    "1-1-6-2": {
        "message": 'At op {op}: The identifier {name} in dataset {dataset} could not be included in the {op} op.',
        "description": ''},
    "1-1-6-4": {
        "message": 'At op {op}: Alias symbol cannot have the name of a component symbol: {symbol_name} - {comp_name}.',
        "description": ''},
    "1-1-6-5": {"message": 'At op {op}: Scalar values are not allowed at sub operator, found {name}.',
                "description": ''},
    "1-1-6-6": {"message": 'Membership is not allowed inside a clause, found {dataset_name}#{comp_name}.',
                "description": ''},
    "1-1-6-7": {"message": 'Cannot use component {comp_name} as it was generated in another calc expression.',
                "description": ''},
    "1-1-6-8": {"message": 'Cannot use component {comp_name} for rename, it is already in the dataset {dataset_name}.',
                "description": ''},
    "1-1-6-9": {"message": 'At op {op}: The following components are repeated: {from_components}.', "description": ''},
    "1-1-6-10": {"message": 'At op {op}: Component {operand} in dataset {dataset_name} is not an identifier',
                 "description": ''},
    "1-1-6-11": {"message": 'Ambiguity for this variable {comp_name}, exists as a Scalar and component.',
                 "description": ''},
    "1-1-6-12": {"message": 'At op {op}: Not allowed to drop the last element.', "description": ''},
    "1-1-6-13": {"message": 'At op {op}: Not allowed to overwrite an identifier: {comp_name}', "description": ''},
    # Comparison errors
    "1-1-7-1": {
        "message": 'At op {op}: Value in {left_name} of type {left_type} is not comparable to value {right_name} of type {right_type}.',
        "description": ''},
    # Conditional errors
    "1-1-9-1": {"message": "At op {op}: The evaluation condition must result in a Boolean expression, found '{type}'.",
                "description": ''},
    "1-1-9-3": {"message": 'At op {op}: Then clause {then_name} and else clause {else_name}, both must be Scalars.',
                "description": ''},
    "1-1-9-4": {"message": 'At op {op}: The condition dataset {name} must contain an unique measure.',
                "description": ''},
    "1-1-9-5": {"message": "At op {op}: The condition dataset Measure must be a Boolean, found '{type}'.",
                "description": ''},
    "1-1-9-6": {
        "message": 'At op {op}: Then-else datasets have different number of identifiers compared with condition dataset.',
        "description": ''},
    "1-1-9-9": {"message": 'At op {op}: {clause} component {clause_name} role must be {role_1}, found {role_2}.',
                "description": ''},
    "1-1-9-10": {
        "message": 'At op {op}: {clause} dataset have different number of identifiers compared with condition dataset.',
        "description": ''},
    "1-1-9-11": {"message": 'At op {op}: Condition component {name} must be Boolean, found {type}.', "description": ''},
    "1-1-9-12": {
        "message": 'At op {op}: then clause {then_symbol} and else clause {else_symbol}, both must be Datasets or at least one of them a Scalar.',
        "description": ''},
    "1-1-9-13": {
        "message": 'At op {op}: then {then} and else {else_clause} datasets must contain the same number of components.',
        "description": ''},
    "2-1-9-1": {"message": 'At op {op}: Condition operators must have the same operator type.', "description": ''},
    "2-1-9-2": {"message": "At op {op}: Condition {name} it's not a boolean.", "description": ''},
    "2-1-9-3": {"message": 'At op {op}: All then and else operands must be scalars.', "description": ''},
    "2-1-9-4": {"message": 'At op {op}: Condition {name} must be boolean type.', "description": ''},
    "2-1-9-5": {"message": 'At op {op}: Condition Dataset {name} measure must be Boolean.', "description": ''},
    "2-1-9-6": {"message": 'At op {op}: At least a then or else operand must be Dataset.', "description": ''},
    "2-1-9-7": {"message": 'At op {op}: All Dataset operands must have the same components.', "description": ''},
    # Data Validation errors
    "1-1-10-1": {"message": 'At op {op}: The {op_type} operand must have exactly one measure of type {me_type}',
                 "description": ''},
    "1-1-10-2": {"message": 'At op {op}: Number of variable has to be equal between the call and signature.',
                 "description": ''},
    "1-1-10-3": {
        "message": 'At op {op}: Name in the call {found} has to be equal to variable rule in signature {expected}.',
        "description": ''},
    "1-1-10-4": {
        "message": 'At op {op}: When a hierarchical ruleset is defined for value domain, it is necessary to specify the component with the rule clause on call.',
        "description": ''},
    "1-1-10-5": {"message": 'No rules to analyze on Hierarchy Roll-up as rules have no = operator.', "description": ''},
    "1-1-10-6": {
        "message": 'At op {op}: Name in the call {found} has to be equal to variable condition in signature {expected} .',
        "description": ''},
    "1-1-10-7": {"message": 'Not found component {comp_name} on signature.', "description": ''},
    "1-1-10-8": {"message": 'At op {op}: Measures involved have to be numerical, other types found {found}.',
                 "description": ''},
    "1-1-10-9": {
        "message": 'Invalid signature for the ruleset {ruleset}. On variables, condComp and ruleComp must be the same',
        "description": ''},
    # General Operators
    "2-1-12-1": {
        "message": 'At op {op}: Create a null measure without a scalar type is not allowed.Please use cast operator.',
        "description": ''},  # RunTimeError.
    # Join Operators
    "1-1-13-1": {"message": 'At op {op}: Duplicated alias {duplicates}.', "description": ''},
    "1-1-13-2": {"message": 'At op {op}: Missing mandatory aliasing.', "description": ''},
    "1-1-13-3": {"message": 'At op {op}: Join conflict with duplicated names for column {name} from original datasets.',
                 "description": ''},
    "1-1-13-4": {
        "message": 'At op {op}: Using clause, using={using_names}, does not define all the identifiers, of non reference dataset {dataset}.',
        "description": ''},
    "1-1-13-5": {
        "message": 'At op {op}: Invalid subcase B1, All the datasets must share as identifiers the using ones.',
        "description": ''},
    "1-1-13-6": {
        "message": "At op {op}: Invalid subcase B2, All the declared using components '{using_components}' must be present as components in the reference dataset '{reference}'.",
        "description": ''},
    "1-1-13-7": {
        "message": 'At op {op}: Invalid subcase B2, All the non reference datasets must share as identifiers the using ones.',
        "description": ''},
    "1-1-13-8": {"message": 'At op {op}: No available using clause.', "description": ''},
    "1-1-13-9": {"message": 'Ambiguity for this variable {comp_name} inside a join clause.', "description": ''},
    "1-1-13-10": {"message": 'The join operator does not perform scalar/component operations.', "description": ''},
    "1-1-13-11": {
        "message": 'At op {op}: Invalid subcase A, {dataset_reference} should be a superset but {component} not found.',
        "description": ''},
    "1-1-13-12": {"message": 'At op {op}: Invalid subcase A. There are different identifiers for the provided datasets',
                  "description": ''},
    "1-1-13-13": {
        "message": 'At op {op}: Invalid subcase A. There are not same number of identifiers for the provided datasets',
        "description": ''},
    "1-1-13-14": {"message": 'Cannot perform a join over a Dataset Without Identifiers: {name}.', "description": ''},
    "1-1-13-15": {
        "message": 'At op {op}: {comp_name} has to be a Measure for all the provided datasets inside the join',
        "description": ''},
    "1-1-13-16": {"message": 'At op {op}: Invalid use, please review : {msg}.', "description": ''},
    "1-1-13-17": {
        "message": 'At op {op}: {comp_name} not present in the dataset(result from join VDS) at the time it is called',
        "description": ''},
    # Operators general errors
    "1-1-14-1": {"message": "At op {op}: Measure names don't match: {left} - {right}.", "description": ''},
    "1-1-14-3": {
        "message": 'At op {op}: Invalid scalar types for identifiers at DataSet {dataset}. One {type} identifier expected, {count} found.',
        "description": ''},
    "1-1-14-5": {"message": 'At op {op}: {names} with type/s {types} is not compatible with {op}', "description": ''},
    "1-1-14-6": {
        "message": 'At op {op}: {comp_name} with type {comp_type} and scalar_set with type {scalar_type} is not compatible with {op}',
        "description": ''},
    "1-1-14-9": {
        "message": 'At op {op}: {names} with type/s {types} is not compatible with {op} on datasets {datasets}.',
        "description": ''},
    # Numeric Operators
    "1-1-15-8": {"message": 'At op {op}: {op} operator cannot have a {comp_type} as parameter.', "description": ''},
    "2-1-15-1": {"message": 'At op {op}: Component {comp_name} from dataset {dataset_name} contains negative values.',
                 # RunTimeError.
                 "description": ''},  # RunTimeError.
    "2-1-15-2": {"message": 'At op {op}: Value {value} could not be negative.', "description": ''},  # RunTimeError.
    "2-1-15-3": {"message": 'At op {op}: Base value {value} could not be less or equal 0.', "description": ''},
    # RunTimeError.
    "2-1-15-4": {"message": 'At op {op}: Invalid values in Component {name}.', "description": ''},  # RunTimeError.
    "2-1-15-5": {"message": 'At op {op}: Invalid values in {name_1} and {name_2}.', "description": ''},  # RunTimeError.
    "2-1-15-6": {"message": 'At op {op}: Scalar division by Zero.', "description": ''},  # RunTimeError.
    "2-1-15-7": {"message": 'At op {op}: {op} operator cannot be a dataset.', "description": ''},  # RunTimeError.
    # Set Operators
    "1-1-17-1": {"message": 'At op {op}: Datasets {dataset_1} and {dataset_2} have different number of components',
                 "description": ''},
    # String Operators
    "1-1-18-1": {"message": 'At op {op}: Invalid Dataset {name}. Dataset with one measure expected.',
                 "description": ''},
    "1-1-18-2": {"message": 'At op {op}: Composition of DataSet and Component is not allowed.', "description": ''},
    "1-1-18-3": {"message": 'At op {op}: Invalid parameter position: {pos}.', "description": ''},
    "1-1-18-4": {"message": 'At op {op}: {param_type} parameter should be {correct_type}.', "description": ''},
    "1-1-18-6": {"message": 'At op {op}: Datasets have different measures.', "description": ''},
    "1-1-18-7": {"message": 'At op {op}: Invalid number of parameters {number}, {expected} expected.',
                 "description": ''},
    "1-1-18-8": {"message": 'At op {op}: {msg} in regexp: {regexp},  in position {pos}.', "description": ''},
    "1-1-18-10": {"message": 'At op {op}: Cannot have a Dataset as parameter', "description": ''},
    # Time operators
    "1-1-19-1": {"message": 'At op {op}: {op} must have a {data_type} type on {comp}.', "description": ''},
    "1-1-19-2": {"message": 'At op {op}: Unknown date type for {op}.', "description": ''},
    "1-1-19-3": {"message": 'At op {op}: Invalid {param} for {op}.', "description": ''},
    "1-1-19-4": {
        "message": 'At op {op}: Invalid values {value_1} and {value_2}, periodIndTo parameter must be a larger duration value than periodIndFrom parameter.',
        "description": ''},
    "1-1-19-5": {
        "message": 'At op {op}: periodIndTo parameter must be a larger duration value than the values to aggregate.',
        "description": ''},
    "1-1-19-6": {"message": 'At op {op}: Time type used in the component {comp} is not supported.', "description": ''},
    "1-1-19-7": {
        "message": 'At op {op}: can be applied only on Data Sets (of time series) and returns a Data Set (of time series).',
        "description": ''},
    "1-1-19-8": {"message": 'At op {op}: {op} can only be applied to a {comp_type}', "description": ''},
    "1-1-19-9": {"message": 'At op {op}: {op} can only be applied to a {comp_type} with a {param}', "description": ''},
    "1-1-19-10": {"message": '{op} can only be applied to operands with data type as Date or Time Period',
                  "description": ''},
    "1-1-19-11": {"message": 'The time aggregation operand has to be defined if not used inside an aggregation.',
                  "description": ''},
    # ---------Semantic Analyzer Common----
    "1-3-1": {"message": "Please don't use twice {alias} like var_to.", "description": ''},
    "1-3-3": {"message": 'Overwriting a dataset/variable is not allowed, trying it with {varId_value}.',
              "description": ''},
    "1-3-4": {"message": 'Cannot perform a rename with two equal values: {left_value} -> {right_value}.',
              "description": ''},
    "1-3-5": {"message": '{node_op} not found or not valid for {op_type}.', "description": ''},
    "1-3-8": {"message": 'Defined Operator {node_value} not previously defined.', "description": ''},
    "1-3-9": {"message": 'Not valid set declaration, found duplicates {duplicates}.', "description": ''},
    "1-3-10": {"message": 'Not valid set declaration, mixed scalar types {scalar_1} and {scalar_2}.',
               "description": ''},
    "1-3-12": {"message": 'Default arguments cannot be followed by non-default arguments.', "description": ''},
    "1-3-15": {"message": 'Missing datastructure definition for required input Dataset {input}.', "description": ''},
    "1-3-17": {"message": 'Operations without output assigned are not available.', "description": ''},
    "1-3-19": {"message": 'No {node_type} {node_value} found.', "description": ''},
    "1-3-20": {"message": 'RuleComp of Hierarchical Ruleset can only be an identifier, {name} is a {role}.',
               "description": ''},
    "1-3-21": {"message": 'Value {value} not valid, kind {node_kind}.', "description": ''},
    "1-3-22": {"message": 'Unable to categorize {node_value}.', "description": ''},
    "1-3-23": {"message": "Missing value domain '{name}' definition, please provide an structure.", "description": ''},
    "1-3-24": {"message": 'Internal error on Analytic operators inside a calc, No partition or order symbol found.',
               "description": ''},
    "1-3-26": {"message": 'Value domain {name} not found.', "description": ''},
    "1-3-27": {"message": 'Dataset without identifiers are not allowed in {op} operator.', "description": ''},
    "1-3-28": {
        "message": 'At op {op}: invalid number of parameters: received {received}, expected at least: {expected}',
        "description": ''},
    "1-3-29": {
        "message": 'At op {op}: can not use user defined operator that returns a component outside clause operator or rule',
        "description": ''},
    "1-3-30": {"message": 'At op {op}: too many parameters: received {received}, expected: {expected}',
               "description": ''},
    "1-3-31": {"message": 'Cannot use component {name} outside an aggregate function in a having clause.',
               "description": ''},
    "1-3-32": {"message": 'Cannot perform operation {op} inside having clause.', "description": ''},
    "1-3-33": {"message": 'Having clause is not permitted if group by clause is not present.', "description": ''},
    "1-3-34": {"message": 'At op {op}: Cannot use constant as a {type} parameter, found on {param}.',
               "description": ''},
    "1-3-35": {"message": 'At op {op}: Cannot perform aggregation inside a calc.', "description": ''},
    # ---------------AST------------
    # AST Helpers
    "1-4-1-1": {"message": 'At op {op}: User defined {option} declared as {type_1}, found {type_2}.',
                "description": ''},
    "1-4-1-2": {"message": 'Using variable {value}, not defined at {op} definition.', "description": ''},
    "1-4-1-3": {"message": 'At op {op}: using variable {value}, not defined as an argument.', "description": ''},
    "1-4-1-4": {"message": 'Found duplicates at arguments naming, please review {type} definition {op}.',
                "description": ''},
    "1-4-1-5": {"message": 'Found duplicates at rule naming: {names}. Please review {type} {ruleset_name} definition.',
                "description": ''},
    "1-4-1-6": {"message": 'At op {op}: Arguments incoherence, {defined} defined {passed} passed.', "description": ''},
    "1-4-1-7": {
        "message": 'All rules must be named or not named, but found mixed criteria at {type} definition {name}.',
        "description": ''},
    "1-4-1-8": {
        "message": "All rules must have different code items in the left side of '=' in hierarchy operator at hierachical ruleset definition {name}.",
        "description": ''},
    "1-4-1-9": {"message": 'At op check_datapoint: {name} has an invalid datatype expected DataSet, found Scalar.',
                "description": ''},
    # AST Creation
    "1-4-2-0": {"message": 'Error creating DAG.', "description": ''},
    "1-4-2-1": {"message": 'Eval could not be called without a {option} type definition.', "description": ''},
    "1-4-2-2": {"message": 'Optional or empty expression node is not allowed in time_agg.', "description": ''},
    "1-4-2-3": {"message": '{value} could not be called in the count.', "description": ''},
    "1-4-2-4": {"message": 'At op {op}: Only one order_by element must be used in Analytic with range windowing.',
                "description": ''},
    "1-4-2-5": {"message": 'At op {op}: User defined operator without returns is not implemented.', "description": ''},
    "1-4-2-6": {"message": 'At op {op}: Window must be provided.', "description": ''},
    "1-4-2-7": {"message": 'At op {op}: Partition by or order by clause must be provided for Analytic operators.',
                "description": ''},
    "1-4-2-8": {"message": 'At op {op}: Vtl Script contains Cycles, no DAG established. Nodes involved: {nodes}.',
                "description": ''},
    # ---------- RunTimeErrors ----------
    "2-1-19-1": {
        "message": 'At op {op}: Invalid values {value_1} and {value_2} for duration, periodIndTo parameter must be a larger duration value than the values to aggregate.',
        "description": ''},
    "2-1-19-2": {"message": 'Invalid period indicator {period}.', "description": ''},
    "2-1-19-3": {"message": 'Only same period indicator allowed for both parameters ({period1} != {period2}).',
                 "description": ''},
    "2-1-19-4": {"message": 'Date setter, ({value} > {date}). Cannot set date1 with a value higher than date2.',
                 "description": ''},
    "2-1-19-5": {"message": 'Date setter, ({value} < {date}). Cannot set date2 with a value lower than date1.',
                 "description": ''},
    "2-1-19-6": {"message": 'Invalid period format, must be YYYY-(L)NNN: {period_format}', "description": ''},
    "2-1-19-7": {"message": 'Period Number must be between 1 and {periods} for period indicator {period_indicator}.',
                 "description": ''},
    "2-1-19-8": {"message": 'Invalid date format, must be YYYY-MM-DD: {date}', "description": ''},
    "2-1-19-9": {"message": 'Invalid day {day} for year {year}.', "description": ''},
    "2-1-19-10": {"message": 'Invalid year {year}, must be between 1900 and 9999.', "description": ''},
    "2-1-19-11": {"message": '{op} operator is not compatible with time values', "description": ''},
    "2-1-19-12": {"message": 'At op {op}: Invalid param type {type} for param {name}, expected {expected}.',
                  "description": ''},
    "2-1-19-13": {"message": 'At op {op}: Invalid param data_type {type} for param {name}, expected {expected}.',
                  "description": ''},
    "2-1-19-14": {"message": 'At op {op}: Invalid dataset {name}, requires at least one Date/Time_Period measure.',
                  "description": ''},
    "2-1-19-15": {"message": '{op} can only be applied according to the iso 8601 format mask', "description": ''},
    "2-1-19-16": {"message": '{op} can only be positive numbers', "description": ''},
    "2-1-19-17": {"message": 'At op {op}: Time operators comparison are only support = and <> comparison operations',
                  "description": ''},
    "2-1-19-18": {
        "message": 'At op {op}: Time operators do not support < and > comparison operations, so its not possible to use get the max or min between two time operators',
        "description": ''},
    "2-1-19-19": {
        "message": 'Time Period comparison (>, <, >=, <=) with different period indicator is not supported, found {value1} {op} {value2}',
        "description": ''},
    "2-1-19-20": {
        "message": 'Time Period operands with different period indicators do not support < and > comparison operations, unable to get the {op}',
        "description": ''},
    # ----------- Interpreter Common ------
    "2-3-1": {"message": '{comp_type} {comp_name} not found.', "description": ''},
    "2-3-2": {"message": '{op_type} cannot be used with {node_op} operators.', "description": ''},
    "2-3-4": {"message": '{op} operator must have a {comp}', "description": ''},
    "2-3-5": {"message": 'Expected {param_type}, got {type_name} on UDO {op}, parameter {param_name}.',
              "description": ''},
    "2-3-6": {"message": 'Dataset or scalar {dataset_name} not found, please check input datastructures.',
              "description": ''},
    "2-3-9": {"message": '{comp_type} {comp_name} not found in {param}.', "description": ''},
    "2-3-10": {"message": 'No {comp_type} have been defined.', "description": ''},
    "2-3-11": {"message": '{pos} operand must be a dataset.', "description": ''},
}
