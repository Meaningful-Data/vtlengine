"""
Exceptions.messages.py
======================

Description
-----------
All exceptions exposed by the Vtl engine.
"""
centralised_messages = {
    # Input Validation errors
    "0-1-2-1": "Invalid json structure because additional properties have been supplied on file {filename}.",
    "0-1-2-2": "Errors found on file {filename}: {errors}",
    "0-1-2-3": "Component {component} is duplicated.",
    "0-1-2-4": "Invalid json structure because {err} on file {filename}.",
    "0-1-2-5": "The library item {li}, used in this module {mdl}, is not found.",
    # JSON Schema validations
    # Infer Data Structure errors
    "0-1-1-1": "A csv file or a dataframe is required.",
    "0-1-1-2": "The provided {source} must have data to can infer the data structure.",
    "0-1-1-3": "Can not infer data structure: {errors}",
    "0-1-1-4": "Dataset has identifiers that can't be nullables, found null values on: {null_identifiers}.",
    "0-1-1-5": "Datasets without identifiers must have 0 or 1 datapoints, found {number} on {dataset_name}.",
    "0-1-1-6": "Duplicated records. Combination of identifiers are repeated.",
    "0-1-1-7": "G1 - The provided CSV file is empty",
    "0-1-1-8": "The following identifiers {ids} were not found , review file {file}",
    "0-1-1-9": "You have a problem related with commas, review rfc4180 standard, review file {file}",
    "0-1-1-10": "{value} is not allowed as {type} in this scalardataset {name}",
    "0-1-1-11": "Wrong data in the file for this scalardataset {name}",
    #
    "0-1-0-1": " Trying to redefine input datasets {dataset}",  # Semantic Error
    # ------------Operators-------------
    # General Semantic errors
    "1-1-1-1": "At op {op}. Unable to validate types.",
    "1-1-1-2": "At op {op}: Component {comp_name} type must be '{type_1}', found '{type_2}'.",
    "1-1-1-3": "At op {op}: Invalid data type for Component {comp_name} and Scalar {scalar_name}.",
    "1-1-1-4": "At op {op}: Dataset {name} contains more than one measure.",
    "1-1-1-5": "At op {op}: Invalid data type {type} for Scalar {scalar_name}.",
    "1-1-1-6": "At op {op}: Internal error: Not same parents.",  # TODO: Deprecated not in use, delete this.
    "1-1-1-7": "At op {op}: Invalid data type {type} for Component {name}.",
    "1-1-1-8": "At op {op}: Invalid Dataset {name}, no measures defined.",
    "1-1-1-9": "At op {op}: Invalid Dataset {name}, all measures must have the same type: {type}.",
    "1-1-1-10": "At op {op}: Component {comp_name} not found in Dataset {dataset_name}.",
    "1-1-1-11": "At op {op}: Identifier {name} is specified more than once.",
    "1-1-1-12": "At op {op}: Different scalar types for component {comp_name} and set {set_name}.",
    "1-1-1-13": "At op {op}: Component {comp_name} role must be '{role_1}', found '{role_2}'.",
    "1-1-1-14": "At op {op}: Dataset {name} type must be '{type_1}'.",
    "1-1-1-15": "At op {op}: Datasets {name_1} and {name_2} does not contain the same number of {type}.",
    "1-1-1-16": "At op {op}: You cannot convert to an Identifier, Component {name} with structure nullable=true.",
    "1-1-1-17": "At op {op}: Problem with nullabillity for this components {name_1} and {name_2}.",
    "1-1-1-18": "No {type} {value} found.",
    "1-1-1-19": "At op {op}: Invalid data type for Scalar {scalar_name_1} and Scalar {scalar_name_2}.",
    "1-1-1-20": "At op {op}: Only applies to datasets, instead of this a Scalar was provided.",
    # General Interpreter errors
    "2-1-1-1": "At op {op}: Unable to evaluate.",
    "2-1-1-2": "At op {op}: Dataset {name} is empty.", # TODO: Review this message, for unpivot for example we can't raise this error, because we can have a empty dataset
    "2-1-1-3": "At op {op}: No rules have results.",
    # Aggregate errors
    "1-1-2-1": "At op {op}: No measures found to aggregate.",
    "1-1-2-2": "At op {op}: Only Identifiers are allowed for grouping, found {id_name} - {id_type}.",
    "1-1-2-3": "Having component output type must be boolean, found {type}.",
    "1-1-2-4": "At op {op}: Component {id_name} not found in dataset",
    # Analytic errors
    "1-1-3-1": "At op {op}: No measures found to analyse.",
    "1-1-3-2": "At op {op}: Only Identifiers are allowed for partitioning, found {id_name} - {id_type}.",
    # Cast errors
    "1-1-5-1": "At op {op}: Type {type_1}, cannot be cast to {type_2}.",
    "1-1-5-3": "At op {op}: Impossible to cast from type {type_1} to {type_2}, without providing a mask.",
    "1-1-5-4": "At op {op}: Invalid mask to cast from type {type_1} to {type_2}.",
    "1-1-5-5": "At op {op}: A mask can't be provided to cast from type {type_1} to {type_2}.\
        Mask provided: {mask_value}.",
    "2-1-5-1": "At op {op}: Impossible to cast {value} from type {type_1} to {type_2}.",
    # Clause errors
    "1-1-6-1": "At op {op}: Component {comp_name} not found in dataset {dataset_name}.",
    "1-1-6-2": "At op {op}: The identifier {name} in dataset {dataset} could not be included in the {op} op.",
    "1-1-6-3": "At op {op}: Found duplicated identifiers after {op} clause.",
    "1-1-6-4": "At op {op}: Alias symbol cannot have the name of a component symbol: {symbol_name} - {comp_name}.",
    "1-1-6-5": "At op {op}: Scalar values are not allowed at sub operator, found {name}.",
    "1-1-6-6": "Membership is not allowed inside a clause, found {dataset_name}#{comp_name}.",
    "1-1-6-7": "Cannot use component {comp_name} as it was generated in another calc expression.",  # all the components used in calccomp must belong to the operand dataset
    "1-1-6-8": "Cannot use component {comp_name} for rename because is yet in the datastructure and can't be duplicated.",  # it is the same error that 1-1-8-1 AND similar but not the same 1-3-1
    "1-1-6-9": "At op {op}: The following components are repeated: {from_components}.",
    "1-1-6-10": "At op {op}: Component {operand} in dataset {dataset_name} is not an identifier",
    "1-1-6-11": "Ambiguity for this variable {comp_name} inside a {clause}, exists as scalardataset and component.", # it is the same as the one that appears in joins, but are differents kinds of failures
    "1-1-6-12": "At op {op}: Not allowed to drop the last element.",
    "1-1-6-13": "At op {op}: Not allowed to create or overwrite an identifier: {comp_name}",
    "1-1-6-15": "At op {op}: Component {comp_name} already exists in dataset {dataset_name}",
    # Comparison errors
    "1-1-7-1": "At op {op}: Value in {left_name} of type {left_type} is not comparable to value {right_name} of type {right_type}.",
    # Conditional errors
    "1-1-9-1": "At op {op}: The evaluation condition must result in a Boolean expression, found '{type}'.",
    "1-1-9-2": "At op {op}: Conditional clauses with different scalar types. Then: {type_1}. Else: {type_2}.",
    "1-1-9-3": "At op {op}: Then clause {then_name} and else clause {else_name}, both must be Scalars.",
    "1-1-9-4": "At op {op}: The condition dataset {name} must contain an unique measure.",
    "1-1-9-5": "At op {op}: The condition dataset Measure must be a Boolean, found '{type}'.",
    "1-1-9-6": "At op {op}: Then-else datasets have different number of identifiers compared with condition dataset.",
    "1-1-9-7": "At op {op}: {clause} component {clause_name} not present at dataset {dataset_name}.",
    "1-1-9-8": "At op {op}: {clause} component {clause_name} type must be {type_1}, found {type_2}.",
    "1-1-9-9": "At op {op}: {clause} component {clause_name} role must be {role_1}, found {role_2}.",
    "1-1-9-10": "At op {op}: {clause} dataset have different number of identifiers compared with condition dataset.",
    "1-1-9-11": "At op {op}: Condition component {name} must be Boolean, found {type}.",
    "1-1-9-12": "At op {op}: then clause {then_symbol} and else clause {else_symbol}, both must be Datasets or at least one of them a Scalar.",
    "1-1-9-13": "At op {op}: then {then} and else {else_clause} datasets must contain the same number of components.",
    # Data Validation errors
    "1-1-10-1": "At op {op}: The {op_type} operand must have exactly one measure of type {me_type}",
    "1-1-10-2": "At op {op}: Number of variable has to be equal between the call and signature.",
    "1-1-10-3": "At op {op}: Name in the call {founded} has to be equal to variable rule in signature {expected} .",
    "1-1-10-4": "At op {op}: When a hierarhical ruleset is defined for value domain, it is necessary to specify the component with the rule clause",
    "1-1-10-5": "At op {op}: Invalid data type {type} for {component}, not compatible with value domain STRING.",
    "1-1-10-6": "At op {op}: Name in the call {founded} has to be equal to variable condition in signature {expected} .",
    "1-1-10-7": "At op {op}: Position in the call {founded} has to be equal to position condition in signature.",
    "1-1-10-8": "At op {op}: Measures involved have to be numerical, other types found {found}.",
    # General Operators
    "1-1-12-1": "At op {op}: You could not recalculate the identifier {name}.",
    "2-1-12-1": "At op {op}: Create a null measure without a scalar type is not allowed. Please use cast operator.",
    # Join Operators
    "1-1-13-1": "At op {op}: Duplicated alias {duplicates}.",
    "1-1-13-2": "At op {op}: Missing mandatory aliasing.",
    "1-1-13-3": "At op {op}: Join conflict with duplicated names for column {name} from original datasets.",
    "1-1-13-4": "At op {op}: Using clause, using={using_names}, does not define all the identifiers, identifiers= {identifiers_names} of non reference dataset {dataset}.",
    "1-1-13-5": "At op {op}: Invalid subcase B1, All the datasets must share as identifiers the using ones.",  # not in use but we keep for later, in use 1-1-13-4
    "1-1-13-6": "At op {op}: Invalid subcase B2, All the declared using components '{using_components}' must be present as components in the reference dataset '{reference}'.",
    "1-1-13-7": "At op {op}: Invalid subcase B2, All the non reference datasets must share as identifiers the using ones.",
    "1-1-13-8": "At op {op}: No available using clause.",
    "1-1-13-9": "At op {op}: Ambiguity for this variable {comp_name} inside a {clause}.",
    "1-1-13-10": "The join operator does not perform scalar/component operations.",
    "1-1-13-11": "At op {op}: Invalid subcase A, {dataset_reference} should be a superset but {component} not found.",  # inner_join and left join
    "1-1-13-12": "At op {op}: Invalid subcase A. There are different identifiers for the provided datasets",  # full_join
    "1-1-13-13": "At op {op}: Invalid subcase A. There are not same number of identifiers for the provided datasets",  # full_join
    "1-1-13-14": "Cannot perform a join over a Dataset Without Identifiers: {name}.",
    "1-1-13-15": "At op {op}: {comp_name} has to be a Measure for all the provided datasets inside the join",
    "1-1-13-16": "At op {op}: Invalid use, please review : {msg}.",
    "1-1-13-17": "At op {op}: {comp_name} not present in the dataset(result from join VDS) at the time it is called",
    "2-1-13-1": "At op {op}: No available using clause.",  # TODO: Delete this
    # Operators Operators
    "1-1-14-1": "At op {op}: Measure names don't match: {left} - {right}.",
    "1-1-14-3": "At op {op}: Invalid scalar types for identifiers at DataSet {dataset}. One {type} identifier expected, {count} found.",
    "1-1-14-5": "At op {op}: {names} with type/s {types} is not compatible with {op}",
    "1-1-14-6": "At op {op}: {comp_name} with type {comp_type} and scalar_set with type {scalar_type} is not compatible with {op}",
    "1-1-14-7": "At op {op}: {entity} {name} cannot be promoted to {target_type}.",
    "1-1-14-8": "At op {op}: Operation not allowed for multimeasure datasets.",
    "1-1-14-9": "At op {op}: {names} with type/s {types} is not compatible with {op} on datasets {datasets}.",

    # Numeric Operators
    "2-1-15-1": "At op {op}: Component {comp_name} from dataset {dataset_name} contains negative values.",
    "2-1-15-2": "At op {op}: Value {value} could not be negative.",
    "2-1-15-3": "At op {op}: Base value {value} could not be less or equal 0.",
    "2-1-15-4": "At op {op}: Invalid values in Component {name}.",
    "2-1-15-5": "At op {op}: Invalid values in {name_1} and {name_2}.",
    "2-1-15-6": "At op {op}: Scalar division by Zero.",
    # Set Operators
    "1-1-17-1": "At op {op}: Datasets {dataset_1} and {dataset_2} have different number of components",
    # String Operators
    "1-1-18-1": "At op {op}: Invalid Dataset {name}. Dataset with one measure expected.",
    "1-1-18-2": "At op {op}: Composition of DataSet and Component is not allowed.",
    "1-1-18-3": "At op {op}: Invalid {pos} parameter {value}.",
    "1-1-18-4": "At op {op}: {param_type} parameter should be {correct_type}.",
    "1-1-18-6": "At op {op}: Datasets have different measures.",
    "1-1-18-7": "At op {op}: Invalid number of parameters {number}, {expected} expected.",
    "1-1-18-8": "At op {op}: {msg} in regexp: {regexp},  in position {pos}.",
    # Time operators
    "1-1-19-1": "At op {op}: Operation with {type} is not defined.",
    "1-1-19-2": "At op {op}: Invalid type {type} for parameter {param}.",
    "1-1-19-3": "At op {op}: Invalid parameter {value}.",
    "1-1-19-4": "At op {op}: Invalid values {value_1} and {value_2}, periodIndTo parameter must be a larger duration value than periodIndFrom parameter.",
    "1-1-19-5": "At op {op}: periodIndTo parameter must be a larger duration value than the values to aggregate.",
    "1-1-19-6": "At op {op}: Time type used in the component {comp} is not supported.",
    "1-1-19-7": "At op {op}: can be applied only on Data Sets (of time series) and returns a Data Set (of time series).",  #flow_to_stock, stock_to_flow
    "2-1-19-1": "At op {op}: Invalid values {value_1} and {value_2} for duration, periodIndTo parameter must be a larger duration value than the values to aggregate.",
    # ----------- Interpreter Common ------
    "2-3-1": "No {type} {value} found.",
    "2-3-2": "Not valid {op_type} for {node_op}.",
    "2-3-3": "Internal error: Not able to categorize {value}.",
    "2-3-4": "At op {op} inside a calc: No {option} found.",
    "2-3-5": "Element Not found, expect found {name_1} but retrieved {name_2}.",
    "2-3-6": "Components in structure and data points are not the same on dataset {dataset_name}. {names} found",
    "2-3-7": "Internal Error. Ambiguity in component extraction. {values}",
    "2-3-8": "At op {op}:{msg}.",
    # ---------Semantic Analyzer Common----
    "1-3-1": "Please don't use twice {alias} like var_to.",
    "1-3-2": "No parent found for {varId_value}.",
    "1-3-3": "Overwriting a dataset/variable is not allowed, trying it with {varId_value}.",
    "1-3-4": "Cannot perform a rename with two equal values: {left_value} -> {right_value}.",
    "1-3-5": "{node_op} not found or not valid for {op_type}.",
    "1-3-6": "Hierarchical ruleset {node_value} not previously defined.",  #1-3-19
    "1-3-7": "Datapoint ruleset {node_value} not previously defined.",  #1-3-19
    "1-3-8": "Defined Operator {node_value} not previously defined.",  # 0-1-2-5
    "1-3-9": "Not valid set declaration, found duplicates {duplicates}.",
    "1-3-10": "Not valid set declaration, mixed scalar types {scalar_1} and {scalar_2}.",
    "1-3-11": "No Parameters {node_params} available for operation: {node_op}.",
    "1-3-12": "Default arguments cannot be followed by non-default arguments.",
    "1-3-13": "Not a valid {node_type} node kind, found {node_kind}.",
    "1-3-14": "Not valid data set {name} no Identifiers defined.",
    "1-3-15": "Missing datastructure definition for required input Dataset {input}.",
    "1-3-16": "Component {name} not found.",
    "1-3-17": "Operations without output assigned are not available.",
    "1-3-18": "No scalar outputs are available.",
    "1-3-19": "No {node_type} {node_value} found.",
    "1-3-20": "RuleComp of Hierarchical Ruleset can only be an identifier, found {name}.",
    "1-3-21": "Value {value} not valid, kind {node_kind}.",
    "1-3-22": "Unable to categorize {node_value}.",
    "1-3-23": "Missing value domain '{name}' definition, please provide an structure.",
    "1-3-24": "Internal error on Analytic operators inside a calc, No partition or order symbol found.",
    "1-3-25": "No symbols found.",
    "1-3-26": "collection or value domain not found.",  # TODO:
    "1-3-27": "Dataset without identifiers are not allowed in {op} operator.",
    "1-3-28": "At op {op}: invalid number of parameters: received {received}, expected at least: {expected}",
    "1-3-29": "At op {op}: can not use user defined operator that returns a component outside clause operator",
    "1-3-30": "At op {op}: too many parameters: received {received}, expected: {expected}",
    "1-3-31": "Cannot use component {name} outside an aggregate function in a having clause.",
    "1-3-32": "Cannot perform operation {op} inside having clause.",
    "1-3-33": "Having clause is not permitted if group by clause is not present.",
    "1-3-34": "At op {op}: Cannot use constant as a {type} parameter, found on {param}.",
    "1-3-35": "At op {op}: Cannot perform aggregation inside a calc.",
    # ---------------AST------------
    # AST Helpers
    "1-4-1-1": "At op {op}: User defined {option} declared as {type_1}, found {type_2}.",
    "1-4-1-2": "Using variable {value}, not defined at {op} definition.",
    "1-4-1-3": "At op {op}: using variable {value}, not defined as an argument.",
    "1-4-1-4": "Found duplicates at arguments naming, please review {type} definition {op}.",
    "1-4-1-5": "Found duplicates at rule naming: {name}. Please review {type} definition.",
    "1-4-1-6": "At op {op}: Arguments incoherence, {defined} defined {passed} passed.",
    "1-4-1-7": "All rules must be named or not named, but found mixed criteria at {type} definition {name}.",
    "1-4-1-8": "All rules must have different code items in the left side of '=' in hierarchy operator at hierachical ruleset definition {name}.",
    "1-4-1-9": "At op check_datapoint: {name} has an invalid datatype expected DataSet, found Scalar.",
    # AST Creation
    "1-4-2-1": "Eval could not be called without a {option} type definition.",
    "1-4-2-2": "Optional or empty expression node is not allowed in time_agg.",
    "1-4-2-3": "{value} could not be called in the count.",
    "1-4-2-4": "At op {op}: Only one order_by element must be used in Analytic with range windowing.",
    "1-4-2-5": "At op {op}: User defined operator without returns is not implemented.",
    # Not Implemented Error
}
