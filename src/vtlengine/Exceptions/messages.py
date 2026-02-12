"""
Exceptions.messages.py
======================

Description
-----------
All exceptions exposed by the Vtl engine.
"""
# -------------- Codification --------------
#
# -------------- INPUT ERRORS --------------
# 0-1-X-X = Input Validation Errors
# 0-2-X-X = JSON Schema Errors
# 0-3-X-X = DataLoad Errors
#
# -------------- SEMANTIC ERRORS --------------
# 1-1-X-X = Operators Semantic Errors
# 1-2-X-X = Semantic Analyzer Errors
# 1-3-X-X = AST Errors
#
# -------------- RUNTIME ERRORS --------------
# 2-X-X-X = RunTime Operator Errors

centralised_messages = {
    # Input Validation errors
    "0-1-0-1": {
        "message": "Trying to redefine input Datasets {dataset}.",
        "description": "Raised when the user attempts to redefine an input Dataset "
        "that already exists.",
    },
    "0-1-1-1": {
        "message": "invalid script format type: {format_}. Input must be a string, "
        "TransformationScheme or Path object",
        "description": "Occurs when the script input type is not valid. It must be "
        "a string, TransformationScheme, or Path object.",
    },
    "0-1-1-2": {
        "message": "The provided input {input} can not be used in this instance.",
        "description": "Raised when an input Dataset or value cannot be used in "
        "the current context.",
    },
    "0-1-1-3": {
        "message": "Invalid file extension: expected {expected_ext}, got {ext}",
        "description": "Occurs when a file has an incorrect extension compared "
        "to the expected one.",
    },
    "0-1-1-6": {
        "message": "Duplicated records. Combination of Identifiers are repeated.",
        "description": "Raised when duplicate rows are detected based on the Identifiers "
        "combination.",
    },
    "0-1-1-7": {
        "message": "G1 - The provided CSV file is empty.",
        "description": "Occurs when the input CSV file does not contain any data.",
    },
    "0-1-1-8": {
        "message": "The following Identifiers {ids} were not found , review file {file}.",
        "description": "Raised when certain expected Identifiers are missing in the input Dataset.",
    },
    "0-1-1-9": {
        "message": "You have a problem related with commas, review rfc4180 standard, "
        "review file {file}.",
        "description": "Occurs when CSV formatting issues are detected, particularly with commas, "
        "violating RFC4180.",
    },
    "0-1-1-11": {
        "message": "Wrong data in the file for this Scalar/Dataset {name}.",
        "description": "Raised when a Scalar or Dataset contains invalid or inconsistent data.",
    },
    "0-1-1-13": {
        "message": "Invalid key on {field} field: {key}{closest_key}.",
        "description": "Occurs when a key provided in the Dataset does not exist in the field.",
    },
    "0-1-1-14": {
        "message": "Empty Datasets {dataset1} and {dataset2} shape missmatch.",
        "description": "Raised when two Datasets are empty or have incompatible shapes.",
    },
    "0-1-2-3": {
        "message": "{element_type} '{element}' is/are duplicated.",
        "description": "Occurs when an element (e.g., Identifier or component) "
        "appears more than once.",
    },
    "0-1-2-5": {
        "message": "File {file} must be encoded in utf-8 (without BOM).",
        "description": "Raised when the file encoding is not UTF-8 without BOM.",
    },
    "0-1-2-6": {
        "message": "Not found scalar {name} in datastructures",
        "description": "Occurs when a scalar value expected in the data structures is missing.",
    },
    "0-1-2-7": {
        "message": "Invalid value '{value}' for type {type_} {op_type} {name}.",
        "description": "Occurs when a value does not match the expected type or operation "
        "constraints.",
    },
    "0-1-2-8": {
        "message": "Cannot pass as inputs Datasets/Scalars defined as outputs of transformations "
        "in the script, please check: {names}",
        "description": "Raised when an output of a transformation is incorrectly used as an input.",
    },
    # Run SDMX errors
    "0-1-3-1": {
        "message": "Expected exactly one input Dataset in the whole script, "
        "found: {number_Datasets}",
        "description": "Raised when the script expects exactly one input Dataset but finds "
        "more than one.",
    },
    "0-1-3-2": {
        "message": "SDMX Dataset {schema} requires to have a Schema object defined as structure",
        "description": "Occurs when an SDMX Dataset is missing its required schema definition.",
    },
    "0-1-3-3": {
        "message": "If no mappings are provided, only one Dataset is allowed.",
        "description": "Raised when multiple Datasets are provided without mappings, "
        "but only one is allowed.",
    },
    "0-1-3-4": {
        "message": "Dataset {short_urn} not found in mapping dictionary.",
        "description": "Occurs when the Datasets short URN is does not exists in the "
        "mapping dictionary.",
    },
    "0-1-3-5": {
        "message": "Dataset {dataset_name} not found in the input Datasets.",
        "description": "Raised when a Dataset expected as input is missing from the "
        "provided Datasets.",
    },
    "0-1-3-6": {
        "message": "Input name {missing} not found in the input Datasets.",
        "description": "Occurs when a named input Dataset cannot be found among "
        "the available inputs.",
    },
    "0-1-3-7": {
        "message": "Invalid input Datasets type: {type_}. Expected a sequence of PandasDataset.",
        "description": "Raised when the type of input Datasets is incorrect; "
        "a sequence of PandasDataset is expected.",
    },
    # JSON Schema errors
    "0-2-1-1": {
        "message": "The provided JSON does not follow the required JSON Schema",
        "description": "Occurs when a JSON input does not comply with the expected schema.",
    },
    "0-2-1-2": {
        "message": "Dataset {dataset} is not valid according to JSON schema",
        "description": "Raised when the Dataset does not conform to the expected JSON schema.",
    },
    # DataLoad errors
    "0-3-1-1": {
        "message": "{file} file not found. Please verify that the file exists and the provided "
        "path is correct.",
        "description": "Occurs when the specified file cannot be located at the given path.",
    },
    "0-3-1-2": {
        "message": "Output folder {folder} not found or invalid. Must be a valid Path or "
        "S3 directory.",
        "description": "Raised when the output folder path is missing or not valid "
        "for saving results.",
    },
    "0-3-1-3": {
        "message": "On Dataset {name} loading:  An Identifier cannot have null values, "
        "found null values on {null_identifier}.",
        "description": "Occurs when a Dataset Identifier contains null values, which "
        "is not allowed.",
    },
    "0-3-1-4": {
        "message": "On Dataset {name} loading: Datasets without Identifiers "
        "must have 0 or 1 datapoints.",
        "description": "Raised when a Dataset without Identifiers has more than one datapoint.",
    },
    "0-3-1-5": {
        "message": "On Dataset {name} loading: Component {comp_name} is missing in Datapoints.",
        "description": "Occurs when a required component is missing from the Dataset datapoints.",
    },
    "0-3-1-6": {
        "message": "On Dataset {name} loading: not possible to cast column {column} to {type}. "
        'Error found: "{error}"',
        "description": "Raised when a Dataset column cannot be cast to the expected data type.",
    },
    "0-3-1-7": {
        "message": "On Dataset {name} loading: Duplicated Identifiers are not allowed, "
        "found on row {row_index}",
        "description": "Occurs when a Dataset contains duplicated Identifiers, "
        "which is not allowed.",
    },
    "0-3-1-8": {
        "message": "Failed to load SDMX file '{file}': {error}",
        "description": "Raised when an SDMX file cannot be parsed by pysdmx.",
    },
    "0-3-1-9": {
        "message": "No datasets found in SDMX file '{file}'",
        "description": "Raised when an SDMX file contains no datasets.",
    },
    "0-3-1-10": {
        "message": "SDMX file '{file}' requires external structure file: {error}. "
        "Use run_sdmx() with a structure file for this format.",
        "description": "Raised when an SDMX file lacks embedded structure and needs an external "
        "structure file. Use run_sdmx() instead of run() for these files.",
    },
    "0-3-1-11": {
        "message": "Failed to load SDMX structure file '{file}': {error}",
        "description": "Raised when an SDMX structure file cannot be parsed by pysdmx.",
    },
    "0-3-1-12": {
        "message": "No data structures found in SDMX structure file '{file}'",
        "description": "Raised when an SDMX structure file contains no DataStructureDefinitions.",
    },
    # ------------Operators-------------
    # General Semantic errors
    "1-1-1-1": {
        "message": "Invalid implicit cast from {type_1} to {type_2}.",
        "description": "Raised when an implicit type conversion from {type_1} to {type_2} "
        "is not allowed.",
    },
    "1-1-1-2": {
        "message": "Invalid implicit cast from {type_1} and {type_2} to {type_check}.",
        "description": "Occurs when the combination of types {type_1} and {type_2} "
        "cannot be implicitly cast to {type_check}.",
    },
    "1-1-1-3": {
        "message": "At op {op}: {entity} {name} cannot be promoted to {target_type}.",
        "description": "Raised when a Dataset or Scalar cannot be promoted to "
        "the required target type in the operation {op}.",
    },
    "1-1-1-4": {
        "message": "At op {op}: Operation not allowed for multimeasure Datasets.",
        "description": "Occurs when an operation is attempted on a Dataset with multiple Measures, "
        "which is not permitted.",
    },
    "1-1-1-5": {
        "message": "At op {op}: Invalid type {type}.",
        "description": "Raised when an operand or component has an invalid type for "
        "the operation {op}.",
    },
    "1-1-1-8": {
        "message": "At op {op}: Invalid Dataset {name}, no Measures defined.",
        "description": "Occurs when a Dataset is expected to have Measures but none are defined.",
    },
    "1-1-1-9": {
        "message": "At op {op}: Invalid Dataset {name}, all Measures must have "
        "the same type: {type}.",
        "description": "Raised when Measures in a Dataset have different types each other.",
    },
    "1-1-1-10": {
        "message": "Component {comp_name} not found in Dataset {dataset_name}.",
        "description": "Occurs when a referenced component is missing from the Dataset.",
    },
    "1-1-1-13": {
        "message": "At op {op}: Component {comp_name} role must be '{role_1}', found '{role_2}'.",
        "description": "Raised when a Dataset component has an unexpected role for the operation.",
    },
    "1-1-1-15": {
        "message": "At op {op}: Datasets {name_1} and {name_2} does not contain the same "
        "number of {type}.",
        "description": "Occurs when two Datasets expected to have the same number of a "
        "specific type of components do not match.",
    },
    "1-1-1-16": {
        "message": "Found structure not nullable and null values.",
        "description": "Raised when null values are found in a structure that "
        "is defined as non-nullable.",
    },
    "1-1-1-20": {
        "message": "At op {op}: Only applies to Datasets, instead of this a Scalar was provided.",
        "description": "Occurs when a Scalar is provided to an operation "
        "that only supports Datasets.",
    },
    # Aggregate errors
    "1-1-2-2": {
        "message": "At op {op}: Only Identifiers are allowed for grouping, "
        "found {id_name} - {id_type}.",
        "description": "Raised when a non-Identifier component is used in a grouping operation.",
    },
    "1-1-2-3": {
        "message": "Having component output type must be boolean, found {type}.",
        "description": "Occurs when the output of a component in a HAVING clause "
        "is not boolean as required.",
    },
    # Analytic errors
    "1-1-3-2": {
        "message": "At op {op}: Only Identifiers are allowed for partitioning, "
        "found {id_name} - {id_type}.",
        "description": "Raised when a non-Identifier component is used as a "
        "partitioning key in an analytic operation.",
    },
    # Cast errors
    "1-1-5-1": {
        "message": "Type {type_1}, cannot be cast to {type_2}.",
        "description": "Occurs when an explicit or implicit cast between incompatible types "
        "is attempted.",
    },
    "1-1-5-3": {
        "message": "Impossible to cast from type {type_1} to {type_2}, without providing a mask.",
        "description": "Raised when a cast requires a mask to resolve ambiguities, "
        "but none is provided.",
    },
    "1-1-5-4": {
        "message": "Invalid mask to cast from type {type_1} to {type_2}.",
        "description": "Occurs when the mask provided for casting is invalid or incompatible "
        "with the types.",
    },
    "1-1-5-5": {
        "message": "A mask can't be provided to cast from type {type_1} to {type_2}. "
        "Mask provided: {mask_value}.",
        "description": "Raised when a mask is provided in a context where it should not be used "
        "for the cast.",
    },
    "2-1-5-1": {
        "message": "Impossible to cast {value} from type {type_1} to {type_2}.",
        "description": "Occurs when a value cannot be converted between the specified types.",
    },
    "2-1-5-2": {
        "message": "Value {value} has decimals, cannot cast to integer",
        "description": "Raised when attempting to cast a decimal value to an integer, "
        "which is not allowed.",
    },
    # Clause errors
    "1-1-6-2": {
        "message": "At op {op}: The Identifier {name} in Dataset {dataset} could not be included "
        "in the {op} op.",
        "description": "Raised when an Identifier cannot be included in the specified operation.",
    },
    "1-1-6-4": {
        "message": "At op {op}: Alias symbol cannot have the name of a "
        "component symbol: {symbol_name} - {comp_name}.",
        "description": "Occurs when an alias uses a name that conflicts with an "
        "existing component symbol.",
    },
    "1-1-6-5": {
        "message": "At op {op}: Scalar values are not allowed at sub operator, found {name}.",
        "description": "Raised when a Scalar is used where only Dataset components are allowed.",
    },
    "1-1-6-6": {
        "message": "Membership is not allowed inside a clause, found {dataset_name}#{comp_name}.",
        "description": "Occurs when a membership operation is attempted inside a clause, "
        "which is invalid.",
    },
    "1-1-6-7": {
        "message": "Cannot use component {comp_name} as it was generated in another "
        "calc expression.",
        "description": "Raised when trying to reuse a component generated in a "
        "different calculation expression.",
    },
    "1-1-6-8": {
        "message": "Cannot use component {comp_name} for rename, it is already in the "
        "Dataset {dataset_name}.",
        "description": "Occurs when attempting to rename a component that already "
        "exists in the Dataset.",
    },
    "1-1-6-9": {
        "message": "At op {op}: The following components are repeated: {from_components}.",
        "description": "Raised when duplicate components are detected in the operation.",
    },
    "1-1-6-10": {
        "message": "At op {op}: Component {operand} in Dataset {dataset_name} is not an Identifier",
        "description": "Occurs when a component expected to be an Identifier is not.",
    },
    "1-1-6-11": {
        "message": "Ambiguity for this variable {comp_name}, exists as a Scalar and component.",
        "description": "Raised when a variable name exists both as a Scalar and component, "
        "creating ambiguity.",
    },
    "1-1-6-12": {
        "message": "At op {op}: Not allowed to drop the last element.",
        "description": "Occurs when attempting to remove the last element, which is not permitted.",
    },
    "1-1-6-13": {
        "message": "At op {op}: Not allowed to overwrite an Identifier: {comp_name}",
        "description": "Raised when an operation attempts to overwrite an existing Identifier.",
    },
    # Comparison errors
    "1-1-7-1": {
        "message": "At op {op}: Value in {left_name} of type {left_type} is not comparable "
        "to value {right_name} of type {right_type}.",
        "description": "Occurs when attempting to compare values of incompatible types.",
    },
    # Conditional errors
    "1-1-9-1": {
        "message": "At op {op}: The evaluation condition must result in a Boolean expression, "
        "found '{type}'.",
        "description": "Raised when the condition in a conditional operation does not evaluate "
        "to Boolean.",
    },
    "1-1-9-3": {
        "message": "At op {op}: Then clause {then_name} and else clause {else_name}, "
        "both must be Scalars.",
        "description": "Occurs when then/else clauses are not both Scalars, which it is required.",
    },
    "1-1-9-4": {
        "message": "At op {op}: The condition Dataset {name} must contain an unique Measure.",
        "description": "Raised when the condition Dataset has multiple Measures "
        "instead of a single one.",
    },
    "1-1-9-5": {
        "message": "At op {op}: The condition Dataset Measure must be a Boolean, found '{type}'.",
        "description": "Occurs when the Measure in the condition Dataset is not Boolean.",
    },
    "1-1-9-6": {
        "message": "At op {op}: Then-else Datasets have different number of Identifiers compared "
        "with condition Dataset.",
        "description": "Raised when the then-else Datasets do not match the Identifier "
        "count of the condition Dataset.",
    },
    "1-1-9-9": {
        "message": "At op {op}: {clause} component {clause_name} role must be {role_1}, "
        "found {role_2}.",
        "description": "Occurs when a component in a clause has an incorrect role type.",
    },
    "1-1-9-10": {
        "message": "At op {op}: {clause} Dataset have different number of Identifiers compared "
        "with condition Dataset.",
        "description": "Raised when a Dataset in a clause has mismatched Identifier count.",
    },
    "1-1-9-11": {
        "message": "At op {op}: Condition component {name} must be Boolean, found {type}.",
        "description": "Occurs when a condition component is not Boolean.",
    },
    "1-1-9-12": {
        "message": "At op {op}: then clause {then_symbol} and else clause {else_symbol}, "
        "both must be Datasets or at least one of them a Scalar.",
        "description": "Raised when then/else clauses do not meet required "
        "type rules (Dataset/Scalar).",
    },
    "1-1-9-13": {
        "message": "At op {op}: then {then} and else {else_clause} Datasets must contain "
        "the same number of components.",
        "description": "Occurs when then and else Datasets have differing numbers of components.",
    },
    "2-1-9-1": {
        "message": "At op {op}: Condition operators must have the same operator type.",
        "description": "Raised when condition operators differ in type, which is invalid.",
    },
    "2-1-9-2": {
        "message": "At op {op}: Condition {name} it's not a boolean.",
        "description": "Occurs when a condition variable is not Boolean.",
    },
    "2-1-9-3": {
        "message": "At op {op}: All then and else operands must be Scalars.",
        "description": "Raised when then/else operands are not all Scalars.",
    },
    "2-1-9-4": {
        "message": "At op {op}: Condition {name} must be boolean type.",
        "description": "Occurs when the condition variable is not of Boolean type.",
    },
    "2-1-9-5": {
        "message": "At op {op}: Condition Dataset {name} Measure must be Boolean.",
        "description": "Raised when the Measure in a condition Dataset is not Boolean.",
    },
    "2-1-9-6": {
        "message": "At op {op}: At least a then or else operand must be Dataset.",
        "description": "Occurs when neither then nor else operands are Datasets.",
    },
    "2-1-9-7": {
        "message": "At op {op}: All Dataset operands must have the same components.",
        "description": "Raised when Dataset operands have mismatched components.",
    },
    # Data Validation errors
    "1-1-10-1": {
        "message": "At op {op}: The {op_type} operand must have exactly one Measure "
        "of type {me_type}",
        "description": "Raised when an operand does not have exactly one Measure "
        "of the required type.",
    },
    "1-1-10-2": {
        "message": "At op {op}: Number of variable has to be equal between the call and signature.",
        "description": "Occurs when the number of variables in the call does not match the "
        "function signature.",
    },
    "1-1-10-3": {
        "message": "At op {op}: Name in the call {found} has to be equal to variable rule "
        "in signature {expected}.",
        "description": "Raised when a variable name in the call differs "
        "from the expected signature.",
    },
    "1-1-10-4": {
        "message": "At op {op}: When a hierarchical ruleset is defined for value domain, "
        "it is necessary to specify the component with the rule clause on call.",
        "description": "Occurs when a hierarchical ruleset defined for value domain,"
        "requires a component but it is missing in the call.",
    },
    "1-1-10-5": {
        "message": "No rules to analyze on Hierarchy Roll-up as rules have no = operator.",
        "description": "Raised when there are no applicable rules in a Hierarchy Roll-up "
        "due to missing '=' operators.",
    },
    "1-1-10-6": {
        "message": "At op {op}: Name in the call {found} has to be equal to variable condition "
        "in signature {expected} .",
        "description": "Occurs when a variable name in the call does not match the "
        "expected condition in the signature.",
    },
    "1-1-10-7": {
        "message": "Not found component {comp_name} on signature.",
        "description": "Raised when a component referenced in the call is not found in the "
        "signature.",
    },
    "1-1-10-8": {
        "message": "At op {op}: Measures involved have to be numerical, other types found {found}.",
        "description": "Occurs when operands involve non-numerical Measures "
        "where numerical are required.",
    },
    "1-1-10-9": {
        "message": "Invalid signature for the ruleset {ruleset}. On variables, condComp and "
        "ruleComp must be the same",
        "description": "Raised when condComp and ruleComp in a ruleset signature do not "
        "match as required.",
    },
    # General Operators
    "2-1-12-1": {
        "message": "At op {op}: Create a null Measure without a Scalar type is not allowed. "
        "Please use Cast operator.",
        "description": "Raised when attempting to create a null Measure without specifying a "
        "Scalar type; a Cast operator must be used.",
    },  # RunTimeError.
    # Join Operators
    "1-1-13-1": {
        "message": "At op {op}: Duplicated alias {duplicates}.",
        "description": "Raised when an alias is used more than once in the same operation.",
    },
    "1-1-13-2": {
        "message": "At op {op}: Missing mandatory aliasing.",
        "description": "Occurs when a required alias for an operation is not provided.",
    },
    "1-1-13-3": {
        "message": "At op {op}: Join conflict with duplicated names for "
        "column {name} from original Datasets.",
        "description": "Raised when a Join operation encounters column "
        "name conflicts across input Datasets.",
    },
    "1-1-13-4": {
        "message": "At op {op}: Using clause, using={using_names}, does not define all the "
        "Identifiers, of non reference Dataset {dataset}.",
        "description": "Occurs when a 'using' clause in a join does not cover all Identifiers of a "
        "non-reference Dataset.",
    },
    "1-1-13-5": {
        "message": "At op {op}: Invalid subcase B1, All the Datasets must share as "
        "Identifiers the using ones.",
        "description": "Raised when not all Datasets in subcase B1 share the declared 'using' "
        "Identifiers.",
    },
    "1-1-13-6": {
        "message": "At op {op}: Invalid subcase B2, All the declared using components "
        "'{using_components}' must be present as components in the reference Dataset"
        " '{reference}'.",
        "description": "Occurs when components declared in 'using' are missing from the "
        "reference Dataset in subcase B2.",
    },
    "1-1-13-7": {
        "message": "At op {op}: Invalid subcase B2, All the non reference Datasets must "
        "share as Identifiers the using ones.",
        "description": "Raised when non-reference Datasets in subcase B2 do not share "
        "the declared 'using' Identifiers.",
    },
    "1-1-13-8": {
        "message": "At op {op}: No available using clause.",
        "description": "Occurs when a join operation requires a 'using' clause but none "
        "is provided.",
    },
    "1-1-13-9": {
        "message": "Ambiguity for this variable {comp_name} inside a Join clause.",
        "description": "Raised when a component name is ambiguous in a Join operation.",
    },
    "1-1-13-10": {
        "message": "The join operator does not perform Scalar/component operations.",
        "description": "Occurs when attempting Scalar or component operations directly with "
        "a Join operator.",
    },
    "1-1-13-11": {
        "message": "At op {op}: Invalid subcase A, {dataset_reference} should be a superset but "
        "{component} not found.",
        "description": "Raised when a subcase A join expects a superset but a required component "
        "is missing.",
    },
    "1-1-13-12": {
        "message": "At op {op}: Invalid subcase A. There are different Identifiers for the "
        "provided Datasets",
        "description": "Occurs when Datasets involved in subcase A have differing Identifiers.",
    },
    "1-1-13-13": {
        "message": "At op {op}: Invalid subcase A. There are not same number of Identifiers "
        "for the provided Datasets",
        "description": "Raised when Datasets in subcase A do not have the same number of "
        "Identifiers.",
    },
    "1-1-13-14": {
        "message": "Cannot perform a join over a Dataset Without Identifiers: {name}.",
        "description": "Occurs when attempting to join a Dataset that lacks Identifiers.",
    },
    "1-1-13-15": {
        "message": "At op {op}: {comp_name} has to be a Measure for all the provided Datasets "
        "inside the Join clause",
        "description": "Raised when a component is not a Measure in all Datasets required "
        "for the Join clause.",
    },
    "1-1-13-16": {
        "message": "At op {op}: Invalid use, please review : {msg}.",
        "description": "Occurs for general invalid use cases in join operations.",
    },
    "1-1-13-17": {
        "message": "At op {op}: {comp_name} not present in the Dataset(result from join VDS) "
        "at the time it is called",
        "description": "Raised when a component is missing from the join result Dataset "
        "when it is referenced.",
    },
    "1-1-13-18": {
        "message": "At op {op}: Incompatible types for common identifier {id_name}: "
        "{type_1} and {type_2}.",
        "description": "Raised when datasets in a join operation have a common identifier "
        "with incompatible types.",
    },
    # Operators general errors
    "1-1-14-1": {
        "message": "At op {op}: Measure names don't match: {left} - {right}.",
        "description": "Occurs when Measure names do not match across Datasets in an operation.",
    },
    "1-1-14-3": {
        "message": "At op {op}: Invalid Scalar types for Identifiers at Dataset {dataset}. "
        "One {type} Identifier expected, {count} found.",
        "description": "Raised when the Dataset has an unexpected number or type of Identifiers.",
    },
    "1-1-14-5": {
        "message": "At op {op}: {names} with type/s {types} is not compatible with {op}",
        "description": "Occurs when the specified components/types are incompatible "
        "with the operation.",
    },
    "1-1-14-6": {
        "message": "At op {op}: {comp_name} with type {comp_type} and Scalar_set with "
        "type {Scalar_type} is not compatible with {op}",
        "description": "Raised when a component and a Scalar set have incompatible "
        "types for an operation.",
    },
    "1-1-14-9": {
        "message": "At op {op}: {names} with type/s {types} is not compatible with {op} on "
        "Datasets {datasets}.",
        "description": "Occurs when components/types across multiple Datasets are incompatible "
        "with the operation.",
    },
    # Numeric Operators
    "1-1-15-8": {
        "message": "At op {op}: {op} operator cannot have a {comp_type} as parameter.",
        "description": "Raised when an operator receives a component type that "
        "is not allowed as a parameter.",
    },
    "2-1-15-1": {
        "message": "At op {op}: Component {comp_name} from Dataset {dataset_name} "
        "contains negative values.",
        "description": "Runtime error raised when a Dataset component contains negative "
        "values that are not allowed.",
    },  # RunTimeError.
    "2-1-15-2": {
        "message": "At op {op}: Value {value} could not be negative.",
        "description": "Runtime error when a value is negative but must be non-negative.",
    },  # RunTimeError.
    "2-1-15-3": {
        "message": "At op {op}: Base value {value} could not be less or equal 0.",
        "description": "Runtime error when a base value is less than or equal to zero, "
        "which is not allowed.",
    },  # RunTimeError.
    "2-1-15-4": {
        "message": "At op {op}: Invalid values in Component {name}.",
        "description": "Runtime error when a component contains invalid values.",
    },  # RunTimeError.
    "2-1-15-5": {
        "message": "At op {op}: Invalid values in {name_1} and {name_2}.",
        "description": "Runtime error when two components contain invalid values.",
    },  # RunTimeError.
    "2-1-15-6": {
        "message": "At op {op}: Scalar division by Zero.",
        "description": "Runtime error when attempting to divide a Scalar by zero.",
    },  # RunTimeError.
    "2-1-15-7": {
        "message": "At op {op}: {op} operator cannot be a Dataset.",
        "description": "Runtime error when an operator is incorrectly applied to a Dataset "
        "instead of allowed types.",
    },  # RunTimeError.
    # Set Operators
    "1-1-17-1": {
        "message": "At op {op}: Datasets {dataset_1} and {dataset_2} have different number "
        "of components",
        "description": "Raised when set operations are performed on Datasets with differing "
        "numbers of components.",
    },
    # String Operators
    "1-1-18-1": {
        "message": "At op {op}: Invalid Dataset {name}. Dataset with one Measure expected.",
        "description": "Raised when a string operation expects a Dataset with exactly one Measure.",
    },
    "1-1-18-2": {
        "message": "At op {op}: Composition of Dataset and Component is not allowed.",
        "description": "Occurs when attempting to combine a Dataset and a component in an "
        "unsupported way.",
    },
    "1-1-18-3": {
        "message": "At op {op}: Invalid parameter position: {pos}.",
        "description": "Raised when a parameter is supplied at an invalid position.",
    },
    "1-1-18-4": {
        "message": "At op {op}: {param_type} parameter should be {correct_type}.",
        "description": "Occurs when a parameter type does not match the expected type.",
    },
    "1-1-18-6": {
        "message": "At op {op}: Datasets have different Measures.",
        "description": "Raised when Datasets involved in a string operation "
        "do not have matching Measures.",
    },
    "1-1-18-7": {
        "message": "At op {op}: Invalid number of parameters {number}, {expected} expected.",
        "description": "Occurs when the number of parameters supplied is incorrect.",
    },
    "1-1-18-8": {
        "message": "At op {op}: {msg} in regexp: {regexp},  in position {pos}.",
        "description": "Raised when a string pattern or regex fails at a specific position.",
    },
    "1-1-18-10": {
        "message": "At op {op}: Cannot have a Dataset as parameter",
        "description": "Occurs when a Dataset is incorrectly used as a parameter in a "
        "string operation.",
    },
    # Time operators
    "1-1-19-1": {
        "message": "At op {op}: {op} must have a {data_type} type on {comp}.",
        "description": "Raised when a Time operator is applied to a component "
        "with an incorrect data type.",
    },
    "1-1-19-2": {
        "message": "At op {op}: Unknown Date type for {op}.",
        "description": "Occurs when the Date type of a component is unknown or "
        "unsupported for the operation.",
    },
    "1-1-19-3": {
        "message": "At op {op}: Invalid {param} for {op}.",
        "description": "Raised when a parameter value for a Time operator is invalid.",
    },
    "1-1-19-4": {
        "message": "At op {op}: Invalid values {value_1} and {value_2}, periodIndTo parameter "
        "must be a larger Duration value than periodIndFrom parameter.",
        "description": "Occurs when periodIndTo is not greater than periodIndFrom "
        "in a Time aggregation.",
    },
    "1-1-19-5": {
        "message": "At op {op}: periodIndTo parameter must be a larger duration value than "
        "the values to aggregate.",
        "description": "Raised when the periodIndTo parameter is too short compared "
        "with the values to aggregate.",
    },
    "1-1-19-6": {
        "message": "At op {op}: Time type used in the component {comp} is not supported.",
        "description": "Occurs when a component has a time type that is unsupported "
        "for the operation.",
    },
    "1-1-19-7": {
        "message": "At op {op}: can be applied only on Data Sets (of time series) and returns "
        "a Data Set (of time series).",
        "description": "Raised when a time operation is applied to an unsupported type; only "
        "Time series Datasets are allowed.",
    },
    "1-1-19-8": {
        "message": "At op {op}: {op} can only be applied to a {comp_type}",
        "description": "Occurs when the time operator is applied to a component of the wrong type.",
    },
    "1-1-19-9": {
        "message": "At op {op}: {op} can only be applied to a {comp_type} with a {param}",
        "description": "Raised when a time operator requires a component with a specific "
        "additional parameter type.",
    },
    "1-1-19-10": {
        "message": "{op} can only be applied to operands with data type as Date or Time Period",
        "description": "Occurs when operands of a time operator are not of type Date or "
        "Time Period.",
    },
    "1-1-19-11": {
        "message": "At op time_agg: If used over a Date, first/last parameter must be declared.",
        "description": "Raised when the first/last parameter is missing in a time aggregation "
        "over a Date type.",
    },
    # ---------Semantic Analyzer Common----
    "1-2-1": {
        "message": "Please don't use twice {alias} like var_to.",
        "description": "Raised when the same alias is used more than once in a variable "
        "assignment.",
    },
    "1-2-2": {
        "message": "Overwriting a Dataset/variable is not allowed, trying it with {varId_value}.",
        "description": "Occurs when an attempt is made to overwrite an existing "
        "Dataset or variable.",
    },
    "1-2-3": {
        "message": "{node_op} not found or not valid for {op_type}.",
        "description": "Occurs when an operator node is not recognized or incompatible with the "
        "operation type.",
    },
    "1-2-4": {
        "message": "Defined Operator {node_value} not previously defined.",
        "description": "Raised when a user-defined operator is used before being defined.",
    },
    "1-2-5": {
        "message": "Operations without output assigned are not available.",
        "description": "Occurs when attempting to execute operations that require an assigned "
        "output but none is provided.",
    },
    "1-2-6": {
        "message": "No {node_type} {node_value} found.",
        "description": "Raised when a required node of a specific type is missing.",
    },
    "1-2-7": {
        "message": "RuleComp of Hierarchical Ruleset can only be an Identifier, "
        "{name} is a {role}.",
        "description": "Occurs when a rule component in a hierarchical ruleset "
        "is not an Identifier.",
    },
    "1-2-8": {
        "message": "Missing value domain '{name}' definition, please provide an structure.",
        "description": "Raised when a required value domain is missing its definition.",
    },
    "1-2-9": {
        "message": "Value domain {name} not found.",
        "description": "Raised when a specified value domain cannot be found.",
    },
    "1-2-10": {
        "message": "Dataset without Identifiers are not allowed in {op} operator.",
        "description": "Occurs when an operator is applied to a Dataset lacking Identifiers.",
    },
    "1-2-11": {
        "message": "At op {op}: invalid number of parameters: received {received}, "
        "expected at least: {expected}",
        "description": "Raised when the number of parameters provided to an operation is "
        "less than expected.",
    },
    "1-2-12": {
        "message": "At op {op}: can not use user defined operator that returns a component outside "
        "Clause operator or Rule",
        "description": "Occurs when a user-defined operator returning a component is used outside "
        "an allowed context.",
    },
    "1-2-13": {
        "message": "Having clause is not permitted if group by clause is not present.",
        "description": "Occurs when a HAVING clause is used without a corresponding "
        "GROUP BY clause.",
    },
    "1-2-14": {
        "message": "At op {op}: Cannot perform aggregation inside a Calc.",
        "description": "Occurs when an aggregation operation is attempted inside a "
        "Calc expression.",
    },
    # AST Helpers
    "1-3-1-1": {
        "message": "At op {op}: User defined {option} declared as {type_1}, found {type_2}.",
        "description": "Occurs when a user-defined option has a type mismatch in its declaration.",
    },
    "1-3-1-2": {
        "message": "Using variable {value}, not defined at {op} definition.",
        "description": "Raised when a variable is used without being defined in "
        "the operation's context.",
    },
    "1-3-1-3": {
        "message": "At op {op}: using variable {value}, not defined as an argument.",
        "description": "Occurs when a variable is referenced but not declared as an argument.",
    },
    "1-3-1-4": {
        "message": "Found duplicates at arguments naming, please review {type} definition {op}.",
        "description": "Raised when duplicate argument names are found in a definition.",
    },
    "1-3-1-5": {
        "message": "Found duplicates at rule naming: {names}. "
        "Please review {type} {ruleset_name} definition.",
        "description": "Occurs when multiple rules share the same name in a ruleset.",
    },
    "1-3-1-6": {
        "message": "At op {op}: Arguments incoherence, {defined} defined {passed} passed.",
        "description": "Raised when the number of defined arguments and passed arguments "
        "do not match.",
    },
    "1-3-1-7": {
        "message": "All rules must be named or not named, but found mixed criteria at "
        "{type} definition {name}.",
        "description": "Occurs when some rules are named and others are not in the same ruleset.",
    },
    "1-3-1-8": {
        "message": "All rules must have different code items in the left side of '=' in "
        "hierarchy operator at hierachical ruleset definition {name}.",
        "description": "Raised when duplicate left-hand side code items are found "
        "in hierarchical ruleset definitions.",
    },
    "1-3-1-9": {
        "message": "At op check_datapoint: {name} has an invalid datatype expected Dataset, "
        "found Scalar.",
        "description": "Occurs when a datapoint has an invalid datatype; a Dataset is expected "
        "but a Scalar was found.",
    },
    "1-3-12": {
        "message": "Default arguments cannot be followed by non-default arguments.",
        "description": "Occurs when a function definition places non-default parameters "
        "after default ones.",
    },
    # AST Creation
    "1-3-2-0": {
        "message": "Error creating DAG.",
        "description": "Raised when the DAG (Directed Acyclic Graph) creation fails.",
    },
    "1-3-2-1": {
        "message": "Eval could not be called without a {option} type definition.",
        "description": "Occurs when an evaluation is attempted without a necessary "
        "type definition.",
    },
    "1-3-2-2": {
        "message": "At op {op}: User defined operator without returns is not implemented.",
        "description": "Occurs when a user-defined operator lacks a return definition.",
    },
    "1-3-2-3": {
        "message": "At op {op}: Vtl Script contains Cycles, no DAG established. "
        "Nodes involved: {nodes}.",
        "description": "Raised when cyclic dependencies are detected in a VTL script, "
        "preventing DAG creation.",
    },
    "1-3-2-4": {
        "message": "The Time aggregation operand has to be defined if not used inside an "
        "aggregation.",
        "description": "Raised when a Time aggregation operator is missing the operand "
        "definition outside an aggregation context.",
    },
    # ---------- Interpreter ----------
    "1-3-5": {
        "message": "{node_op} not found or not valid for {op_type}.",
        "description": "Occurs when an operator is undefined or incompatible with "
        "the given operation type.",
    },
    "1-3-6": {
        "message": "Language {language} not supported on Eval operator. Only"
        " SQL language is supported.",
        "description": "Raised when an unsupported language is specified in an Eval operation.",
    },
    # ---------- RunTimeErrors ----------
    "2-1-19-1": {
        "message": "At op time_agg: Invalid value {value} to aggregate to "
        "periodIndTo {new_indicator} , "
        "periodIndTo parameter must be a larger duration value than the values to aggregate.",
        "description": "Raised when the periodIndTo parameter is smaller than "
        "the values being aggregated.",
    },
    "2-1-19-2": {
        "message": "Invalid period indicator {period}.",
        "description": "Occurs when the specified period indicator is not recognized or valid.",
    },
    "2-1-19-3": {
        "message": "Only same period indicator allowed for both parameters "
        "({period1} != {period2}).",
        "description": "Raised when two parameters have different period indicators where "
        "the same indicator is required.",
    },
    "2-1-19-4": {
        "message": "Date setter, ({value} > {date}). Cannot set date1 with a "
        "value higher than date2.",
        "description": "Occurs when date1 is set to a value greater than date2, which is invalid.",
    },
    "2-1-19-5": {
        "message": "Date setter, ({value} < {date}). Cannot set date2 with a value lower "
        "than date1.",
        "description": "Occurs when date2 is set to a value less than date1, which is invalid.",
    },
    "2-1-19-6": {
        "message": "Invalid period format, must be YYYY-(L)NNN: {period_format}",
        "description": "Raised when a period string does not match the expected format.",
    },
    "2-1-19-7": {
        "message": "Period Number must be between 1 and {periods} for "
        "period indicator {period_indicator}.",
        "description": "Occurs when the period number is outside the valid range "
        "for the given period indicator.",
    },
    "2-1-19-8": {
        "message": "Invalid date format, must be YYYY-MM-DD: {date}",
        "description": "Raised when a date does not follow the required YYYY-MM-DD format.",
    },
    "2-1-19-9": {
        "message": "Invalid day {day} for year {year}.",
        "description": "Occurs when the day value is invalid for the given year.",
    },
    "2-1-19-10": {
        "message": "Invalid year {year}, must be between 1900 and 9999.",
        "description": "Raised when the year is outside the allowed range.",
    },
    "2-1-19-11": {
        "message": "{op} operator is not compatible with time values",
        "description": "Occurs when a time operator is applied to incompatible values.",
    },
    "2-1-19-12": {
        "message": "At op {op}: Invalid param type {type} for param {name}, expected {expected}.",
        "description": "Raised when a parameter has a type different from what is expected.",
    },
    "2-1-19-13": {
        "message": "At op {op}: Invalid param data_type {type} for param {name}, "
        "expected {expected}.",
        "description": "Occurs when a parameter has an invalid data type for the operation.",
    },
    "2-1-19-14": {
        "message": "At op {op}: Invalid Dataset {name}, requires at least one "
        "Date/Time_Period Measure.",
        "description": "Raised when a Dataset lacks required Date/Time_Period Measures.",
    },
    "2-1-19-15": {
        "message": "{op} can only be applied according to the iso 8601 format mask",
        "description": "Occurs when the operator is applied to values not conforming to "
        "ISO 8601 format.",
    },
    "2-1-19-16": {
        "message": "{op} can only be positive numbers",
        "description": "Raised when the operator is applied to non-positive numbers.",
    },
    "2-1-19-17": {
        "message": "At op {op}: Time operators comparison are only support = and <> comparison "
        "operations",
        "description": "Occurs when a time operator comparison uses unsupported operators "
        "like < or >.",
    },
    "2-1-19-18": {
        "message": "At op {op}: Time operators do not support < and > comparison operations, "
        "so its not possible to use get the max or min between two time operators",
        "description": "Raised when attempting to compute max or min using unsupported "
        "Time operator comparisons.",
    },
    "2-1-19-19": {
        "message": "Time Period comparison (>, <, >=, <=) with different period indicator "
        "is not supported, found {value1} {op} {value2}",
        "description": "Occurs when comparing two time periods with different indicators "
        "using unsupported Comparison operators.",
    },
    "2-1-19-20": {
        "message": "Time Period operands with different period indicators do not support < and > "
        "Comparison operations, unable to get the {op}",
        "description": "Raised when < or > comparisons are attempted between operands "
        "with different period indicators.",
    },
    # ----------- Interpreter Common ------
    "2-3-4": {
        "message": "{op} operator must have a {comp}",
        "description": "Occurs when a required component is missing for the operator.",
    },
    "2-3-5": {
        "message": "Expected {param_type}, got {type_name} on UDO {op}, parameter {param_name}.",
        "description": "Raised when a user-defined operator receives a parameter of "
        "an unexpected type.",
    },
    "2-3-6": {
        "message": "Dataset or Scalar {dataset_name} not found, please check input datastructures.",
        "description": "Occurs when an input Dataset or Scalar is missing.",
    },
    "2-3-7": {
        "message": "Ruleset Dataset not found, please check the ruleset definition.",
        "description": "Occurs when the Ruleset Dataset is missing.",
    },
    "2-3-9": {
        "message": "{comp_type} {comp_name} not found in {param}.",
        "description": "Raised when a component is not found within a specified parameter.",
    },
    "2-3-10": {
        "message": "No {comp_type} have been defined.",
        "description": "Occurs when no components of the required type have been defined.",
    },
    "2-3-11": {
        "message": "{pos} operand must be a Dataset.",
        "description": "Raised when an operand expected to be a Dataset is not a Dataset.",
    },
}
