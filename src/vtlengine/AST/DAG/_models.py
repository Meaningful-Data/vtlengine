from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class StatementDeps:
    """Per-statement dependency info tracked during AST visiting.

    Attributes:
        inputs: Variables consumed by this statement (excluding its own outputs).
        outputs: Variables produced by this statement via ``:=`` assignment.
        persistent: Variables produced by this statement via ``<-`` assignment.
        unknown_variables: Variables inside RegularAggregation context that could
            be either components of the dataset or external scalars.
        has_dataset_op: Whether this statement involves a dataset operation
            (RegularAggregation, JoinOp, Aggregation, Analytic, MEMBERSHIP,
            UDO with dataset params, etc.).
        dataset_inputs: Subset of inputs that are definitively datasets
            (e.g., UDO params typed as dataset). Empty means all inputs in a
            ``has_dataset_op`` statement are considered dataset inputs.
        scalar_inputs: Subset of inputs that are definitively scalars
            (e.g., UDO params typed as a scalar type like number, string, etc.).
    """

    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    persistent: List[str] = field(default_factory=list)
    unknown_variables: List[str] = field(default_factory=list)
    has_dataset_op: bool = False
    dataset_inputs: List[str] = field(default_factory=list)
    scalar_inputs: List[str] = field(default_factory=list)


@dataclass
class Schedule:
    """Typed result of DAG dataset usage analysis.

    Tracks when datasets should be loaded/unloaded for memory-efficient execution,
    and classifies global inputs into four categories based on AST context.

    Attributes:
        insertion: Statement index to list of datasets to load at that point
            (first use).
        deletion: Statement index to list of datasets to unload at that point
            (last use).
        global_inputs: All external dependencies not produced by the script.
            Union of the four ``global_input_*`` categories below (no duplicates).
        global_input_datasets: Definite datasets — used in dataset operations
            (RegularAggregation operand, Identifier with kind="DatasetID",
            UDO dataset params, MEMBERSHIP left operand, JoinOp, etc.).
        global_input_scalars: Definite scalars — feed exclusively into scalar
            chains propagated from constant assignments with no dataset ops.
        global_input_dataset_or_scalar: Ambiguous at top level (e.g.,
            ``DS_r <- X + 2`` where X could be a dataset or scalar).
            The caller may provide either.
        global_input_component_or_scalar: Ambiguous inside RegularAggregation
            (e.g., ``DS_1[calc Me_2 := Me_1 + X]`` where X could be a component
            of DS_1 or an external scalar). Semantic error 1-1-6-11 is raised
            at runtime if it collides with a component name.
        persistent: Outputs written with ``<-`` (persistent assignment).
        all_outputs: All variables produced by the script (both ``:=`` and
            ``<-`` assignments).
    """

    insertion: Dict[int, List[str]] = field(default_factory=dict)
    deletion: Dict[int, List[str]] = field(default_factory=dict)
    global_inputs: List[str] = field(default_factory=list)
    global_input_datasets: List[str] = field(default_factory=list)
    global_input_scalars: List[str] = field(default_factory=list)
    global_input_dataset_or_scalar: List[str] = field(default_factory=list)
    global_input_component_or_scalar: List[str] = field(default_factory=list)
    persistent: List[str] = field(default_factory=list)
    all_outputs: List[str] = field(default_factory=list)
