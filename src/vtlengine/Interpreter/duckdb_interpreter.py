from typing import Any, Optional, List, Dict

import duckdb

from vtlengine import AST as AST
from vtlengine.AST.ASTTemplate import ASTTemplate
from vtlengine.Model import Dataset


class InterpreterAnalyzer(ASTTemplate):
    # Model elements
    datasets: Dict[str, Dataset]
    value_domains: Optional[Dict[str, ValueDomain]] = None
    external_routines: Optional[Dict[str, ExternalRoutine]] = None
    # Analysis mode
    only_semantic: bool = False
    # Memory efficient
    ds_analysis: Optional[Dict[str, Any]] = None
    datapoints_paths: Optional[Dict[str, Path]] = None
    output_path: Optional[Union[str, Path]] = None
    # Time Period Representation
    time_period_representation: Optional[TimePeriodRepresentation] = None
    # Flags to change behavior
    nested_condition: Union[str, bool] = False
    is_from_assignment: bool = False
    is_from_component_assignment: bool = False
    is_from_regular_aggregation: bool = False
    is_from_grouping: bool = False
    is_from_having: bool = False
    is_from_if: bool = False
    is_from_rule: bool = False
    is_from_join: bool = False
    is_from_condition: bool = False
    is_from_hr_val: bool = False
    is_from_hr_agg: bool = False
    condition_stack: Optional[List[str]] = None
    # Handlers for simplicity
    regular_aggregation_dataset: Optional[Dataset] = None
    aggregation_grouping: Optional[List[str]] = None
    aggregation_dataset: Optional[Dataset] = None
    then_condition_dataset: Optional[List[Any]] = None
    else_condition_dataset: Optional[List[Any]] = None
    ruleset_dataset: Optional[Dataset] = None
    rule_data: Optional[duckdb.DuckDBPyRelation] = None
    ruleset_signature: Optional[Dict[str, str]] = None
    udo_params: Optional[List[Dict[str, Any]]] = None
    hr_agg_rules_computed: Optional[Dict[str, pd.DataFrame]] = None
    ruleset_mode: Optional[str] = None
    hr_input: Optional[str] = None
    hr_partial_is_valid: Optional[List[bool]] = None
    hr_condition: Optional[Dict[str, str]] = None
    # DL
    dprs: Optional[Dict[str, Optional[Dict[str, Any]]]] = None
    udos: Optional[Dict[str, Optional[Dict[str, Any]]]] = None
    hrs: Optional[Dict[str, Optional[Dict[str, Any]]]] = None

    def _load_datapoints_efficient(self, statement_num: int):
        pass

    def _save_datapoints_efficient(self, statement_num: int):
        pass

    def visit_Start(self, node: AST.Start) -> Any:
        # Start node is the root of the AST, we can start processing from here
        querys = []
        for statement in node.statements:
            querys.append(self.visit(statement))
        return None
