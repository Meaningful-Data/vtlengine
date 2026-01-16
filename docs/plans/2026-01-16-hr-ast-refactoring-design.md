# AST Refactoring for Hierarchical and Datapoint Validation Operators

## Overview

Refactor the AST representation for `check_datapoint`, `check_hierarchy`, and `hierarchy` operators. Currently these use the generic `ParamOp` node. This design introduces dedicated AST nodes with typed fields and enums for better type safety, clarity, and maintainability.

## Goals

1. **Type safety & clarity** - Dedicated node types that explicitly represent HR/validation operators
2. **Reduce complexity in the interpreter** - Simplify the large `visit_ParamOp` method by having separate visitor methods
3. **Shared structure extraction** - Explicitly represent common fields (dataset, ruleset_name, conditions, modes) rather than packing into generic `children`/`params` lists

## Design

### New Enums (in AST/__init__.py)

```python
from enum import Enum

class ValidationMode(Enum):
    """Validation mode for hierarchy operations."""
    NON_NULL = "non_null"
    NON_ZERO = "non_zero"
    PARTIAL_NULL = "partial_null"
    PARTIAL_ZERO = "partial_zero"
    ALWAYS_NULL = "always_null"
    ALWAYS_ZERO = "always_zero"

class HRInputMode(Enum):
    """Input mode for hierarchy operator."""
    RULE = "rule"
    DATASET = "dataset"
    RULE_PRIORITY = "rule_priority"

class CHInputMode(Enum):
    """Input mode for check_hierarchy operator."""
    DATASET = "dataset"
    DATASET_PRIORITY = "dataset_priority"

class ValidationOutput(Enum):
    """Output mode for check_datapoint and check_hierarchy."""
    INVALID = "invalid"
    ALL = "all"
    ALL_MEASURES = "all_measures"

class HierarchyOutput(Enum):
    """Output mode for hierarchy operator."""
    COMPUTED = "computed"
    ALL = "all"
```

### New AST Nodes (in AST/__init__.py)

#### HROperation

Covers both `hierarchy` and `check_hierarchy` operators:

```python
@dataclass
class HROperation(AST):
    """
    HROperation: Hierarchical ruleset operations (hierarchy, check_hierarchy)

    op: "hierarchy" or "check_hierarchy"
    dataset: The input dataset expression
    ruleset_name: Name of the hierarchical ruleset (HRuleset)
    rule_component: Optional component ID for the RULE clause
    conditions: List of condition components (from conditionClause)
    validation_mode: Mode for validation (non_null, non_zero, etc.)
    input_mode: Input mode - HRInputMode for hierarchy, CHInputMode for check_hierarchy
    output: Output mode - HierarchyOutput for hierarchy, ValidationOutput for check_hierarchy
    """
    op: str
    dataset: AST
    ruleset_name: str
    rule_component: Optional[AST] = None
    conditions: List[AST] = field(default_factory=list)
    validation_mode: Optional[ValidationMode] = None
    input_mode: Optional[Union[HRInputMode, CHInputMode]] = None
    output: Optional[Union[HierarchyOutput, ValidationOutput]] = None
```

#### DPValidation

Covers the `check_datapoint` operator:

```python
@dataclass
class DPValidation(AST):
    """
    DPValidation: Datapoint ruleset validation (check_datapoint)

    dataset: The input dataset expression
    ruleset_name: Name of the datapoint ruleset (DPRuleset)
    components: Optional list of component IDs (from COMPONENTS clause)
    output: Output mode (invalid, all, all_measures)
    """
    dataset: AST
    ruleset_name: str
    components: List[AST] = field(default_factory=list)
    output: Optional[ValidationOutput] = None
```

## Files to Modify

| File | Changes |
|------|---------|
| **AST/__init__.py** | Add 5 enums + 2 new dataclasses (`HROperation`, `DPValidation`) |
| **AST/ASTConstructorModules/Expr.py** | Update `visitHierarchyFunctions` to return `HROperation`, update `visitValidateHRruleset` to return `HROperation`, update `visitValidateDPruleset` to return `DPValidation` |
| **AST/ASTString.py** | Add `visit_HROperation` and `visit_DPValidation` methods to convert back to VTL string |
| **Interpreter/__init__.py** | Add `visit_HROperation` and `visit_DPValidation` methods, extract relevant logic from the large `visit_ParamOp` method |

## Migration Approach

### AST Construction (Expr.py)

- Parse mode strings from grammar and convert to enums immediately
- Example: `"non_null"` → `ValidationMode.NON_NULL`
- Handle missing/default values at interpretation time (not AST construction) to keep the AST a faithful representation of the source

### Interpreter

- New `visit_HROperation` handles both `hierarchy` and `check_hierarchy` with a single branch on `node.op`
- New `visit_DPValidation` handles `check_datapoint`
- Apply default modes based on operator type:
  - `hierarchy`: defaults to `HRInputMode.RULE`, `HierarchyOutput.COMPUTED`
  - `check_hierarchy`: defaults to `CHInputMode.DATASET`, `ValidationOutput.INVALID`
  - `check_datapoint`: defaults to `ValidationOutput.INVALID`

### ASTString (pretty-printing)

- New visitor methods reconstruct VTL syntax from the typed nodes
- Use enum values directly (e.g., `node.validation_mode.value` → `"non_null"`)

### Backward Compatibility

- No external API changes - internal refactor only
- Existing tests should pass once visitor methods are updated
