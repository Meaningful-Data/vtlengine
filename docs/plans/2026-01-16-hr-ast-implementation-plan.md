# HR/DP AST Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Introduce dedicated AST nodes (`HROperation`, `DPValidation`) and enums for `check_datapoint`, `check_hierarchy`, and `hierarchy` operators, replacing generic `ParamOp` usage.

**Architecture:** Two new AST node classes - `HROperation` for hierarchy/check_hierarchy (share hierarchical rulesets), `DPValidation` for check_datapoint. Five new enums for type-safe mode parameters. Visitors updated in Expr.py, ASTString.py, and Interpreter.

**Tech Stack:** Python dataclasses, Enum, existing VTL AST visitor pattern

---

## Task 1: Add Enums to AST/__init__.py

**Files:**
- Modify: `src/vtlengine/AST/__init__.py:1-15`

**Step 1: Add enum import**

Add `Enum` to the imports at the top of the file:

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union
```

**Step 2: Add the five enum definitions**

Add after the imports, before the `AST` class definition (around line 16):

```python
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

**Step 3: Run tests to verify no regressions**

Run: `cd /home/javier/Programacion/vtlengine && python -c "from vtlengine.AST import ValidationMode, HRInputMode, CHInputMode, ValidationOutput, HierarchyOutput; print('Enums imported successfully')"`

Expected: `Enums imported successfully`

**Step 4: Commit**

```bash
git add src/vtlengine/AST/__init__.py
git commit -m "feat(AST): add enums for HR/DP operation modes"
```

---

## Task 2: Add HROperation AST Node

**Files:**
- Modify: `src/vtlengine/AST/__init__.py`

**Step 1: Add `field` to dataclass imports**

Verify the import line includes `field`:

```python
from dataclasses import dataclass, field
```

**Step 2: Add HROperation dataclass**

Add after `HRuleset` class (around line 624):

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

    __eq__ = AST.ast_equality
```

**Step 3: Run test to verify import**

Run: `cd /home/javier/Programacion/vtlengine && python -c "from vtlengine.AST import HROperation; print('HROperation imported successfully')"`

Expected: `HROperation imported successfully`

**Step 4: Commit**

```bash
git add src/vtlengine/AST/__init__.py
git commit -m "feat(AST): add HROperation node for hierarchy/check_hierarchy"
```

---

## Task 3: Add DPValidation AST Node

**Files:**
- Modify: `src/vtlengine/AST/__init__.py`

**Step 1: Add DPValidation dataclass**

Add after `HROperation` class:

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
    components: List[str] = field(default_factory=list)
    output: Optional[ValidationOutput] = None

    __eq__ = AST.ast_equality
```

**Step 2: Run test to verify import**

Run: `cd /home/javier/Programacion/vtlengine && python -c "from vtlengine.AST import DPValidation; print('DPValidation imported successfully')"`

Expected: `DPValidation imported successfully`

**Step 3: Commit**

```bash
git add src/vtlengine/AST/__init__.py
git commit -m "feat(AST): add DPValidation node for check_datapoint"
```

---

## Task 4: Update Expr.py - Add imports and helper functions

**Files:**
- Modify: `src/vtlengine/AST/ASTConstructorModules/Expr.py:7-32`

**Step 1: Update imports**

Add the new AST nodes and enums to the import block:

```python
from vtlengine.AST import (
    ID,
    Aggregation,
    Analytic,
    Assignment,
    BinOp,
    Case,
    CaseObj,
    CHInputMode,
    Constant,
    DPValidation,
    EvalOp,
    HierarchyOutput,
    HRInputMode,
    HROperation,
    Identifier,
    If,
    JoinOp,
    MulOp,
    ParamConstant,
    ParamOp,
    ParFunction,
    RegularAggregation,
    RenameNode,
    TimeAggregation,
    UDOCall,
    UnaryOp,
    Validation,
    ValidationMode,
    ValidationOutput,
    VarID,
    Windowing,
)
```

**Step 2: Run test**

Run: `cd /home/javier/Programacion/vtlengine && python -c "from vtlengine.AST.ASTConstructorModules.Expr import Expr; print('Expr imported successfully')"`

Expected: `Expr imported successfully`

**Step 3: Commit**

```bash
git add src/vtlengine/AST/ASTConstructorModules/Expr.py
git commit -m "refactor(Expr): add imports for new HR/DP AST nodes and enums"
```

---

## Task 5: Update visitHierarchyFunctions to return HROperation

**Files:**
- Modify: `src/vtlengine/AST/ASTConstructorModules/Expr.py:1075-1145`

**Step 1: Rewrite visitHierarchyFunctions method**

Replace the entire method:

```python
def visitHierarchyFunctions(self, ctx: Parser.HierarchyFunctionsContext):
    """
    HIERARCHY LPAREN op=expr COMMA hrName=IDENTIFIER (conditionClause)? (RULE ruleComponent=componentID)? (validationMode)? (inputModeHierarchy)? outputModeHierarchy? RPAREN
    """  # noqa E501
    ctx_list = list(ctx.getChildren())
    c = ctx_list[0]

    op = c.getSymbol().text
    dataset_node = self.visitExpr(ctx_list[2])
    ruleset_name = ctx_list[4].getSymbol().text

    conditions = []
    validation_mode: Optional[ValidationMode] = None
    input_mode: Optional[HRInputMode] = None
    output: Optional[HierarchyOutput] = None
    rule_comp = None

    for c in ctx_list:
        if isinstance(c, Parser.ConditionClauseContext):
            conditions.append(Terminals().visitConditionClause(c))
        elif isinstance(c, Parser.ComponentIDContext):
            rule_comp = Terminals().visitComponentID(c)
        elif isinstance(c, Parser.ValidationModeContext):
            mode_str = Terminals().visitValidationMode(c)
            validation_mode = ValidationMode(mode_str)
        elif isinstance(c, Parser.InputModeHierarchyContext):
            input_str = Terminals().visitInputModeHierarchy(c)
            if input_str == DATASET_PRIORITY:
                raise NotImplementedError("Dataset Priority input mode on HR is not implemented")
            input_mode = HRInputMode(input_str)
        elif isinstance(c, Parser.OutputModeHierarchyContext):
            output_str = Terminals().visitOutputModeHierarchy(c)
            output = HierarchyOutput(output_str)

    if len(conditions) != 0:
        # AST_ASTCONSTRUCTOR.22
        conditions = conditions[0]
    else:
        conditions = []

    if not rule_comp and ruleset_name in de_ruleset_elements:
        if isinstance(de_ruleset_elements[ruleset_name], list):
            rule_element = de_ruleset_elements[ruleset_name][-1]
        else:
            rule_element = de_ruleset_elements[ruleset_name]
        if rule_element.kind == "DatasetID":
            check_hierarchy_rule = rule_element.value
            rule_comp = Identifier(
                value=check_hierarchy_rule, kind="ComponentID", **extract_token_info(ctx)
            )
        else:  # ValuedomainID
            raise SemanticError("1-1-10-4", op=op)

    return HROperation(
        op=op,
        dataset=dataset_node,
        ruleset_name=ruleset_name,
        rule_component=rule_comp,
        conditions=conditions if isinstance(conditions, list) else [conditions],
        validation_mode=validation_mode,
        input_mode=input_mode,
        output=output,
        **extract_token_info(ctx),
    )
```

**Step 2: Run a quick syntax check**

Run: `cd /home/javier/Programacion/vtlengine && python -c "from vtlengine.AST.ASTConstructorModules.Expr import Expr; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add src/vtlengine/AST/ASTConstructorModules/Expr.py
git commit -m "refactor(Expr): visitHierarchyFunctions returns HROperation"
```

---

## Task 6: Update visitValidateHRruleset to return HROperation

**Files:**
- Modify: `src/vtlengine/AST/ASTConstructorModules/Expr.py:1201-1280`

**Step 1: Rewrite visitValidateHRruleset method**

Replace the entire method:

```python
def visitValidateHRruleset(self, ctx: Parser.ValidateHRrulesetContext):
    """
    CHECK_HIERARCHY LPAREN op=expr COMMA hrName=IDENTIFIER conditionClause? (RULE componentID)? validationMode? inputMode? validationOutput? RPAREN     # validateHRruleset
    """  # noqa E501

    ctx_list = list(ctx.getChildren())
    c = ctx_list[0]

    op = c.getSymbol().text

    dataset_node = self.visitExpr(ctx_list[2])
    ruleset_name = ctx_list[4].getSymbol().text

    conditions = []
    validation_mode: Optional[ValidationMode] = None
    input_mode: Optional[CHInputMode] = None
    output: Optional[ValidationOutput] = None
    rule_comp = None

    for c in ctx_list:
        if isinstance(c, Parser.ConditionClauseContext):
            conditions.append(Terminals().visitConditionClause(c))
        elif isinstance(c, Parser.ComponentIDContext):
            rule_comp = Terminals().visitComponentID(c)
        elif isinstance(c, Parser.ValidationModeContext):
            mode_str = Terminals().visitValidationMode(c)
            validation_mode = ValidationMode(mode_str)
        elif isinstance(c, Parser.InputModeContext):
            input_str = Terminals().visitInputMode(c)
            if input_str == DATASET_PRIORITY:
                raise NotImplementedError("Dataset Priority input mode on HR is not implemented")
            input_mode = CHInputMode(input_str)
        elif isinstance(c, Parser.ValidationOutputContext):
            output_str = Terminals().visitValidationOutput(c)
            output = ValidationOutput(output_str)

    if len(conditions) != 0:
        # AST_ASTCONSTRUCTOR.22
        conditions = conditions[0]
    else:
        conditions = []

    if not rule_comp:
        if ruleset_name in de_ruleset_elements:
            if isinstance(de_ruleset_elements[ruleset_name], list):
                rule_element = de_ruleset_elements[ruleset_name][-1]
            else:
                rule_element = de_ruleset_elements[ruleset_name]

            if rule_element.kind == "DatasetID":
                check_hierarchy_rule = rule_element.value
                rule_comp = Identifier(
                    value=check_hierarchy_rule,
                    kind="ComponentID",
                    **extract_token_info(ctx),
                )
            else:  # ValuedomainID
                raise SemanticError("1-1-10-4", op=op)

    return HROperation(
        op=op,
        dataset=dataset_node,
        ruleset_name=ruleset_name,
        rule_component=rule_comp,
        conditions=conditions if isinstance(conditions, list) else [conditions],
        validation_mode=validation_mode,
        input_mode=input_mode,
        output=output,
        **extract_token_info(ctx),
    )
```

**Step 2: Run a quick syntax check**

Run: `cd /home/javier/Programacion/vtlengine && python -c "from vtlengine.AST.ASTConstructorModules.Expr import Expr; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add src/vtlengine/AST/ASTConstructorModules/Expr.py
git commit -m "refactor(Expr): visitValidateHRruleset returns HROperation"
```

---

## Task 7: Update visitValidateDPruleset to return DPValidation

**Files:**
- Modify: `src/vtlengine/AST/ASTConstructorModules/Expr.py:1161-1198`

**Step 1: Rewrite visitValidateDPruleset method**

Replace the entire method:

```python
def visitValidateDPruleset(self, ctx: Parser.ValidateDPrulesetContext):
    """
    validationDatapoint: CHECK_DATAPOINT '(' expr ',' IDENTIFIER (COMPONENTS componentID (',' componentID)*)? (INVALID|ALL_MEASURES|ALL)? ')' ;
    """  # noqa E501
    ctx_list = list(ctx.getChildren())

    dataset_node = self.visitExpr(ctx_list[2])
    ruleset_name = ctx_list[4].getSymbol().text

    components = [
        Terminals().visitComponentID(comp)
        for comp in ctx_list
        if isinstance(comp, Parser.ComponentIDContext)
    ]
    component_names = []
    for x in components:
        if isinstance(x, BinOp):
            component_names.append(x.right.value)
        else:
            component_names.append(x.value)

    # Default value for output is invalid (None means use default at interpretation)
    output: Optional[ValidationOutput] = None

    if isinstance(ctx_list[-2], Parser.ValidationOutputContext):
        output_str = Terminals().visitValidationOutput(ctx_list[-2])
        output = ValidationOutput(output_str)

    return DPValidation(
        dataset=dataset_node,
        ruleset_name=ruleset_name,
        components=component_names,
        output=output,
        **extract_token_info(ctx),
    )
```

**Step 2: Run a quick syntax check**

Run: `cd /home/javier/Programacion/vtlengine && python -c "from vtlengine.AST.ASTConstructorModules.Expr import Expr; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add src/vtlengine/AST/ASTConstructorModules/Expr.py
git commit -m "refactor(Expr): visitValidateDPruleset returns DPValidation"
```

---

## Task 8: Update ASTString.py - Add imports

**Files:**
- Modify: `src/vtlengine/AST/ASTString.py:5-11`

**Step 1: Update imports**

The file already imports `AST` module. Add to ensure the new types are accessible. Check the existing import pattern - the file uses `import vtlengine.AST as AST` and then references types like `AST.MulOp`. The enums and new nodes will be accessible via this pattern.

No changes needed - the existing `import vtlengine.AST as AST` already provides access to all AST members.

**Step 2: Commit (skip if no changes)**

No commit needed.

---

## Task 9: Add visit_HROperation to ASTString.py

**Files:**
- Modify: `src/vtlengine/AST/ASTString.py` (add after visit_ParamOp method, around line 420)

**Step 1: Add visit_HROperation method**

Add the new visitor method:

```python
def visit_HROperation(self, node: AST.HROperation) -> str:
    operand = self.visit(node.dataset)
    rule_name = node.ruleset_name

    # Handle case with no rule component
    if node.rule_component is None:
        if self.pretty:
            return f"{node.op}({nl}{tab * 2}{operand},{nl}{tab * 2}{rule_name}{nl})"
        else:
            return f"{node.op}({operand}, {rule_name})"

    component_name = self.visit(node.rule_component)

    # Build condition string
    condition_str = ""
    if node.conditions:
        condition_str += "condition "
        conditions = [self.visit(condition) for condition in node.conditions]
        condition_str += ", ".join(conditions)
        condition_str += f"{nl}{tab * 2}" if self.pretty else " "

    # Determine defaults based on operator
    default_input = "dataset" if node.op == CHECK_HIERARCHY else "rule"
    default_output = "invalid" if node.op == CHECK_HIERARCHY else "computed"

    # Build mode strings (only include if different from default)
    param_mode = ""
    if node.validation_mode is not None and node.validation_mode.value != "non_null":
        param_mode = f" {node.validation_mode.value}"

    param_input = ""
    if node.input_mode is not None and node.input_mode.value != default_input:
        param_input = f" {node.input_mode.value}"

    param_output = ""
    if node.output is not None and node.output.value != default_output:
        param_output = f" {node.output.value}"

    if self.pretty:
        return (
            f"{node.op}({nl}{tab * 2}{operand},"
            f"{nl}{tab * 2}{rule_name}{nl}{tab * 2}{condition_str}rule "
            f"{component_name}"
            f"{param_mode}{param_input}{param_output})"
        )
    else:
        return (
            f"{node.op}({operand}, {rule_name} {condition_str}rule {component_name}"
            f"{param_mode}{param_input}{param_output})"
        )
```

**Step 2: Run a quick syntax check**

Run: `cd /home/javier/Programacion/vtlengine && python -c "from vtlengine.AST.ASTString import ASTString; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add src/vtlengine/AST/ASTString.py
git commit -m "feat(ASTString): add visit_HROperation for hierarchy/check_hierarchy"
```

---

## Task 10: Add visit_DPValidation to ASTString.py

**Files:**
- Modify: `src/vtlengine/AST/ASTString.py` (add after visit_HROperation)

**Step 1: Add visit_DPValidation method**

Add the new visitor method:

```python
def visit_DPValidation(self, node: AST.DPValidation) -> str:
    operand = self.visit(node.dataset)
    rule_name = node.ruleset_name

    # Build components string if present
    components_str = ""
    if node.components:
        components_str = " components " + ", ".join(node.components)

    # Output mode (only include if not default "invalid")
    output_str = ""
    if node.output is not None and node.output.value != "invalid":
        output_str = f" {node.output.value}"

    if self.pretty:
        return f"{CHECK_DATAPOINT}({nl}{tab}{operand},{nl}{tab}{rule_name}{components_str}{output_str}{nl})"
    else:
        return f"{CHECK_DATAPOINT}({operand}, {rule_name}{components_str}{output_str})"
```

**Step 2: Run a quick syntax check**

Run: `cd /home/javier/Programacion/vtlengine && python -c "from vtlengine.AST.ASTString import ASTString; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add src/vtlengine/AST/ASTString.py
git commit -m "feat(ASTString): add visit_DPValidation for check_datapoint"
```

---

## Task 11: Update Interpreter imports

**Files:**
- Modify: `src/vtlengine/Interpreter/__init__.py:9`

**Step 1: Verify AST import**

The file already uses `import vtlengine.AST as AST`, so the new nodes are already accessible. No changes needed.

**Step 2: Add token imports if missing**

Verify the tokens file imports include what we need. Check line 15-48 - we may need to add mode-related tokens. Actually, the interpreter uses string values directly, not tokens for the modes. No changes needed.

**Step 3: Commit (skip if no changes)**

No commit needed.

---

## Task 12: Add visit_HROperation to Interpreter

**Files:**
- Modify: `src/vtlengine/Interpreter/__init__.py` (add new method after visit_ParamOp)

**Step 1: Add visit_HROperation method**

Add this new method to the InterpreterAnalyzer class:

```python
def visit_HROperation(self, node: AST.HROperation) -> Dataset:
    """Handle hierarchy and check_hierarchy operators."""
    from vtlengine.AST.Grammar.tokens import CHECK_HIERARCHY, HIERARCHY

    # Visit dataset and get component if present
    dataset = self.visit(node.dataset)
    component: Optional[str] = self.visit(node.rule_component) if node.rule_component else None
    hr_name = node.ruleset_name
    cond_components = [self.visit(c) for c in node.conditions] if node.conditions else []

    # Get mode values with defaults
    if node.op == HIERARCHY:
        mode = node.validation_mode.value if node.validation_mode else "non_null"
        input_ = node.input_mode.value if node.input_mode else "rule"
        output = node.output.value if node.output else "computed"
    else:  # CHECK_HIERARCHY
        mode = node.validation_mode.value if node.validation_mode else "non_null"
        input_ = node.input_mode.value if node.input_mode else "dataset"
        output = node.output.value if node.output else "invalid"

    # Validate hierarchical ruleset exists
    if self.hrs is None:
        raise SemanticError("1-2-6", node_type="Hierarchical Rulesets", node_value="")
    if hr_name not in self.hrs:
        raise SemanticError("1-2-6", node_type="Hierarchical Ruleset", node_value=hr_name)

    if not isinstance(dataset, Dataset):
        raise SemanticError("1-1-1-20", op=node.op)

    hr_info = self.hrs[hr_name]

    if hr_info is not None:
        if len(cond_components) != len(hr_info["condition"]):
            raise SemanticError("1-1-10-2", op=node.op)

        if hr_info["node"].signature_type == "variable" and hr_info["signature"] != component:
            raise SemanticError(
                "1-1-10-3",
                op=node.op,
                found=component,
                expected=hr_info["signature"],
            )
        elif hr_info["node"].signature_type == "valuedomain" and component is None:
            raise SemanticError("1-1-10-4", op=node.op)
        elif component is None:
            raise NotImplementedError(
                "Hierarchical Ruleset handling without component "
                "and signature type variable is not implemented yet."
            )

        cond_info = {}
        for i, cond_comp in enumerate(hr_info["condition"]):
            if hr_info["node"].signature_type == "variable" and cond_components[i] != cond_comp:
                raise SemanticError(
                    "1-1-10-6",
                    op=node.op,
                    expected=cond_comp,
                    found=cond_components[i],
                )
            cond_info[cond_comp] = cond_components[i]

        if node.op == HIERARCHY:
            aux = []
            for rule in hr_info["rules"]:
                if rule.rule.op == EQ or rule.rule.op == WHEN and rule.rule.right.op == EQ:
                    aux.append(rule)
            if len(aux) == 0:
                raise SemanticError("1-1-10-5")
            hr_info["rules"] = aux

            hierarchy_ast = AST.HRuleset(
                name=hr_name,
                signature_type=hr_info["node"].signature_type,
                element=hr_info["node"].element,
                rules=aux,
                line_start=node.line_start,
                line_stop=node.line_stop,
                column_start=node.column_start,
                column_stop=node.column_stop,
            )
            HRDAGAnalyzer().visit(hierarchy_ast)

        Check_Hierarchy.validate_hr_dataset(dataset, component)

        # Set up interpreter state for rule processing
        self.ruleset_dataset = dataset
        self.ruleset_signature = {**{"RULE_COMPONENT": component}, **cond_info}
        self.ruleset_mode = mode
        self.hr_input = input_
        rule_output_values = {}

        if node.op == HIERARCHY:
            self.is_from_hr_agg = True
            self.hr_agg_rules_computed = {}
            for rule in hr_info["rules"]:
                self.visit(rule)
            self.is_from_hr_agg = False
        else:
            self.is_from_hr_val = True
            for rule in hr_info["rules"]:
                rule_output_values[rule.name] = {
                    "errorcode": rule.erCode,
                    "errorlevel": rule.erLevel,
                    "output": self.visit(rule),
                }
            self.is_from_hr_val = False

        # Clean up interpreter state
        self.ruleset_signature = None
        self.ruleset_dataset = None
        self.ruleset_mode = None
        self.hr_input = None

        # Final evaluation
        if node.op == CHECK_HIERARCHY:
            result = Check_Hierarchy.analyze(
                dataset_element=dataset,
                rule_info=rule_output_values,
                output=output,
            )
            del rule_output_values
        else:
            result = Hierarchy.analyze(dataset, self.hr_agg_rules_computed, output)
            self.hr_agg_rules_computed = None
        return result

    raise SemanticError("1-3-5", op_type="HROperation", node_op=node.op)
```

**Step 2: Run a quick syntax check**

Run: `cd /home/javier/Programacion/vtlengine && python -c "from vtlengine.Interpreter import InterpreterAnalyzer; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add src/vtlengine/Interpreter/__init__.py
git commit -m "feat(Interpreter): add visit_HROperation for hierarchy/check_hierarchy"
```

---

## Task 13: Add visit_DPValidation to Interpreter

**Files:**
- Modify: `src/vtlengine/Interpreter/__init__.py` (add new method after visit_HROperation)

**Step 1: Add visit_DPValidation method**

Add this new method:

```python
def visit_DPValidation(self, node: AST.DPValidation) -> Dataset:
    """Handle check_datapoint operator."""
    if self.dprs is None:
        raise SemanticError("1-2-6", node_type="Datapoint Rulesets", node_value="")

    dpr_name = node.ruleset_name
    if dpr_name not in self.dprs:
        raise SemanticError("1-2-6", node_type="Datapoint Ruleset", node_value=dpr_name)
    dpr_info = self.dprs[dpr_name]

    # Extract dataset
    dataset_element = self.visit(node.dataset)
    if not isinstance(dataset_element, Dataset):
        raise SemanticError("1-1-1-20", op=CHECK_DATAPOINT)

    # Check component list validity
    if node.components:
        for comp_name in node.components:
            if comp_name not in dataset_element.components:
                raise SemanticError(
                    "1-1-1-10",
                    comp_name=comp_name,
                    dataset_name=dataset_element.name,
                )
            if dpr_info is not None and dpr_info["signature_type"] == "variable":
                for i, comp_name in enumerate(node.components):
                    if comp_name != dpr_info["params"][i]:
                        raise SemanticError(
                            "1-1-10-3",
                            op=CHECK_DATAPOINT,
                            expected=dpr_info["params"][i],
                            found=comp_name,
                        )

    # Get output mode with default
    output = node.output.value if node.output else "invalid"

    if dpr_info is None:
        dpr_info = {}

    rule_output_values = {}
    self.ruleset_dataset = dataset_element
    self.ruleset_signature = dpr_info.get("signature")
    self.ruleset_mode = output

    # Gather rule data
    if dpr_info:
        for rule in dpr_info["rules"]:
            rule_output_values[rule.name] = {
                "errorcode": rule.erCode,
                "errorlevel": rule.erLevel,
                "output": self.visit(rule),
            }

    self.ruleset_mode = None
    self.ruleset_signature = None
    self.ruleset_dataset = None

    # Final evaluation
    return Check_Datapoint.analyze(
        dataset_element=dataset_element,
        rule_info=rule_output_values,
        output=output,
    )
```

**Step 2: Run a quick syntax check**

Run: `cd /home/javier/Programacion/vtlengine && python -c "from vtlengine.Interpreter import InterpreterAnalyzer; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add src/vtlengine/Interpreter/__init__.py
git commit -m "feat(Interpreter): add visit_DPValidation for check_datapoint"
```

---

## Task 14: Remove old cases from visit_ParamOp

**Files:**
- Modify: `src/vtlengine/Interpreter/__init__.py:1272-1462`

**Step 1: Remove CHECK_DATAPOINT case**

Remove lines 1272-1329 (the `elif node.op == CHECK_DATAPOINT:` block).

**Step 2: Remove CHECK_HIERARCHY and HIERARCHY case**

Remove lines 1330-1462 (the `elif node.op in (CHECK_HIERARCHY, HIERARCHY):` block).

**Step 3: Run a quick syntax check**

Run: `cd /home/javier/Programacion/vtlengine && python -c "from vtlengine.Interpreter import InterpreterAnalyzer; print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add src/vtlengine/Interpreter/__init__.py
git commit -m "refactor(Interpreter): remove HR/DP cases from visit_ParamOp"
```

---

## Task 15: Remove old cases from visit_ParamOp in ASTString.py

**Files:**
- Modify: `src/vtlengine/AST/ASTString.py:322-390`

**Step 1: Remove CHECK_HIERARCHY and HIERARCHY case from visit_ParamOp**

Remove lines 332-379 (the `elif node.op in (CHECK_HIERARCHY, HIERARCHY):` block).

**Step 2: Remove CHECK_DATAPOINT case from visit_ParamOp**

Remove lines 381-390 (the `elif node.op == CHECK_DATAPOINT:` block).

**Step 3: Run a quick syntax check**

Run: `cd /home/javier/Programacion/vtlengine && python -c "from vtlengine.AST.ASTString import ASTString; print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add src/vtlengine/AST/ASTString.py
git commit -m "refactor(ASTString): remove HR/DP cases from visit_ParamOp"
```

---

## Task 16: Run full test suite

**Files:**
- Test: `tests/`

**Step 1: Run hierarchical tests**

Run: `cd /home/javier/Programacion/vtlengine && python -m pytest tests/Hierarchical/ -v`

Expected: All tests pass

**Step 2: Run datapoint tests**

Run: `cd /home/javier/Programacion/vtlengine && python -m pytest tests/DatapointRulesets/ -v`

Expected: All tests pass

**Step 3: Run full test suite**

Run: `cd /home/javier/Programacion/vtlengine && python -m pytest tests/ -x --tb=short`

Expected: All tests pass

**Step 4: Commit if any fixes were needed**

If fixes were made, commit them with appropriate messages.

---

## Task 17: Final cleanup and documentation

**Files:**
- Review all modified files

**Step 1: Verify imports are clean**

Check for any unused imports in modified files.

**Step 2: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup for HR/DP AST refactoring"
```

---

## Summary of Changes

| File | Changes |
|------|---------|
| `AST/__init__.py` | +5 enums, +2 dataclasses (`HROperation`, `DPValidation`) |
| `AST/ASTConstructorModules/Expr.py` | Updated imports, rewrote 3 visitor methods to return new AST types |
| `AST/ASTString.py` | Added 2 new visitor methods, removed 2 cases from `visit_ParamOp` |
| `Interpreter/__init__.py` | Added 2 new visitor methods, removed 2 cases from `visit_ParamOp` |
