from vtlengine.AST.Grammar.tokens import MOD, POWER

TO_SQL_TOKEN = {
    # Numeric operators
    MOD: "%",
    POWER: "^",
}