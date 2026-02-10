"""Tests for SQLBuilder class."""

import pytest

from vtlengine.duckdb_transpiler.Transpiler.sql_builder import (
    SQLBuilder,
    build_binary_expr,
    build_column_expr,
    build_function_expr,
    quote_identifier,
    quote_identifiers,
)

# =============================================================================
# SQLBuilder Tests
# =============================================================================


class TestSQLBuilderSelect:
    """Tests for SQLBuilder SELECT functionality."""

    def test_simple_select(self):
        """Test basic SELECT query."""
        sql = SQLBuilder().select('"Id_1"', '"Me_1"').from_table('"DS_1"').build()
        assert sql == 'SELECT "Id_1", "Me_1" FROM "DS_1"'

    def test_select_all(self):
        """Test SELECT * query."""
        sql = SQLBuilder().select_all().from_table('"DS_1"').build()
        assert sql == 'SELECT * FROM "DS_1"'

    def test_select_with_alias(self):
        """Test SELECT with table alias."""
        sql = SQLBuilder().select('"Id_1"').from_table('"DS_1"', "t").build()
        assert sql == 'SELECT "Id_1" FROM "DS_1" AS t'

    def test_select_distinct(self):
        """Test SELECT DISTINCT."""
        sql = SQLBuilder().distinct().select('"Id_1"').from_table('"DS_1"').build()
        assert sql == 'SELECT DISTINCT "Id_1" FROM "DS_1"'

    def test_select_distinct_on(self):
        """Test SELECT DISTINCT ON (DuckDB)."""
        sql = SQLBuilder().distinct_on('"Id_1"', '"Id_2"').select_all().from_table('"DS_1"').build()
        assert sql == 'SELECT DISTINCT ON ("Id_1", "Id_2") * FROM "DS_1"'


class TestSQLBuilderFrom:
    """Tests for SQLBuilder FROM functionality."""

    def test_from_table(self):
        """Test FROM with simple table."""
        sql = SQLBuilder().select_all().from_table('"DS_1"').build()
        assert sql == 'SELECT * FROM "DS_1"'

    def test_from_table_with_alias(self):
        """Test FROM with table alias."""
        sql = SQLBuilder().select_all().from_table('"DS_1"', "t").build()
        assert sql == 'SELECT * FROM "DS_1" AS t'

    def test_from_subquery(self):
        """Test FROM with subquery."""
        sql = SQLBuilder().select('"Id_1"').from_subquery('SELECT * FROM "DS_1"', "t").build()
        assert sql == 'SELECT "Id_1" FROM (SELECT * FROM "DS_1") AS t'


class TestSQLBuilderWhere:
    """Tests for SQLBuilder WHERE functionality."""

    def test_where_single(self):
        """Test single WHERE condition."""
        sql = SQLBuilder().select_all().from_table('"DS_1"').where('"Me_1" > 10').build()
        assert sql == 'SELECT * FROM "DS_1" WHERE "Me_1" > 10'

    def test_where_multiple(self):
        """Test multiple WHERE conditions (AND)."""
        sql = (
            SQLBuilder()
            .select_all()
            .from_table('"DS_1"')
            .where('"Me_1" > 10')
            .where('"Me_2" < 100')
            .build()
        )
        assert sql == 'SELECT * FROM "DS_1" WHERE "Me_1" > 10 AND "Me_2" < 100'

    def test_where_all(self):
        """Test where_all with list of conditions."""
        sql = (
            SQLBuilder()
            .select_all()
            .from_table('"DS_1"')
            .where_all(['"Me_1" > 10', '"Me_2" < 100'])
            .build()
        )
        assert sql == 'SELECT * FROM "DS_1" WHERE "Me_1" > 10 AND "Me_2" < 100'


class TestSQLBuilderJoins:
    """Tests for SQLBuilder JOIN functionality."""

    @pytest.mark.parametrize(
        "join_method,expected_join_type",
        [
            ("inner_join", "INNER JOIN"),
            ("left_join", "LEFT JOIN"),
        ],
    )
    def test_join_with_on_clause(self, join_method, expected_join_type):
        """Test JOINs with ON clause."""
        builder = SQLBuilder().select_all().from_table('"DS_1"', "a")
        join_func = getattr(builder, join_method)
        sql = join_func('"DS_2"', "b", 'a."Id_1" = b."Id_1"').build()
        expected = (
            f'SELECT * FROM "DS_1" AS a {expected_join_type} "DS_2" AS b ON a."Id_1" = b."Id_1"'
        )
        assert sql == expected

    def test_inner_join_using(self):
        """Test INNER JOIN with USING clause."""
        sql = (
            SQLBuilder()
            .select_all()
            .from_table('"DS_1"', "a")
            .inner_join('"DS_2"', "b", using=["Id_1", "Id_2"])
            .build()
        )
        assert sql == 'SELECT * FROM "DS_1" AS a INNER JOIN "DS_2" AS b USING ("Id_1", "Id_2")'

    def test_left_join_using(self):
        """Test LEFT JOIN with USING clause."""
        sql = (
            SQLBuilder()
            .select_all()
            .from_table('"DS_1"', "a")
            .left_join('"DS_2"', "b", using=["Id_1"])
            .build()
        )
        assert sql == 'SELECT * FROM "DS_1" AS a LEFT JOIN "DS_2" AS b USING ("Id_1")'

    def test_cross_join(self):
        """Test CROSS JOIN."""
        sql = SQLBuilder().select_all().from_table('"DS_1"', "a").cross_join('"DS_2"', "b").build()
        assert sql == 'SELECT * FROM "DS_1" AS a CROSS JOIN "DS_2" AS b'


class TestSQLBuilderGroupBy:
    """Tests for SQLBuilder GROUP BY and HAVING functionality."""

    def test_group_by(self):
        """Test GROUP BY clause."""
        sql = (
            SQLBuilder()
            .select('"Id_1"', 'SUM("Me_1") AS "total"')
            .from_table('"DS_1"')
            .group_by('"Id_1"')
            .build()
        )
        assert sql == 'SELECT "Id_1", SUM("Me_1") AS "total" FROM "DS_1" GROUP BY "Id_1"'

    def test_having(self):
        """Test HAVING clause."""
        sql = (
            SQLBuilder()
            .select('"Id_1"', 'SUM("Me_1") AS "total"')
            .from_table('"DS_1"')
            .group_by('"Id_1"')
            .having('SUM("Me_1") > 100')
            .build()
        )
        assert (
            sql
            == 'SELECT "Id_1", SUM("Me_1") AS "total" FROM "DS_1" GROUP BY "Id_1" HAVING SUM("Me_1") > 100'
        )


class TestSQLBuilderOrderByLimit:
    """Tests for SQLBuilder ORDER BY and LIMIT functionality."""

    def test_order_by(self):
        """Test ORDER BY clause."""
        sql = (
            SQLBuilder()
            .select_all()
            .from_table('"DS_1"')
            .order_by('"Id_1" ASC', '"Me_1" DESC')
            .build()
        )
        assert sql == 'SELECT * FROM "DS_1" ORDER BY "Id_1" ASC, "Me_1" DESC'

    @pytest.mark.parametrize("limit_value", [1, 10, 100, 1000])
    def test_limit(self, limit_value):
        """Test LIMIT clause with various values."""
        sql = SQLBuilder().select_all().from_table('"DS_1"').limit(limit_value).build()
        assert sql == f'SELECT * FROM "DS_1" LIMIT {limit_value}'


class TestSQLBuilderComplex:
    """Tests for complex SQLBuilder queries."""

    def test_complex_query(self):
        """Test complex query with multiple clauses."""
        sql = (
            SQLBuilder()
            .select('"Id_1"', 'SUM("Me_1") AS "total"')
            .from_subquery('SELECT * FROM "DS_1" WHERE "active" = TRUE', "t")
            .where('"Id_1" IS NOT NULL')
            .group_by('"Id_1"')
            .having('SUM("Me_1") > 0')
            .order_by('"total" DESC')
            .limit(100)
            .build()
        )
        expected = (
            'SELECT "Id_1", SUM("Me_1") AS "total" '
            'FROM (SELECT * FROM "DS_1" WHERE "active" = TRUE) AS t '
            'WHERE "Id_1" IS NOT NULL '
            'GROUP BY "Id_1" '
            'HAVING SUM("Me_1") > 0 '
            'ORDER BY "total" DESC '
            "LIMIT 100"
        )
        assert sql == expected

    def test_reset(self):
        """Test builder reset."""
        builder = SQLBuilder()
        sql1 = builder.select('"Id_1"').from_table('"DS_1"').build()
        sql2 = builder.reset().select('"Id_2"').from_table('"DS_2"').build()

        assert sql1 == 'SELECT "Id_1" FROM "DS_1"'
        assert sql2 == 'SELECT "Id_2" FROM "DS_2"'

    def test_chaining(self):
        """Test method chaining returns self."""
        builder = SQLBuilder()
        result = builder.select('"col"').from_table('"table"').where("1=1")
        assert result is builder


# =============================================================================
# Helper Functions Tests
# =============================================================================


class TestQuoteIdentifier:
    """Tests for identifier quoting functions."""

    @pytest.mark.parametrize(
        "input_id,expected",
        [
            ("Id_1", '"Id_1"'),
            ("column name", '"column name"'),
            ("Me_1", '"Me_1"'),
            ("table", '"table"'),
        ],
    )
    def test_quote_identifier(self, input_id, expected):
        """Test single identifier quoting."""
        assert quote_identifier(input_id) == expected

    def test_quote_identifiers(self):
        """Test multiple identifier quoting."""
        result = quote_identifiers(["Id_1", "Id_2", "Me_1"])
        assert result == ['"Id_1"', '"Id_2"', '"Me_1"']

    def test_quote_identifiers_empty(self):
        """Test quoting empty list."""
        result = quote_identifiers([])
        assert result == []


class TestBuildColumnExpr:
    """Tests for column expression builder."""

    @pytest.mark.parametrize(
        "col,alias,table_alias,expected",
        [
            ("Me_1", None, None, '"Me_1"'),
            ("Me_1", "measure", None, '"Me_1" AS "measure"'),
            ("Me_1", None, "t", 't."Me_1"'),
            ("Me_1", "measure", "t", 't."Me_1" AS "measure"'),
        ],
    )
    def test_build_column_expr(self, col, alias, table_alias, expected):
        """Test column expression with various options."""
        result = build_column_expr(col, alias=alias, table_alias=table_alias)
        assert result == expected


class TestBuildFunctionExpr:
    """Tests for function expression builder."""

    @pytest.mark.parametrize(
        "func,col,alias,expected",
        [
            ("SUM", "Me_1", None, 'SUM("Me_1")'),
            ("SUM", "Me_1", "total", 'SUM("Me_1") AS "total"'),
            ("AVG", "Me_1", "average", 'AVG("Me_1") AS "average"'),
            ("COUNT", "Id_1", "cnt", 'COUNT("Id_1") AS "cnt"'),
        ],
    )
    def test_build_function_expr(self, func, col, alias, expected):
        """Test function expression with various options."""
        result = build_function_expr(func, col, alias=alias)
        assert result == expected


class TestBuildBinaryExpr:
    """Tests for binary expression builder."""

    @pytest.mark.parametrize(
        "left,op,right,alias,expected",
        [
            ('"Me_1"', "+", '"Me_2"', None, '("Me_1" + "Me_2")'),
            ('"Me_1"', "*", "2", "doubled", '("Me_1" * 2) AS "doubled"'),
            ('"a"', "-", '"b"', "diff", '("a" - "b") AS "diff"'),
            ('"x"', "/", '"y"', None, '("x" / "y")'),
        ],
    )
    def test_build_binary_expr(self, left, op, right, alias, expected):
        """Test binary expression with various options."""
        result = build_binary_expr(left, op, right, alias=alias)
        assert result == expected
