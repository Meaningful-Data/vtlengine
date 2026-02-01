"""Tests for SQLBuilder class."""


from vtlengine.duckdb_transpiler.Transpiler.sql_builder import (
    SQLBuilder,
    build_binary_expr,
    build_column_expr,
    build_function_expr,
    quote_identifier,
    quote_identifiers,
)


class TestSQLBuilder:
    """Tests for SQLBuilder class."""

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
        sql = (
            SQLBuilder()
            .distinct_on('"Id_1"', '"Id_2"')
            .select_all()
            .from_table('"DS_1"')
            .build()
        )
        assert sql == 'SELECT DISTINCT ON ("Id_1", "Id_2") * FROM "DS_1"'

    def test_from_subquery(self):
        """Test FROM with subquery."""
        sql = (
            SQLBuilder()
            .select('"Id_1"')
            .from_subquery('SELECT * FROM "DS_1"', "t")
            .build()
        )
        assert sql == 'SELECT "Id_1" FROM (SELECT * FROM "DS_1") AS t'

    def test_where_single(self):
        """Test single WHERE condition."""
        sql = (
            SQLBuilder()
            .select_all()
            .from_table('"DS_1"')
            .where('"Me_1" > 10')
            .build()
        )
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

    def test_inner_join_on(self):
        """Test INNER JOIN with ON clause."""
        sql = (
            SQLBuilder()
            .select_all()
            .from_table('"DS_1"', "a")
            .inner_join('"DS_2"', "b", 'a."Id_1" = b."Id_1"')
            .build()
        )
        assert (
            sql
            == 'SELECT * FROM "DS_1" AS a INNER JOIN "DS_2" AS b ON a."Id_1" = b."Id_1"'
        )

    def test_inner_join_using(self):
        """Test INNER JOIN with USING clause."""
        sql = (
            SQLBuilder()
            .select_all()
            .from_table('"DS_1"', "a")
            .inner_join('"DS_2"', "b", using=["Id_1", "Id_2"])
            .build()
        )
        assert (
            sql
            == 'SELECT * FROM "DS_1" AS a INNER JOIN "DS_2" AS b USING ("Id_1", "Id_2")'
        )

    def test_left_join(self):
        """Test LEFT JOIN."""
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
        sql = (
            SQLBuilder()
            .select_all()
            .from_table('"DS_1"', "a")
            .cross_join('"DS_2"', "b")
            .build()
        )
        assert sql == 'SELECT * FROM "DS_1" AS a CROSS JOIN "DS_2" AS b'

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

    def test_limit(self):
        """Test LIMIT clause."""
        sql = SQLBuilder().select_all().from_table('"DS_1"').limit(10).build()
        assert sql == 'SELECT * FROM "DS_1" LIMIT 10'

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


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_quote_identifier(self):
        """Test single identifier quoting."""
        assert quote_identifier("Id_1") == '"Id_1"'
        assert quote_identifier("column name") == '"column name"'

    def test_quote_identifiers(self):
        """Test multiple identifier quoting."""
        result = quote_identifiers(["Id_1", "Id_2", "Me_1"])
        assert result == ['"Id_1"', '"Id_2"', '"Me_1"']

    def test_build_column_expr_simple(self):
        """Test simple column expression."""
        assert build_column_expr("Me_1") == '"Me_1"'

    def test_build_column_expr_with_alias(self):
        """Test column expression with alias."""
        assert build_column_expr("Me_1", alias="measure") == '"Me_1" AS "measure"'

    def test_build_column_expr_with_table_alias(self):
        """Test column expression with table alias."""
        assert build_column_expr("Me_1", table_alias="t") == 't."Me_1"'

    def test_build_column_expr_full(self):
        """Test column expression with table and column alias."""
        result = build_column_expr("Me_1", alias="measure", table_alias="t")
        assert result == 't."Me_1" AS "measure"'

    def test_build_function_expr_simple(self):
        """Test simple function expression."""
        assert build_function_expr("SUM", "Me_1") == 'SUM("Me_1")'

    def test_build_function_expr_with_alias(self):
        """Test function expression with alias."""
        result = build_function_expr("SUM", "Me_1", alias="total")
        assert result == 'SUM("Me_1") AS "total"'

    def test_build_binary_expr_simple(self):
        """Test simple binary expression."""
        result = build_binary_expr('"Me_1"', "+", '"Me_2"')
        assert result == '("Me_1" + "Me_2")'

    def test_build_binary_expr_with_alias(self):
        """Test binary expression with alias."""
        result = build_binary_expr('"Me_1"', "*", "2", alias="doubled")
        assert result == '("Me_1" * 2) AS "doubled"'
