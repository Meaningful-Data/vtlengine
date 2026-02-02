# VTL to SQL Query Mapping for agg1 Transformations

This document shows the VTL script and corresponding DuckDB SQL queries for operations involving `agg1`.

## Table of Contents
- [agg1 - Aggregation](#agg1---aggregation)
- [agg2 - Aggregation](#agg2---aggregation)
- [chk101 - Check (agg1 + agg2)](#chk101---check-agg1--agg2)
- [chk201 - Check (agg1 - agg2)](#chk201---check-agg1---agg2)
- [chk301 - Check (agg1 * agg2)](#chk301---check-agg1--agg2-1)
- [chk401 - Check (agg1 / agg2)](#chk401---check-agg1--agg2-2)

---

## agg1 - Aggregation

**Description:** Sum with filter on VOCESOTVOC range 5889000-5889099

### VTL Script

```vtl
agg1 <-
	 sum(
			PoC_Dataset
				[filter between(VOCESOTVOC,5889000,5889099)] 
			group by DATA_CONTABILE,ENTE_SEGN,DIVISA,DURATA
		);
```

### SQL Query

```sql
SELECT "DATA_CONTABILE", "ENTE_SEGN", "DIVISA", "DURATA", SUM("IMPORTO") AS "IMPORTO"
                        FROM (SELECT * FROM "PoC_Dataset" WHERE ("VOCESOTVOC" BETWEEN 5889000 AND 5889099)) AS t
                        GROUP BY "DATA_CONTABILE", "ENTE_SEGN", "DIVISA", "DURATA"
```

---

## agg2 - Aggregation

**Description:** Sum with filter on VOCESOTVOC range 5889100-5889199

### VTL Script

```vtl
agg2 <-
	 sum(
			PoC_Dataset
				[filter between(VOCESOTVOC,5889100,5889199)] 
			group by DATA_CONTABILE,ENTE_SEGN,DIVISA,DURATA
		);
```

### SQL Query

```sql
SELECT "DATA_CONTABILE", "ENTE_SEGN", "DIVISA", "DURATA", SUM("IMPORTO") AS "IMPORTO"
                        FROM (SELECT * FROM "PoC_Dataset" WHERE ("VOCESOTVOC" BETWEEN 5889100 AND 5889199)) AS t
                        GROUP BY "DATA_CONTABILE", "ENTE_SEGN", "DIVISA", "DURATA"
```

---

## chk101 - Check (agg1 + agg2)

**Description:** Validation that sum is less than 1000

### VTL Script

```vtl
chk101 <-
	 check(
		agg1 
		+
		agg2 
		<
		 1000 
		errorlevel 8
		imbalance agg1 + agg2 - 1000);
```

### SQL Query

```sql
SELECT t.*,
                       CASE WHEN t."bool_var" = FALSE OR t."bool_var" IS NULL
                            THEN 'NULL' ELSE NULL END AS errorcode,
                       CASE WHEN t."bool_var" = FALSE OR t."bool_var" IS NULL
                            THEN 8 ELSE NULL END AS errorlevel, imb."IMPORTO" AS imbalance
                FROM (SELECT "DATA_CONTABILE", "DIVISA", "DURATA", "ENTE_SEGN", ("IMPORTO" < 1000) AS "bool_var" FROM (
                    SELECT a."DATA_CONTABILE", a."DIVISA", a."DURATA", a."ENTE_SEGN", (a."IMPORTO" + b."IMPORTO") AS "IMPORTO"
                    FROM "agg1" AS a
                    INNER JOIN "agg2" AS b ON a."DATA_CONTABILE" = b."DATA_CONTABILE" AND a."DIVISA" = b."DIVISA" AND a."DURATA" = b."DURATA" AND a."ENTE_SEGN" = b."ENTE_SEGN"
                )) AS t
                
                            LEFT JOIN (SELECT "DATA_CONTABILE", "DIVISA", "DURATA", "ENTE_SEGN", ("IMPORTO" - 1000) AS "IMPORTO" FROM (
                    SELECT a."DATA_CONTABILE", a."DIVISA", a."DURATA", a."ENTE_SEGN", (a."IMPORTO" + b."IMPORTO") AS "IMPORTO"
                    FROM "agg1" AS a
                    INNER JOIN "agg2" AS b ON a."DATA_CONTABILE" = b."DATA_CONTABILE" AND a."DIVISA" = b."DIVISA" AND a."DURATA" = b."DURATA" AND a."ENTE_SEGN" = b."ENTE_SEGN"
                )) AS imb ON t."DATA_CONTABILE" = imb."DATA_CONTABILE" AND t."DIVISA" = imb."DIVISA" AND t."DURATA" = imb."DURATA" AND t."ENTE_SEGN" = imb."ENTE_SEGN"
```

**Note:** Uses direct table references `"agg1"` and `"agg2"` in JOINs instead of subquery wrappers.

---

## chk201 - Check (agg1 - agg2)

**Description:** Validation that difference is less than 1000

### VTL Script

```vtl
chk201 <-
	 check(
		agg1 
		-
		agg2 
		<
		 1000 
		errorlevel 8
		imbalance agg1 - agg2 - 1000);
```

### SQL Query

```sql
SELECT t.*,
                       CASE WHEN t."bool_var" = FALSE OR t."bool_var" IS NULL
                            THEN 'NULL' ELSE NULL END AS errorcode,
                       CASE WHEN t."bool_var" = FALSE OR t."bool_var" IS NULL
                            THEN 8 ELSE NULL END AS errorlevel, imb."IMPORTO" AS imbalance
                FROM (SELECT "DATA_CONTABILE", "DIVISA", "DURATA", "ENTE_SEGN", ("IMPORTO" < 1000) AS "bool_var" FROM (
                    SELECT a."DATA_CONTABILE", a."DIVISA", a."DURATA", a."ENTE_SEGN", (a."IMPORTO" - b."IMPORTO") AS "IMPORTO"
                    FROM "agg1" AS a
                    INNER JOIN "agg2" AS b ON a."DATA_CONTABILE" = b."DATA_CONTABILE" AND a."DIVISA" = b."DIVISA" AND a."DURATA" = b."DURATA" AND a."ENTE_SEGN" = b."ENTE_SEGN"
                )) AS t
                
                            LEFT JOIN (SELECT "DATA_CONTABILE", "DIVISA", "DURATA", "ENTE_SEGN", ("IMPORTO" - 1000) AS "IMPORTO" FROM (
                    SELECT a."DATA_CONTABILE", a."DIVISA", a."DURATA", a."ENTE_SEGN", (a."IMPORTO" - b."IMPORTO") AS "IMPORTO"
                    FROM "agg1" AS a
                    INNER JOIN "agg2" AS b ON a."DATA_CONTABILE" = b."DATA_CONTABILE" AND a."DIVISA" = b."DIVISA" AND a."DURATA" = b."DURATA" AND a."ENTE_SEGN" = b."ENTE_SEGN"
                )) AS imb ON t."DATA_CONTABILE" = imb."DATA_CONTABILE" AND t."DIVISA" = imb."DIVISA" AND t."DURATA" = imb."DURATA" AND t."ENTE_SEGN" = imb."ENTE_SEGN"
```

---

## chk301 - Check (agg1 * agg2)

**Description:** Validation that product is less than 1000

### VTL Script

```vtl
chk301 <-
	 check(
		agg1 * agg2 
		<
		 1000 
		errorlevel 8
		imbalance(agg1 * agg2) - 1000);
```

### SQL Query

```sql
SELECT t.*,
                       CASE WHEN t."bool_var" = FALSE OR t."bool_var" IS NULL
                            THEN 'NULL' ELSE NULL END AS errorcode,
                       CASE WHEN t."bool_var" = FALSE OR t."bool_var" IS NULL
                            THEN 8 ELSE NULL END AS errorlevel, imb."IMPORTO" AS imbalance
                FROM (SELECT "DATA_CONTABILE", "DIVISA", "DURATA", "ENTE_SEGN", ("IMPORTO" < 1000) AS "bool_var" FROM (
                    SELECT a."DATA_CONTABILE", a."DIVISA", a."DURATA", a."ENTE_SEGN", (a."IMPORTO" * b."IMPORTO") AS "IMPORTO"
                    FROM "agg1" AS a
                    INNER JOIN "agg2" AS b ON a."DATA_CONTABILE" = b."DATA_CONTABILE" AND a."DIVISA" = b."DIVISA" AND a."DURATA" = b."DURATA" AND a."ENTE_SEGN" = b."ENTE_SEGN"
                )) AS t
                
                            LEFT JOIN (SELECT "DATA_CONTABILE", "DIVISA", "DURATA", "ENTE_SEGN", ("IMPORTO" - 1000) AS "IMPORTO" FROM ((
                    SELECT a."DATA_CONTABILE", a."DIVISA", a."DURATA", a."ENTE_SEGN", (a."IMPORTO" * b."IMPORTO") AS "IMPORTO"
                    FROM "agg1" AS a
                    INNER JOIN "agg2" AS b ON a."DATA_CONTABILE" = b."DATA_CONTABILE" AND a."DIVISA" = b."DIVISA" AND a."DURATA" = b."DURATA" AND a."ENTE_SEGN" = b."ENTE_SEGN"
                ))) AS imb ON t."DATA_CONTABILE" = imb."DATA_CONTABILE" AND t."DIVISA" = imb."DIVISA" AND t."DURATA" = imb."DURATA" AND t."ENTE_SEGN" = imb."ENTE_SEGN"
```

---

## chk401 - Check (agg1 / agg2)

**Description:** Validation that quotient is less than 1000

### VTL Script

```vtl
chk401 <-
	 check(
		agg1 / agg2
			[filter IMPORTO <> 0] 
		<
		 1000 
		errorlevel 8
		imbalance(agg1 / agg2) - 1000);
```

### SQL Query

```sql
SELECT t.*,
                       CASE WHEN t."bool_var" = FALSE OR t."bool_var" IS NULL
                            THEN 'NULL' ELSE NULL END AS errorcode,
                       CASE WHEN t."bool_var" = FALSE OR t."bool_var" IS NULL
                            THEN 8 ELSE NULL END AS errorlevel, imb."IMPORTO" AS imbalance
                FROM (SELECT "DATA_CONTABILE", "DIVISA", "DURATA", "ENTE_SEGN", ("IMPORTO" < 1000) AS "bool_var" FROM (
                    SELECT a."DATA_CONTABILE", a."DIVISA", a."DURATA", a."ENTE_SEGN", (a."IMPORTO" / b."IMPORTO") AS "IMPORTO"
                    FROM "agg1" AS a
                    INNER JOIN (SELECT * FROM "agg2" WHERE ("IMPORTO" <> 0)) AS b ON a."DATA_CONTABILE" = b."DATA_CONTABILE" AND a."DIVISA" = b."DIVISA" AND a."DURATA" = b."DURATA" AND a."ENTE_SEGN" = b."ENTE_SEGN"
                )) AS t
                
                            LEFT JOIN (SELECT "DATA_CONTABILE", "DIVISA", "DURATA", "ENTE_SEGN", ("IMPORTO" - 1000) AS "IMPORTO" FROM ((
                    SELECT a."DATA_CONTABILE", a."DIVISA", a."DURATA", a."ENTE_SEGN", (a."IMPORTO" / b."IMPORTO") AS "IMPORTO"
                    FROM "agg1" AS a
                    INNER JOIN "agg2" AS b ON a."DATA_CONTABILE" = b."DATA_CONTABILE" AND a."DIVISA" = b."DIVISA" AND a."DURATA" = b."DURATA" AND a."ENTE_SEGN" = b."ENTE_SEGN"
                ))) AS imb ON t."DATA_CONTABILE" = imb."DATA_CONTABILE" AND t."DIVISA" = imb."DIVISA" AND t."DURATA" = imb."DURATA" AND t."ENTE_SEGN" = imb."ENTE_SEGN"
```

---


