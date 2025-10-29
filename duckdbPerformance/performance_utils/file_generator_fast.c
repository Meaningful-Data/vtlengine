#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <time.h>
#include <math.h>

#define ASCII_CHARS 26
#define MAX_COLS 256
#define MAX_NAME 128
#define MAX_TYPE 32
#define MAX_PATHLEN 2048

typedef enum {
    TYPE_INTEGER = 0,
    TYPE_NUMBER  = 1,
    TYPE_BOOL = 2,
    TYPE_STR  = 3,
    TYPE_INVALID = -1
} TypeCode;

typedef struct {
    char name[MAX_NAME];
    char type[MAX_TYPE];
    int is_identifier;
    TypeCode type_code;
} Column;

typedef struct {
    char *data;
    size_t size;
    size_t cap;
} Buffer;

static void buf_init(Buffer *b, size_t cap) {
    if (cap == 0) cap = 16;
    b->data = (char*)malloc(cap);
    b->size = 0;
    b->cap  = cap;
}

static void buf_free(Buffer *b) {
    free(b->data);
    b->data = NULL;
    b->size = b->cap = 0;
}

static void buf_reserve(Buffer *b, size_t add) {
    if (b->size + add <= b->cap) return;
    size_t nc = b->cap ? b->cap : 16;
    while (nc < b->size + add) nc <<= 1;
    char *p = (char*)realloc(b->data, nc);
    if (!p) {
        fprintf(stderr, "Out of memory\n");
        exit(99);
    }
    b->data = p;
    b->cap  = nc;
}

static void buf_put_char(Buffer *b, char c) {
    if (b->size + 1 > b->cap) buf_reserve(b, 1);
    b->data[b->size++] = c;
}

static void buf_write(Buffer *b, const char *s, size_t n) {
    buf_reserve(b, n);
    memcpy(b->data + b->size, s, n);
    b->size += n;
}

static int starts_with_id(const char *s) {
    if (!s || !s[0]) return 0;
    return (tolower((char)s[0])=='i' && tolower((char)s[1])=='d');
}

static int min_str_length(uint64_t length) {
    if (length <= 1) return 1;
    double v = log((double)length) / log((double)ASCII_CHARS);
    int r = (int)ceil(v);
    return r < 1 ? 1 : r;
}

static void str_from_int(uint64_t n, int width, char *out) {
    for (int i = 0; i < width; ++i) out[i] = 'A';
    int idx = width - 1;

    if (n == 0) {
        out[idx] = 'A';
    }
    else {
        while (n > 0 && idx >= 0) {
            out[idx--] = (char)('A' + (int)(n % ASCII_CHARS));
            n /= ASCII_CHARS;
        }
    }

    out[width] = '\0';
}

// Faster way to generate pseudo-random numbers than rand()
static uint32_t rand_xor_shift(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

static void write_int(Buffer *b, uint64_t v) {
    char tmp[32];
    int i = 0;
    if (v == 0) {
        buf_put_char(b, '0');
        return;
    }

    while (v > 0) {
        tmp[i++] = (char)('0' + (v % 10));
        v /= 10;
    }

    while (i--) buf_put_char(b, tmp[i]);
}

static void write_double(Buffer *b, double x) {
    char tmp[64];
    int n = snprintf(tmp, sizeof(tmp), "%.6f", x);
    if (n > 0) buf_write(b, tmp, (size_t)n);
}

static void write_bool(Buffer *b, int val) {
    if (val) buf_write(b, "True", 4);
    else buf_write(b, "False", 5);
}

static void random_string(uint32_t *st, int len, char *out) {
    static const char chars[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    const int L = (int)(sizeof(chars) - 1);
    for (int i = 0; i < len; ++i) {
        out[i] = chars[rand_xor_shift(st) % L];
    }
    out[len] = '\0';
}

typedef struct {
    uint64_t rows;
    uint64_t chunk_size;
    uint64_t max_num;
    int max_str;
    char csv_path[MAX_PATHLEN];
    char ds_path[MAX_PATHLEN];
    int columns;
    Column cols[MAX_COLS];
} Config;

static TypeCode map_type(const char *t) {
    if (strcmp(t, "Integer") == 0) return TYPE_INTEGER;
    else if (strcmp(t, "Number")  == 0) return TYPE_NUMBER;
    else if (strcmp(t, "Boolean") == 0) return TYPE_BOOL;
    else if (strcmp(t, "String")  == 0) return TYPE_STR;
    return TYPE_INVALID;
}

static int parse_config(const char *path, Config *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("open config");
        return 0;
    }

    char line[4096];
    int col_index = 0;

    while (fgets(line, sizeof(line), f)) {
        char *nl = strchr(line, '\n'); if (nl) *nl = 0;

        if (!strncmp(line, "rows=", 5)) cfg->rows = strtoull(line+5, NULL, 10);
        else if (!strncmp(line, "chunk_size=",11)) cfg->chunk_size = strtoull(line+11, NULL, 10);
        else if (!strncmp(line, "max_num=", 8)) cfg->max_num = strtoull(line+8, NULL, 10);
        else if (!strncmp(line, "max_str=", 8)) cfg->max_str = atoi(line+8);
        else if (!strncmp(line, "csv_path=", 9)) strncpy(cfg->csv_path, line+9, sizeof(cfg->csv_path)-1);
        else if (!strncmp(line, "ds_path=", 8)) strncpy(cfg->ds_path, line+8, sizeof(cfg->ds_path)-1);
        else if (!strncmp(line, "columns=", 8)) cfg->columns = atoi(line+8);
        else if (!strncmp(line, "col=", 4)) {
            if (col_index >= MAX_COLS) {
                fclose(f);
                return 0;
            }

            char *spec = line + 4;
            char *bar  = strchr(spec, '|');
            if (!bar) {
                fclose(f);
                return 0;
            }
            *bar = 0;

            Column *c = &cfg->cols[col_index++];
            strncpy(c->name, spec, MAX_NAME-1);
            strncpy(c->type, bar+1, MAX_TYPE-1);
            c->is_identifier = starts_with_id(spec);
            c->type_code     = map_type(c->type);
        }
    }
    fclose(f);

    if (col_index != cfg->columns) return 0;
    if (cfg->rows == 0 || cfg->chunk_size == 0) return 0;
    for (int i = 0; i < cfg->columns; ++i) if (cfg->cols[i].type_code == TYPE_INVALID) return 0;

    return 1;
}

typedef struct {
    int id_count;
    int id_pos_of_col[MAX_COLS];
    uint64_t *bases;
    uint64_t *factors;
    int str_len;
} IdPlan;

static int build_id_plan(const Config *cfg, IdPlan *plan) {
    memset(plan, 0, sizeof(*plan));
    for (int i = 0; i < MAX_COLS; ++i) plan->id_pos_of_col[i] = -1;

    for (int i = 0; i < cfg->columns; ++i)
        if (cfg->cols[i].is_identifier) plan->id_count++;

    plan->str_len = min_str_length(cfg->rows);

    if (plan->id_count == 0) return 1;

    plan->bases   = (uint64_t*)malloc(sizeof(uint64_t) * (size_t)plan->id_count);
    plan->factors = (uint64_t*)malloc(sizeof(uint64_t) * (size_t)plan->id_count);
    if (!plan->bases || !plan->factors) return 0;

    int p = 0;
    for (int i = 0; i < cfg->columns; ++i) {
        if (cfg->cols[i].is_identifier) {
            plan->id_pos_of_col[i] = p;
            if (cfg->cols[i].type_code == TYPE_INTEGER) {
                plan->bases[p] = (uint64_t)cfg->max_num;
            }
            else if (cfg->cols[i].type_code == TYPE_STR) {
                double pw = pow((double)ASCII_CHARS, (double)plan->str_len);
                if (pw > 1e19) pw = 1e19;
                plan->bases[p] = (uint64_t)pw;
            }
            else {
                plan->bases[p] = 1;
            }
            p++;
        }
    }

    uint64_t prod = 1;
    for (int i = 0; i < plan->id_count; ++i) {
        plan->factors[i] = prod;
        if (plan->bases[i] == 0) return 0;
        if (prod > UINT64_MAX / plan->bases[i]) {
            fprintf(stderr, "Overflow calculating ids combinations\n");
            return 0;
        }
        prod *= plan->bases[i];
    }

    uint64_t total = plan->factors[plan->id_count - 1] * plan->bases[plan->id_count - 1];
    if (cfg->rows > total) {
        fprintf(stderr, "Cannot generate %llu unique rows with defined ids. Max: %llu\n",
                (unsigned long long)cfg->rows, (unsigned long long)total);
        return 0;
    }
    return 1;
}

static void free_id_plan(IdPlan *plan) {
    free(plan->bases);
    free(plan->factors);
    plan->bases = plan->factors = NULL;
}

static int write_header(FILE *out, const Config *cfg) {
    for (int i = 0; i < cfg->columns; ++i) {
        const char *name = cfg->cols[i].name;
        fwrite(name, 1, strlen(name), out);
        if (i < cfg->columns - 1) fputc(',', out);
    }
    fputc('\n', out);
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;

    Config cfg;
    if (!parse_config(argv[1], &cfg)) {
        fprintf(stderr, "Invald config\n");
        return 2;
    }

    IdPlan plan;
    if (!build_id_plan(&cfg, &plan)) {
        free_id_plan(&plan);
        return 3;
    }

    FILE *out = fopen(cfg.csv_path, "wb");
    if (!out) {
        perror("Open csv");
        free_id_plan(&plan);
        return 4;
    }
    setvbuf(out, NULL, _IOFBF, 16 * 1024 * 1024);

    write_header(out, &cfg);

    Buffer buf;
    buf_init(&buf, 1 << 20);

    uint32_t seed_base = (uint32_t)time(NULL);
    char idbuf[64];
    char randbuf[512];

    uint64_t written = 0;
    uint64_t total_chunks = (cfg.rows + cfg.chunk_size - 1) / cfg.chunk_size;
    uint64_t chunk_idx = 0;

    while (written < cfg.rows) {
        uint64_t cur = cfg.chunk_size;
        if (written + cur > cfg.rows) cur = cfg.rows - written;

        buf.size = 0;

        for (uint64_t r = 0; r < cur; ++r) {
            uint64_t g = written + r;

            for (int c = 0; c < cfg.columns; ++c) {
                const Column *col = &cfg.cols[c];

                if (col->is_identifier) {
                    int pos = plan.id_pos_of_col[c];

                    if (plan.id_count <= 1) {
                        if (col->type_code == TYPE_INTEGER) {
                            write_int(&buf, g);
                        }
                        else if (col->type_code == TYPE_STR) {
                            str_from_int(g, plan.str_len, idbuf);
                            buf_write(&buf, idbuf, (size_t)plan.str_len);
                        }
                    }
                    else {
                        uint64_t base   = plan.bases[pos];
                        uint64_t factor = plan.factors[pos];
                        uint64_t val    = (g / factor) % base;

                        if (col->type_code == TYPE_INTEGER) {
                            write_int(&buf, val);
                        }
                        else if (col->type_code == TYPE_STR) {
                            str_from_int(val, plan.str_len, idbuf);
                            buf_write(&buf, idbuf, (size_t)plan.str_len);
                        }
                    }
                }
                else {
                    uint32_t s = seed_base ^ (uint32_t)g ^ (uint32_t)(c * 0x9E37);
                    switch (col->type_code) {
                        case TYPE_INTEGER: {
                            uint32_t v = rand_xor_shift(&s);
                            uint64_t val = (cfg.max_num == 0) ? 0 : (uint64_t)(v % (uint32_t)cfg.max_num);
                            write_int(&buf, val);
                        } break;
                        case TYPE_NUMBER: {
                            double u = (double)rand_xor_shift(&s) / 4294967296;
                            write_double(&buf, u * (double)cfg.max_num);
                        } break;
                        case TYPE_BOOL: {
                            write_bool(&buf, (int)(rand_xor_shift(&s) & 1u));
                        } break;
                        case TYPE_STR: {
                            int L = cfg.max_str > 0 ? cfg.max_str : 12;
                            if (L > (int)sizeof(randbuf)-1) L = (int)sizeof(randbuf)-1;
                            random_string(&s, L, randbuf);
                            buf_write(&buf, randbuf, (size_t)L);
                        } break;
                        default: {
                            buf_write(&buf, "NA", 2);
                        } break;
                    }
                }

                if (c < cfg.columns - 1) buf_put_char(&buf, ',');
            }
            buf_put_char(&buf, '\n');
        }

        fwrite(buf.data, 1, buf.size, out);
        written += cur;
        chunk_idx++;

        printf("  -> Written: %llu / %llu rows (%llu / %llu chunks)\n",
               (unsigned long long)written,
               (unsigned long long)cfg.rows,
               (unsigned long long)chunk_idx,
               (unsigned long long)total_chunks);
        fflush(stdout);
    }

    buf_free(&buf);
    free_id_plan(&plan);
    fclose(out);
    return 0;
}
