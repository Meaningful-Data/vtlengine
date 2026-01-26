CC = gcc
CFLAGS = -O3 -Wall -Wextra
DEBUG_FLAGS = -g -O0 -Wall -Wextra -DDEBUG
LDFLAGS = -lm

TARGET = file_generator
SOURCE = file_generator.c

.PHONY: all debug clean help

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

debug: $(SOURCE)
	$(CC) $(DEBUG_FLAGS) -o $(TARGET) $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

help:
	@echo "CSV File Generator - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make        - Build file_generator binary"
	@echo "  make debug  - Build with debug symbols"
	@echo "  make clean  - Remove binary"
	@echo "  make help   - Show this help"
	@echo ""
	@echo "After building, run:"
	@echo "  python file_generator.py --rows 1000000"
