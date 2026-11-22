
# YOLOv3-tiny INT8 C inference engine
#
# Usage:
# make test MODEL=../reference/results/ref_out_int8/model \
#           INPUT=../reference/results/ref_out_int8/int8/input.bin \
#           REF=../reference/results/ref_out_int8/int8

CC      = gcc
CFLAGS  ?= -O2 -std=c99 -Wall -Wextra -pedantic

MODEL   ?= ../reference/results/ref_out_int8/model
INPUT   ?= ../reference/results/ref_out_int8/int8/input.bin
REF     ?= ../reference/results/ref_out_int8/int8

THRESH  ?= 0.25
NMS     ?= 0.45

BUILD_DIR   := build
RESULTS_DIR := results/detections

SRCS := tests/test_host.c \
        src/yolo_int8.c \
        src/yolo_decode.c

HDRS := include/yolo_int8.h \
        include/yolo_decode.h

TARGET := $(BUILD_DIR)/test_host
DECODE_TABLES := $(BUILD_DIR)/decode_tables.h

.PHONY: all test asan clean

all: $(TARGET)

$(DECODE_TABLES): $(MODEL)/layers.json tools/gen_decode_tables.py
	mkdir -p $(BUILD_DIR)
	python3 tools/gen_decode_tables.py $(MODEL)/layers.json $@

$(TARGET): $(SRCS) $(HDRS) $(DECODE_TABLES) $(MODEL)/layers.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -Iinclude -I$(MODEL) -I$(BUILD_DIR) $(SRCS) -o $@

test: $(TARGET)
	mkdir -p $(RESULTS_DIR)
	$(TARGET) $(MODEL) $(INPUT) $(REF) $(THRESH) $(NMS) \
		$(RESULTS_DIR)/detections_c.txt

asan: $(DECODE_TABLES)
	mkdir -p $(BUILD_DIR)
	$(CC) -O1 -g -std=c99 -Wall -Wextra \
		-fsanitize=address,undefined -fno-sanitize-recover=undefined \
		-Iinclude -I$(MODEL) -I$(BUILD_DIR) $(SRCS) \
		-o $(BUILD_DIR)/test_host_asan

clean:
	rm -rf $(BUILD_DIR)

