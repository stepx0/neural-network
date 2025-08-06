CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11

SRCDIR = src
BUILDDIR = build

SRCS = activation.c loss.c tensor_utils.c train.c layer.c matrix.c xor_dataset.c main.c
OBJS = $(patsubst %.c,$(BUILDDIR)/%.o,$(SRCS))

TARGET = train

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -Wl,-e,mainCRTStartup

$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR) $(TARGET)
