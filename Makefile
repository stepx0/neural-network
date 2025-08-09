CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11

SRCDIR = src
BUILDDIR = build

SRCS = activation_functions.c loss_functions.c tensor.c train.c layers.c main.c
OBJS = $(patsubst %.c,$(BUILDDIR)/%.o,$(SRCS))

TARGET = train

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR) $(TARGET)
