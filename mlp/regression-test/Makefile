export CC_USE_MATH = 1

TARGET   = regression-test
CLASSES  =
SOURCE   = $(TARGET).c $(CLASSES:%=%.c)
OBJECTS  = $(TARGET).o $(CLASSES:%=%.o)
HFILES   = $(CLASSES:%=%.h)
OPT      = -O2 -Wall
CFLAGS   = $(OPT) -I.
LDFLAGS  = -Llibai -lai -Llibcc -lcc -ldl -lpthread -lm
CCC      = gcc

all: $(TARGET)

$(TARGET): $(OBJECTS) libai libcc
	$(CCC) $(OPT) $(OBJECTS) -o $@ $(LDFLAGS)

.PHONY: libai libcc

libai:
	$(MAKE) -C libai

libcc:
	$(MAKE) -C libcc

clean:
	rm -f $(OBJECTS) *~ \#*\# $(TARGET)
	$(MAKE) -C libai clean
	$(MAKE) -C libcc clean
	rm libai libcc

$(OBJECTS): $(HFILES)
