export CC_USE_MATH = 1
export CC_USE_RNG  = 1

TARGET   = regression-test
CLASSES  =
SOURCE   = $(TARGET).c $(CLASSES:%=%.c)
OBJECTS  = $(TARGET).o $(CLASSES:%=%.o)
HFILES   = $(CLASSES:%=%.h)
OPT      = -O2 -Wall
CFLAGS   = $(OPT) -I.
LDFLAGS  = -Llibai -lai -Ljsmn/wrapper -l jsmn -Llibcc -lcc -ldl -lpthread -lm
CCC      = gcc

all: $(TARGET)

$(TARGET): $(OBJECTS) libai libcc jsmn
	$(CCC) $(OPT) $(OBJECTS) -o $@ $(LDFLAGS)

.PHONY: libai libcc jsmn

libai:
	$(MAKE) -C libai

libcc:
	$(MAKE) -C libcc

jsmn:
	$(MAKE) -C jsmn/wrapper

clean:
	rm -f $(OBJECTS) *~ \#*\# $(TARGET)
	$(MAKE) -C libai clean
	$(MAKE) -C libcc clean
	$(MAKE) -C jsmn/wrapper clean
	rm libai libcc jsmn pcg-c-basic

$(OBJECTS): $(HFILES)
