TARGET  = libai.a
CLASSES = \
	mlp/ai_mlp      \
	mlp/ai_mlpFact  \
	mlp/ai_mlpLayer
SOURCE  = $(CLASSES:%=%.c)
OBJECTS = $(SOURCE:.c=.o)
HFILES  = $(CLASSES:%=%.h)
OPT     = -O2 -Wall
CFLAGS  = $(OPT)
LDFLAGS = -lm
AR      = ar

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(AR) rcs $@ $(OBJECTS)

clean:
	rm -f $(OBJECTS) *~ \#*\# $(TARGET)

$(OBJECTS): $(HFILES)
