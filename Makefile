TARGET   = path
OBJ_DIR  = obj
SRC_DIR  = src

SRC     := $(wildcard $(SRC_DIR)/*.c)
OBJ 	:= $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC))

CC       = clang
CFLAGS   = -Wall -g -Iinclude -fopenmp
OPTFLAGS = -O2
LDFLAGS  = -lm -fopenmp
LDLIBS   =

.PHONY: all clean opt

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

opt: $(OBJ)
	$(CC) $(OPTFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $(TARGET)


$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $@

clean:
	$(RM) -rv $(OBJ_DIR) $(TARGET)

-include $(OBJ:.o=.d)
