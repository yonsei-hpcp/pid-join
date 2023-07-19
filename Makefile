CXX = g++-11
DPU_OPTS = `dpu-pkg-config --cflags --libs dpu` 

.PHONY: default all clean lib test

SRCS  = $(shell find ./src     -type f -name *.cc)
TEST_JOIN_SRCS  = $(shell find ./test/join     -type f -name *.cc)
TEST_USG_SRCS = $(shell find ./test/usg -type f -name *.cc)
INCLUDES := -I./include -I./src/dpu/include
#INTEL_ADVISOR_HEADER_DIR = -I/opt/intel/oneapi/advisor/latest/sdk/include
LIBS= -lnuma -ldl #-L/opt/intel/oneapi/advisor/latest/sdk/lib64 -littnotify 
OBJS = $(SRCS:.cc=.o_lib)
TEST_JOIN_OBJS = $(TEST_JOIN_SRCS:.cc=.o_test)
TEST_USG_OBJS =$(TEST_USG_SRCS:.cc=.o_test)

HEADS = $(shell find ./src     -type f -name *.hpp)
TARGET = ./lib/libpidjoin.so 
TEST_JOIN_TARGET = pidjoin_test.bin
all: lib test

# Default mode
lib: CFLAGS = -std=c++14 -O3 -pthread -g #-DCOLLECT_LOGS #-DINTEL_ITTNOTIFY_API
#default: INCLUDES += $(INTEL_ADVISOR_HEADER_DIR)
lib: $(TARGET) subsystem

test: $(TEST_JOIN_TARGET)

$(TARGET): $(OBJS) $(HEADS)
	$(CXX) $(CFLAGS) -shared -fPIC -o $@ $(OBJS) -DBLOCK_SIZE=$(BLOCK_SIZE) $(LIBS)  $(INCLUDES) $(DPU_OPTS)

%.o_lib:%.cc
	$(CXX) $(CFLAGS) -fPIC -c $< -o $@ -DBLOCK_SIZE=$(BLOCK_SIZE) $(LIBS)  $(INCLUDES) $(DPU_OPTS)

$(TEST_JOIN_TARGET): $(TEST_JOIN_OBJS) $(HEADS)
	$(CXX) $(CFLAGS) -o $@ $(TEST_JOIN_OBJS) -DBLOCK_SIZE=$(BLOCK_SIZE) $(LIBS) -L./lib -lpidjoin $(INCLUDES) $(DPU_OPTS)

$(TEST_USG_TARGET): $(TEST_USG_OBJS) $(HEADS)
	$(CXX) $(CFLAGS) -o $@ $(TEST_USG_OBJS) -DBLOCK_SIZE=$(BLOCK_SIZE) $(LIBS) -L./lib -lpidjoin $(INCLUDES) $(DPU_OPTS)

%.o_test:%.cc
	$(CXX) $(CFLAGS) -c $< -o $@ -DBLOCK_SIZE=$(BLOCK_SIZE) $(LIBS) -L./lib -lpidjoin  $(INCLUDES) $(DPU_OPTS)

subsystem:
	$(MAKE) -C src/dpu;

clean:
	cd src/dpu && $(MAKE) clean
	rm -f $(TARGET)
	rm -f $(OBJS)
	
