BINDIR = bin
LIBDIR = lib
SRCDIR = src

OBLIVC_PATH = lib/obliv-c
OBLIVCC = $(OBLIVC_PATH)/bin/oblivcc

SOURCES := $(shell find -L ${SRCDIR} -type f -name '*.cpp' -not -path '*/cmd/*')
OBJECTS := $(patsubst %.cpp, %.o, ${SOURCES})
SOURCES_OBLIVC := $(shell find -L ${SRCDIR} -type f -name '*.oc')
OBJECTS_OBLIVC := $(patsubst %.oc, %.oo, ${SOURCES_OBLIVC})
SOURCES_BIN := $(shell find -L ${SRCDIR}/cmd -type f -name '*.cpp')
OBJECTS_BIN := $(patsubst %.cpp, %.o, ${SOURCES_BIN})

LIBRARIES := $(LIBDIR)/libscapi.a $(LIBDIR)/libobliv.a $(LIBDIR)/liback.a
BINARIES := $(patsubst ${SRCDIR}/cmd/%.cpp, ${BINDIR}/%, ${SOURCES_BIN})

LDFLAGS = -lpthread -l:libssl.so.1.0.0 -l:libcrypto.so.1.0.0 -lrt $(LIBRARIES) \
	-lgmpxx -lboost_program_options -lboost_system -lboost_thread -lboost_iostreams \
	-lboost_serialization -lgmp -lgcrypt
INCLUDES = -Isrc -Ilib/ABY/src -Ilib/libscapi/include \
	-I$(OBLIVC_PATH)/src/ext/oblivc -Ilib/liback/src
CXXFLAGS = -O3 -pthread $(INCLUDES) -g -std=gnu++11
OBLIVCCFLAGS = -g

export CXXFLAGS

all: build_libraries
	$(MAKE) $(BINARIES)

# extract dependencies
%.d: %.cpp $(LIBRARIES)
	@set -e; rm -f $@; \
	$(CXX) -MM $(CXXFLAGS) -MT "$*.o $@" $< > $@;

%.od: %.oc $(LIBRARIES)
	@set -e; rm -f $@; \
	$(CXX) -MM $(CXXFLAGS) -MT "$*.oo $@" $< > $@;

-include $(SOURCES:.cpp=.d)
-include $(SOURCES_BIN:.cpp=.d)
-include $(SOURCES_OBLIVC:.oc=.od)

$(BINDIR)/%: $(LIBRARIES) $(OBJECTS) $(OBJECTS_OBLIVC) $(SRCDIR)/cmd/%.o
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(OBJECTS_OBLIVC) $(SRCDIR)/cmd/$*.o -o $@ $(LDFLAGS)

# do not delete intermediate objects
.SECONDARY: $(OBJECTS) $(OBJECTS_BIN) $(OBJECTS_OBLIVC)

# compile using obliv-c
%.oo : %.oc
	$(OBLIVCC) $(INCLUDES) $(OBLIVCCFLAGS) -o $@ -c $^

# update libraries if necessary
$(LIBDIR)/%.a: $(build_libraries)

.PHONY: build_libraries
build_libraries:
	$(MAKE) -C $(LIBDIR)

# cleanup
.PHONY: clean cleanall
clean:
	$(RM) -r $(BINDIR)
	$(RM) $(OBJECTS) $(OBJECTS_BIN) $(OBJECTS_OBLIVC)
	$(RM) $(OBJECTS:.o=.d) $(OBJECTS_BIN:.o=.d) $(OBJECTS_OBLIVC:.oo=.od)

cleanall: clean
	$(MAKE) -C $(LIBDIR) cleanall
