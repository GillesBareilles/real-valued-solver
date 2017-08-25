CXX=g++
CFLAGS=-W -Wall -ansi -pedantic -std=c++11 -I"./include" -O3 -m64 -I${MKLROOT}/include -g -fopenmp
LDFLAGS=  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl

# If we compile for sequential code
ifndef OMP
	CFLAGS= -W  -ansi -pedantic  -Wall -std=c++11 -I"./include" -I"./include/spectra-0.5.0/include" -O3 -m64 -I${MKLROOT}/include # -g -fopenmp
	LDFLAGS=  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
endif


EXEC=main

SRCDIR   = src
OBJDIR   = obj
BINDIR   = bin

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
OBJECTS  = $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

# .PHONY: main.o

# all: $(BINDIR)/$(EXEC)

# $(BINDIR)/$(EXEC): $(OBJECTS)
# 	$(CXX) -o $@ $^ $(LDFLAGS)

all: main
	@ echo -------------------------------------------------------
	@ echo
	@ ./bin/main

main: obj/main.o obj/getRAMusage.o obj/utils.o
	$(CXX) -o $(BINDIR)/$@ $^ $(LDFLAGS)

timings: obj/timings.o obj/getRAMusage.o obj/utils.o
	$(CXX) -o $(BINDIR)/$@ $^ $(LDFLAGS)

obj/timings.o : src/timings.cpp include/RealValuedSolverLLT.hpp include/RealValuedSolverLDLT.hpp include/RealValuedSolverPardiso.hpp

obj/main.o : src/main.cpp include/RealValuedSolverLLT.hpp include/RealValuedSolverLDLT.hpp include/RealValuedSolverPardiso.hpp

obj/utils.o : ./src/utils.cpp ./src/utils.hpp

obj/getRAMusage.o: src/getRAMusage.hpp

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CFLAGS) -o $@ -c $<


clean:
	@rm -rf obj/*.o
	@rm -rf bin/*


@ Reminder :
@  - $@ : current build target
@  - $^ : recall dependencies
@  - $< : recall dependencies
@  -.PHONY : dependencies always rebuilt
