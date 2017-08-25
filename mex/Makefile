MATLABROOT=/usr/local/MATLAB/R2016a
CXX=g++

LIBS= -L$(MATLABROOT)/bin/glnxa64 -L$(MATLABROOT)/extern/lib/glnxa64 -lmex -lmx -lmwlapack -lmwblas -leng

CFLAG= -Wall -m64 -O3 -I$(MATLABROOT)/extern/include $(SRC) $(LIBS) -o $(EXE)
MEXFLAG=-m64 -shared -DMATLAB_MEX_FILE -I$(MATLABROOT)/extern/include -Wl,--export-all-symbols $(LIBS) $(MEXSRC) -o $(MEXTGT).mexw64

EXE=../bin/engwindemo.exe

SRC=engwindemo.c

all:$(EXE)

$(EXE):  $(SRC)
    $(CXX) $(CFLAG) -ladvapi32 -luser32 -lgdi32 -lkernel32 -lmingwex -o $(EXE)
		@rm -f *.o*

$(MEXTGT):  $(MEXSRC)
    $(CXX) $(MEXFLAG) -ladvapi32 -luser32 -lgdi32 -lkernel32 -lmingwex
    @rm -f *.o*
