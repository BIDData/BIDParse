include Makefile.incl

OBJS=$(shell echo rules/*.cu | sed -e s/\.cu/.$(OBJ)/g | sed -e s+rules\/+bin\/+g) bin/BIDMat_PARSE.$(OBJ) bin/xPcopy.$(OBJ) bin/xViterbi.$(OBJ)

LIBFILE=$(LIBPREPEND)jniparse$(LIBAPPEND)

.SUFFIXES: .$(OBJ) .c

all: $(LIBPREPEND)jniparse$(LIBAPPEND)

$(LIBPREPEND)jniparse$(LIBAPPEND): $(OBJS)
	$(LD) $(LDFLAGS) $(OBJS) $(CUDA_LIBS) $(OUTFLG)$@

bin/%.$(OBJ) : %.c
	icl $(CPPFLAGS) $(CFLAGS) $(OUTFLG)$*.$(OBJ) $*.c

bin/%.$(OBJ) : %.cpp
	icl $(CPPFLAGS) $(CFLAGS) $(OUTFLG)$*.$(OBJ) $*.cpp

bin/%.$(OBJ) : %.cu testSparse.h
	$(NVCC) $(NVCCFLAGS) -o bin/$*.$(OBJ) $*.cu

bin/%.$(OBJ) : rules/%.cu testSparse.h
	$(NVCC) $(NVCCFLAGS) -Xptxas -dlcm=cg -o bin/$*.$(OBJ) rules/$*.cu

install: /code/bin/$(LIBFILE)

/code/bin/$(LIBFILE): $(LIBFILE)
	cp $(LIBFILE) /code/bin

clean:
	rm -f *.$(OBJ) *.lib *.pdb *$(LIBAPPEND) bin/*

distclean: clean
	rm -f *$(LIBAPPEND) *.exp *.lib *.jnilib Makefile.incl

