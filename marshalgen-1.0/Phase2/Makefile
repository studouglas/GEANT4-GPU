CFLAGS=-g -O0
CC=gcc ${CFLAGS}

# test: marshalgen
#	@ echo ""
#	./marshal < marshalgen.in

marshal: marshalgen.tab.o lex.yy.o mast.o
	${CC} -o marshal marshalgen.tab.o lex.yy.o mast.o

mast.o : mast.c mast.h
	${CC} -c mast.c

# -d causes marshalgen.tab.h to be generated
marshalgen.tab.c marshalgen.tab.h: marshalgen.y
	bison -d -v marshalgen.y

lex.yy.o: lex.yy.c marshalgen.tab.h
	${CC} -c lex.yy.c

lex.yy.c: marshalgen.l
	flex marshalgen.l

marshalgen.tab.o: marshalgen.tab.c
	${CC} -o marshalgen.tab.o -c -DYYERROR_VERBOSE marshalgen.tab.c

clean:
	rm -f core marshalgen marshalgen.tab.h *.o *~ marshalgen.output\
	  marshalgen.tab.c lex.yy.c marshalgen.output

distclean: clean
	rm -f marshal

dist: distclean
	dir=`basename $$PWD`; cd ..; tar cvf $$dir.tar ./$$dir; \
          gzip $$dir.tar && ( cd $$dir; ls -l ../$$dir.tar.gz )

test-debug: marshalgen.tab.c lex.yy.o
	rm -f marshalgen.tab.o
	make DEBUG=-DYYDEBUG test
