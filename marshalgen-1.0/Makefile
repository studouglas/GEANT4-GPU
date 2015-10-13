
compile:
	cd Phase2 && make

dist: distclean
	dir=`basename $$PWD`; cd ..; tar cvf $$dir.tar ./$$dir && gzip $$dir.tar
	dir=`basename $$PWD`; ls -l ../$$dir.tar.gz

clean:
	cd Phase2 && make clean

distclean:
	cd Phase2 && make distclean
