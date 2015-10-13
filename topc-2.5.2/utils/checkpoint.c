#ifdef __GNUC__

// For pre-elf, something like this would be enough.

int main() {
	extern unsigned long *__data_start;
	extern unsigned long *data_start;
	extern unsigned long *_edata;
	printf("start: %x;  end: %x\n", __data_start, _edata);
	printf("start: %x;  end: %x\n", data_start, _edata);
}
#endif
