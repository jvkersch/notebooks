#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
	FILE *f = fopen("/mnt/data/data.txt", "r");
	if (f == NULL) {
		perror("fopen");
		return -1;
	}
	char buf[256];
	size_t r = fread(buf, sizeof(buf), 1, f);
	printf("Data read: %s\n", buf);
	return 0;
}
