#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

int main() {
	int fd = open("/home/sandbox/test.txt", O_RDWR | O_APPEND | O_CREAT);
	if (fd < 0) {
		perror("open");
		return -1;
	}
	char *msg = "hello world\n";
	ssize_t w = write(fd, msg, strlen(msg));
	printf("Bytes written: %ld\n", w);
	return 0;
}
