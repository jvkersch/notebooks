#include <stdio.h>
#include <unistd.h>
#include <limits.h>

int main(int argc, char **argv) {
	printf("Command-line arguments:\n");
	for (int i = 0; i < argc; i++) {
		printf("%d: %s\n", i, argv[i]);
	}

	char cwd[PATH_MAX];
	if (getcwd(cwd, sizeof(cwd)) != NULL) {
		printf("Working directory: %s\n", cwd);
	}
	return 0;
}
