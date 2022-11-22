all: main

main: main.c
	gcc main.c -o main `pkg-config --cflags --libs gstreamer-1.0 nnstreamer` -D DBG -lm

clean:
	rm -f main
