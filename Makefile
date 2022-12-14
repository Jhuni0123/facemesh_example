all: main gstcropscale.so

gstcropscale.o: gstcropscale.c
	gcc -Wall -fPIC -O2 -c -o gstcropscale.o gstcropscale.c `pkg-config --cflags --libs gstreamer-1.0 nnstreamer`

gstcropscale.so: gstcropscale.o
	gcc -shared -o gstcropscale.so gstcropscale.o `pkg-config --cflags --libs gstreamer-1.0 nnstreamer`

main: main.c face_detect.c
	gcc -Wall -O2 main.c -o main `pkg-config --cflags --libs gstreamer-1.0 nnstreamer` -D DBG -lm

clean:
	rm -f main gstcropscale.o gstcropscale.so
