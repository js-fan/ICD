CXX = g++

# add path for OpenCV if necessary
OPENCV_INCLUDE = #-I/your/OpenCV/include
OPENCV_LIBRARY = #-L/your/OpenCV/lib

CFLAGS = $(OPENCV_INCLUDE) -Iinclude -std=c++11 -O2
CLINKS = $(OPENCV_LIBRARY) -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lboost_system -lboost_filesystem

SRCDIR = src
OBJDIR = build

SRCFILES = $(wildcard $(SRCDIR)/*.cpp)
OBJFILES = $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRCFILES))


CFLAGS += -fopenmp -Duse_openmp_


makeSuperpixel: $(OBJFILES) makeSuperpixel.cpp
	$(CXX) -o $@ $^ $(CFLAGS) $(CLINKS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) -c -o $@ $< $(CFLAGS)


clean:
	rm -f build/*.o build/*.so makeSuperpixel
