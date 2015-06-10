LPGPLOT = -L/usr/local/pgplot -lpgplot -lcpgplot -L/usr/X11R6/lib -lpng -lz -lX11
LFFTW = -lfftw3f

INCDIR = /usr/local/src/psrdada/src
LIBDIR = /usr/local/src/psrdada/src
IPCFILES = $(INCDIR)/ipcio.c $(INCDIR)/ipcbuf.c $(INCDIR)/ipcutil.c $(INCDIR)/ascii_header.c

DEDISP_CFLAGS = -I/usr/local/src/dedisp-read-only/include
DEDISP_LIBS = -L/usr/local/src/dedisp-read-only/lib -ldedisp

F77 = gfortran -std=legacy -ffixed-line-length-none 
CC = gcc
NVCC = nvcc

all: dat2vdif vdif2fil

dat2vdif: dat2vdif.c
	cc -I/usr/local/include -o dat2vdif dat2vdif.c -L/usr/local/lib -lvdifio

vdif2fil: vdif2fil.c
	$(CC) -g -O2 -I/usr/local/include -I/usr/local/pgplot -c vdif2fil.c
	$(F77) -o vdif2fil vdif2fil.o $(LPGPLOT) $(LFFTW) -lm -lvdifio
#	cc -I/usr/local/include -I/usr/local/pgplot -o vdif2fil vdif2fil.c -L/usr/local/lib -lvdifio $(LPGPLOT) $(LFFTW) -lm

original_vdif2fil: original_vdif2fil.c
	$(CC) -g -O2 -I/usr/local/include -I/usr/local/pgplot -c original_vdif2fil.c
	$(F77) -o original_vdif2fil original_vdif2fil.o $(LPGPLOT) $(LFFTW) -lm -lvdifio

CUDA_vdif2fil: CUDA_vdif2fil.cu
	$(NVCC) -g -O2 -I/usr/local/cuda -I/usr/local/include -I/usr/local/pgplot -c CUDA_vdif2fil.cu
	$(NVCC) -o CUDA_vdif2fil CUDA_vdif2fil.o -L/usr/local/cuda/lib64 $(LPGPLOT) -lm -lvdifio -lcufft -lgfortran

CUDA_vdif2fil_DADA: CUDA_vdif2fil_DADA.cu
	$(NVCC) -g -O2 -I/usr/local/cuda -I/usr/local/include -I/usr/local/pgplot -I$(INCDIR) -c CUDA_vdif2fil_DADA.cu $(IPCFILES)
	$(NVCC) -o CUDA_vdif2fil_DADA CUDA_vdif2fil_DADA.o ipcio.o ipcbuf.o ipcutil.o ascii_header.o -L/usr/local/cuda/lib64 $(LPGPLOT) -lm -lvdifio -lcufft -lgfortran -L$(LIBDIR) 

vdifDADA2stats: vdifDADA2stats.cu
	$(NVCC) -g -O2 -I/usr/local/cuda -I/usr/local/include -I$(INCDIR) -c vdifDADA2stats.cu $(IPCFILES)
	$(NVCC) -o vdifDADA2stats vdifDADA2stats.o ipcio.o ipcbuf.o ipcutil.o ascii_header.o -L/usr/local/cuda/lib64 -lm -lvdifio -lcufft -lgfortran -L$(LIBDIR) 

vdifDADA2fil_PIPELINE: vdifDADA2fil_PIPELINE.cu
	$(NVCC) -g -O2 -I/usr/local/cuda -I/usr/local/include -I$(INCDIR) -c vdifDADA2fil_PIPELINE.cu $(IPCFILES)
	$(NVCC) -o vdifDADA2fil_PIPELINE vdifDADA2fil_PIPELINE.o ipcio.o ipcbuf.o ipcutil.o ascii_header.o -L/usr/local/cuda/lib64 -lm -lvdifio -lcufft -lgfortran -L$(LIBDIR) 

vdifFILE2fil_PIPELINE: vdifFILE2fil_PIPELINE.cu
	$(NVCC) -g -O2 -I/usr/local/cuda -I/usr/local/include -I$(INCDIR) -c vdifFILE2fil_PIPELINE.cu $(IPCFILES)
	$(NVCC) -o vdifFILE2fil_PIPELINE vdifFILE2fil_PIPELINE.o ipcio.o ipcbuf.o ipcutil.o ascii_header.o -L/usr/local/cuda/lib64 -lm -lvdifio -lcufft -lgfortran -L$(LIBDIR) 

vdifFILE2fil_PIPELINE_TEST: vdifFILE2fil_PIPELINE_TEST.cu
	$(NVCC) -g -O2 -I/usr/local/cuda -I/usr/local/include -I$(INCDIR) -c vdifFILE2fil_PIPELINE_TEST.cu $(IPCFILES)
	$(NVCC) -o vdifFILE2fil_PIPELINE_TEST vdifFILE2fil_PIPELINE_TEST.o ipcio.o ipcbuf.o ipcutil.o ascii_header.o -L/usr/local/cuda/lib64 -lm -lvdifio -lcufft -lgfortran -L$(LIBDIR) 


vdifFILE2dedisp_PIPELINE: vdifFILE2dedisp_PIPELINE.cu
	$(NVCC) -g -O2 -I/usr/local/cuda -I/usr/local/include -I$(INCDIR) $(DEDISP_CFLAGS) -c vdifFILE2dedisp_PIPELINE.cu $(IPCFILES)
	$(NVCC) -o vdifFILE2dedisp_PIPELINE vdifFILE2dedisp_PIPELINE.o ipcio.o ipcbuf.o ipcutil.o ascii_header.o -L/usr/local/cuda/lib64 -lm -lvdifio -lcufft -lgfortran -L$(LIBDIR) $(DEDISP_LIBS)

vdifFILE2dedisp_PIPELINE_TEST: vdifFILE2dedisp_PIPELINE_TEST.cu
	$(NVCC) -g -O2 -I/usr/local/cuda -I/usr/local/include -I$(INCDIR) $(DEDISP_CFLAGS) -c vdifFILE2dedisp_PIPELINE_TEST.cu $(IPCFILES)
	$(NVCC) -o vdifFILE2dedisp_PIPELINE_TEST vdifFILE2dedisp_PIPELINE_TEST.o ipcio.o ipcbuf.o ipcutil.o ascii_header.o -L/usr/local/cuda/lib64 -lm -lvdifio -lcufft -lgfortran -L$(LIBDIR) $(DEDISP_LIBS)


file2dada: file2dada.c
	$(CC) -g -O2 -I/usr/local/include -I$(INCDIR) -c file2dada.c $(IPCFILES)
	$(F77) -o file2dada file2dada.o ipcio.o ipcbuf.o ipcutil.o -lm -lgfortran -L$(LIBDIR) 


