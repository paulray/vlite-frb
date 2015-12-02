LPGPLOT = -L/usr/local/pgplot -lpgplot -lcpgplot -L/usr/lib -lpng -lz -lX11
LFFTW = -lfftw3f

INCDIR = /home/vlite-master/frb-search/psrdada/src
LIBDIR = /home/vlite-master/frb-search/psrdada/src
IPCFILES = $(INCDIR)/ipcio.c $(INCDIR)/ipcbuf.c $(INCDIR)/ipcutil.c $(INCDIR)/ascii_header.c

#DEDISP_CFLAGS = -I/usr/local/src/dedisp-read-only/include
#DEDISP_LIBS = -L/usr/local/src/dedisp-read-only/lib -ldedisp

F77 = gfortran -std=legacy -ffixed-line-length-none 
CC = gcc
NVCC = nvcc

all: 


vdifDADA2timeseries: vdifDADA2timeseries.cu
	$(NVCC) -g -O2 -I/usr/cuda -I/usr/include -I$(INCDIR) -c vdifDADA2timeseries.cu $(IPCFILES)
	$(NVCC) -o vdifDADA2timeseries vdifDADA2timeseries.o ipcio.o ipcbuf.o ipcutil.o ascii_header.o -L/usr/cuda/lib64 -L/home/vlite-master/frb-src/vdifio/lib -lm -lvdifio -lcufft -lgfortran -L$(LIBDIR)
