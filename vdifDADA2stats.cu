/*
 * Code to read VDIF data from a DADA ring buffer, generate filterbanks, calculate and 
 *  record statistics of the power in frequency channels.
 *
 * Uses CUDA FFT to perform FFTs on GPU
 *
 * DADA buffer contains VDIF data for multiple Frames consisting of a header followed by
 *   a data block.
 *   2 VDIF Threads for 2 polarizations, frames alternate pols
 *
 *
 * This version of vdif2fil is meant to be called as a step in a pipeline. 
 *    Speed is a factor.
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "vdifio.h"
#include <cufft.h>
#include <time.h>
#include "dada_def.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include <unistd.h>

//version of this software
#define VERSION "1.0"

#define FRAMELEN 5032      //vdif frame length, bytes
#define HDRLEN 32          //vdif header length, bytes
#define DATALEN 5000       //vdif data length, bytes (5000 samples per frame, 1 sample = 1 byte)
#define FRAMESPERSEC 25600 //# of vdif fames per sec
#define VLITERATE 128e6    //bytes per sec = samples per sec = FRAMESPERSEC*DATALEN
#define ANTENNA "XX"       //not currently used
#define DEVICE 0           //GPU. on furby: 0 = TITAN Black, 1 = GTX 780

#define VLITEBW 64.0       //VLITE bandwidth, MHz
#define VLITELOWFREQ 320.0 //Low end of VLITE band, MHz
#define MUOSLOWFREQ 360.0  //Low end of MUOS-polluted bands, MHz

float MAXTIME=60.0;   //[sec]
float TIMESTATS=1.0;  //[sec]
float HALFCYCLE=0.05; //half of a 10 Hz cal signal cycle [sec]
int NHALFCYC;         //number of cal signal half cycles in MAXTIME
int NCOLSTATBLK;
int NSTATBLKS;

int TWONCHAN; //2 * # of chan across VLITEBW, for FFT 
int NAVE;     //# of chan samples to average for final chan output
int NBUNDLE;  //# of TWONCHAN*NAVE bundles to process on GPU at once
int NLOOP;    //number of bundles in MAXTIME
int NFFT;

vdif_header hdr0, hdr1;      //vdif frame headers, 2 pol
char *tmphdr0,*tmphdr1;      //frame headers read from DADA into tmp buffers
unsigned char *data0,*data1; //vdif frame data, 2 pols
int INDREAD;                 //index of read location in buffers
unsigned char *skip0,*skip1; //vdif frame data to insert for skipped frames
unsigned char *tmpdata0,*tmpdata1; //buffers to hold frame data when skipped frames found

////////////////////
//DADA
////////////////////
ipcio_t dadain; //DADA buffer to read time data from
key_t keyin;    //DADA buffer key to read time data from
#define DADA_DEFAULT_INPUT_KEY  0x0000dada

int FLAGEODADA;    //initialized to 1, set to 0 when EOD in buffer reached
char OUTNAME[256]; //base name for output files
FILE *outflog;     //log file pointer

//functions for handling vdif data
int checkframeskip0();
int checkframeskip1();
int initskip01();
int initframePOL0();
int initframePOL1();
//int checkframeskip();
int initframe01();
int readoneframePOL0();
int readoneframePOL1();
int readtwoframe();
int filltimebufferHOST(unsigned char *timein0, unsigned char *timein1);
int NPARTFILL,NPARTLEFT;
int EXPECTEDFRAMENUM0,EXPECTEDFRAMENUM1;
int NSKIP0,NSKIP1;
int FILL;

//command line functions
void print_usage();
int parser(int argc, char *argv[]);

//GPU functions
__global__ void sumpower(cufftComplex *freqout0, cufftComplex *freqout1, cufftReal *powTOT);
__global__ void avepower(cufftReal *powTOT, cufftReal *powAVE);
__global__ void convertarray(cufftReal *time, unsigned char *utime);
__global__ void subtractbpCAL(cufftReal *FULLFB);
__global__ void calcstats(cufftReal *FULLFB, float *CHANAVE, float *CHANSIGMA, float *CHANOUTLIER);

//GPU constants
__constant__ int CSTRIDEAVE;
__constant__ int CNFREQ;
__constant__ int CNAVE;
__constant__ int CNTOT_SUM;
__constant__ int CNTOT_AVE;
__constant__ int CNTOT_CONVERT;
__constant__ int CNCHANOUT;
__constant__ int CMAXNT;
__constant__ int CNHALFCYC;
__constant__ int CNSTATBLKS;
__constant__ int CNCOLSTATBLK;

//from sigproc.h
int strings_equal (char *string1, char *string2);

//functions for dealing with DADA buffer "iread"
int ipcbuf_get_iread(ipcbuf_t* id){
  return(id->iread);
}

void ipcbuf_force_iread(ipcbuf_t* id){
  if(id->iread != -1){
    fprintf(outflog," iread= %d != -1, setting to -1\n",id->iread);
    id->iread=-1;
  }
  else
    fprintf(outflog," iread= %d, do not need to force\n",id->iread);
}

int loopcnt;

int main(int argc, char *argv[])
{
  int MAXNT;
  int NCHAN;    //# chan across VLITEBW
  int NCHANOUT; //# chan in ouput filterbank file (for possibly excluding MUOS band)
  int NFREQ;    //NCHAN + 1, for Nyquist freq
  int NSKIP;    //# chan to skip when writing = NCHAN-NCHANOUT
  double CHANBW;//BW of a single frequency channel, MHz
  char STATSFILE[256];
  char LOGFILE[256]; //log file name for run-time messages
  double fch1,foffNEG;
  //int nifs=1;
  //int nbits=8*sizeof(float);
  double tstart,tsamp,tBUNDLE;
  int j;
  FILE *outf1;
  cufftComplex *freqout0,*freqout1;
  cufftReal *timein0,*timein1;
  unsigned char *timein0_host,*timein1_host;
  unsigned char *utime0,*utime1;
  cufftReal *powTOT,*powAVE;
  cufftReal *powFULLFB;
  int NTHREAD_STAT,NBLOCK_STAT;
  int NTHREAD_SUB,NBLOCK_SUB;
  int NTHREAD_SUM,NTHREAD_AVE,NTOT_SUM;
  int NBLOCK_SUM,NBLOCK_AVE,NTOT_AVE;
  int NBLOCK_CONVERT,NTHREAD_CONVERT,NTOT_CONVERT;
  int STRIDEAVE;
  size_t MEMSIZESTATBLK;    //size of statistics block
  size_t MEMSIZEFILTERBANK; //size of filterbank to write at a time
  size_t MEMSIZEUTIME;      //size of unsinged char time array to copy to GPU
  size_t MEMSIZETIME;       //size of float time array (conversion done on GPU)
  size_t MEMSIZEPOWER;      //size of power array to copy from GPU
  size_t MEMSIZEFULLFB;     //max size of full filterbank array up to MAXTIME [on GPU]
  cufftHandle plan;
  float *CHANAVE,*CHANSIGMA,*CHANAVE_HOST,*CHANSIGMA_HOST;
  float *CHANOUTLIER,*CHANOUTLIER_HOST;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop,DEVICE);
  
  //check and parse command line
  if(parser(argc,argv)) return(1);
  
  //set STATS and LOG file names and open output files:
  // 1 --- binary file for frequency channel statistics
  // log - file for messages
  //
  sprintf(LOGFILE,"%s.log",OUTNAME);
  outflog = fopen(LOGFILE,"w");
  if (outflog<0) {
    fprintf(stderr,"Unable to open output log file: %s\n",LOGFILE);
    return(2);
  }  
  //File access faster if a preexisting file is deleted before opening
  //if(access(SIGNAME_HEAD,F_OK)!=-1)
  //remove(SIGNAME_HEAD);//file exists, delete it
  sprintf(STATSFILE,"%s.stats",OUTNAME);
  outf1 = fopen(STATSFILE,"wb");
  if (outf1<0) {
    fprintf(stderr,"Unable to open output STATS file: %s\n",STATSFILE);
    return(2);
  }
  fclose(outf1);


  //write device info to log
  fprintf(outflog,"Running vdifDADA2stats version %s\n",VERSION);
  fprintf(outflog,"Running on device: %s\n",prop.name);


  //parameters
  tsamp            = NAVE*TWONCHAN/VLITERATE;  //[sec]
  NSTATBLKS        = (int)(MAXTIME/TIMESTATS);
  NCOLSTATBLK      = (int)(TIMESTATS/tsamp);
  MAXNT            = (int)(MAXTIME/tsamp);
  NHALFCYC         = MAXNT/128;
  if(MAXNT % 128 != 0) NHALFCYC++;
  NLOOP            = MAXNT/NBUNDLE;
  NCHAN            = (int)(0.5*TWONCHAN);
  NFREQ            = NCHAN+1;         //+1 for Nyquist bin
  CHANBW           = VLITEBW/NCHAN;   //channel BW, MHz 
  NCHANOUT         = (int)((MUOSLOWFREQ-VLITELOWFREQ)/CHANBW)+1; //# of channels written to output files
  if((NCHANOUT % 2) == 1) NCHANOUT++; //make NCHANOUT even
  NSKIP            = NCHAN-NCHANOUT;
  NSKIP++;                            //+1 to skip Nyquist freq
  NFFT             = NBUNDLE*NAVE*TWONCHAN; //size of each pol time series to FFT
  foffNEG          = -1.0*CHANBW;           //negative for SIGPROC header because fch1 is highest freq
  fch1             = (VLITELOWFREQ+(NCHANOUT*CHANBW))-(0.5*CHANBW);//center freq of 1st chan in ouput, MHz
  tBUNDLE          = (double)(NAVE*TWONCHAN*NBUNDLE)/VLITERATE;    //amt of time data sent in each "bundle" transfer to GPU [s]
  MEMSIZEFILTERBANK= sizeof(float)*NCHANOUT;
  MEMSIZETIME      = sizeof(cufftReal)*NFFT;
  MEMSIZEUTIME     = sizeof(unsigned char)*NFFT;
  MEMSIZEPOWER     = sizeof(cufftReal)*NFREQ*NBUNDLE; 
  MEMSIZEFULLFB    = MEMSIZEFILTERBANK*MAXNT;
  MEMSIZESTATBLK   = MEMSIZEFILTERBANK*NSTATBLKS;
  //write to log
  fprintf(outflog,"NFREQ   = %d\n",NFREQ);
  fprintf(outflog,"NCHAN   = %d\n",NCHAN);
  fprintf(outflog,"NCHANOUT= %d\n",NCHANOUT);
  fprintf(outflog,"NAVE    = %d\n",NAVE);
  fprintf(outflog,"NBUNDLE = %d\n",NBUNDLE);
  fprintf(outflog,"CHANBW  = %lf [MHz]  (foffNEG= %lf)\n",CHANBW,foffNEG);
  fprintf(outflog,"NLOOP   = %d\n",NLOOP);
  fprintf(outflog,"NFFT    = %d\n",NFFT);
  fprintf(outflog,"fch1    = %lf [MHz]\n",fch1);
  fprintf(outflog,"Will memcpy and filterbank time data in %g [ms] bundles.\n",tBUNDLE*1000.0);

  //device (GPU) parameters
  NBLOCK_STAT    = NCHANOUT;
  NTHREAD_STAT   = NSTATBLKS;
  NBLOCK_SUB     = NCHANOUT; //for BP+cal subtraction
  NTHREAD_SUB    = NHALFCYC; //for BP+cal subtraction
  NTHREAD_SUM    = 512;
  NTHREAD_AVE    = NTHREAD_SUM;
  NTHREAD_CONVERT= NTHREAD_SUM;
  NTOT_SUM       = NFREQ*NAVE*NBUNDLE;
  NTOT_AVE       = NFREQ*NBUNDLE;
  NTOT_CONVERT   = NFFT;
  NBLOCK_SUM     = (int)(NTOT_SUM/NTHREAD_SUM)+1;
  NBLOCK_AVE     = (int)ceil(NTOT_AVE/NTHREAD_AVE)+1;
  NBLOCK_CONVERT = (int)ceil(NTOT_CONVERT/NTHREAD_CONVERT)+1;
  STRIDEAVE      = NAVE*NFREQ;
  fprintf(outflog,"NTOT_SUM= %d   NTOT_AVE= %d  NTOT_CONVERT= %d\n",NTOT_SUM,NTOT_AVE,NTOT_CONVERT);
  fprintf(outflog,"  NTHREAD_SUM= %d  NBLOCK_SUM= %d\n",NTHREAD_SUM,NBLOCK_SUM);
  fprintf(outflog,"  NTHREAD_AVE= %d  NBLOCK_AVE= %d\n",NTHREAD_AVE,NBLOCK_AVE);
  fprintf(outflog,"  NTHREAD_CONVERT= %d  NBLOCK_CONVERT= %d\n",NTHREAD_CONVERT,NBLOCK_CONVERT);
  fprintf(outflog,"  NTHREAD_SUB = %d  NBLOCK_SUB = %d\n",NTHREAD_SUB,NBLOCK_SUB);
  fprintf(outflog,"  NTHREAD_STAT= %d  NBLOCK_STAT= %d\n",NTHREAD_STAT,NBLOCK_STAT);

  //copy to constant memory on device
  cudaMemcpyToSymbol(CNTOT_SUM,&NTOT_SUM,sizeof(int));
  cudaMemcpyToSymbol(CNTOT_AVE,&NTOT_AVE,sizeof(int));
  cudaMemcpyToSymbol(CNTOT_CONVERT,&NTOT_CONVERT,sizeof(int));
  cudaMemcpyToSymbol(CNFREQ,&NFREQ,sizeof(int));
  cudaMemcpyToSymbol(CNAVE,&NAVE,sizeof(int));
  cudaMemcpyToSymbol(CSTRIDEAVE,&STRIDEAVE,sizeof(int));
  cudaMemcpyToSymbol(CNCHANOUT,&NCHANOUT,sizeof(int));
  cudaMemcpyToSymbol(CMAXNT,&MAXNT,sizeof(int));
  cudaMemcpyToSymbol(CNHALFCYC,&NHALFCYC,sizeof(int));
  cudaMemcpyToSymbol(CNCOLSTATBLK,&NCOLSTATBLK,sizeof(int));
  cudaMemcpyToSymbol(CNSTATBLKS,&NSTATBLKS,sizeof(int));

  //DADA ring buffers:
  //connect to existing buffer and open as primary read client
  fprintf(outflog,"DADA buffer at start:\n");
  fprintf(outflog," Connecting to DADA buffer with key: %x ...\n",keyin);
  if(ipcio_connect(&dadain,keyin) < 0){
    fprintf(outflog," Could not connect to DADA buffer!\n");
    return -1;
  }
  ////////////////////////////////////////////////////////
  //HACK to deal with iread param being not -1
  // iread = -1 is required to open as primary read client
  ////////////////////////////////////////////////////////
  ipcbuf_force_iread(((ipcbuf_t *)(&dadain)));
  ////////////////////////////////////////////////////////
  if(ipcio_open(&dadain,'R') < 0){  //'R' = primary read client, 'r' = secondary 
    fprintf(outflog," Could not open for reading DADA buffer!\n");
    ipcio_disconnect(&dadain);
    return -1;
  }  
  fprintf(outflog," nbufs= %lu\n",ipcbuf_get_nbufs((ipcbuf_t *)(&dadain)));
  fprintf(outflog," bufsz= %lu\n",ipcbuf_get_bufsz(((ipcbuf_t *)(&dadain))));
  fprintf(outflog," nfull= %lu\n",ipcbuf_get_nfull(((ipcbuf_t *)(&dadain))));
  fprintf(outflog," nclear= %lu\n",ipcbuf_get_nclear(((ipcbuf_t *)(&dadain))));
  fprintf(outflog," write count= %lu\n",ipcbuf_get_write_count(((ipcbuf_t *)(&dadain))));
  fprintf(outflog," read count = %lu\n",ipcbuf_get_read_count(((ipcbuf_t *)(&dadain))));


  //allocate
  cudaHostAlloc((void **)&timein0_host,MEMSIZEUTIME,cudaHostAllocDefault);
  cudaHostAlloc((void **)&timein1_host,MEMSIZEUTIME,cudaHostAllocDefault);
  cudaHostAlloc((void **)&CHANAVE_HOST,MEMSIZESTATBLK,cudaHostAllocDefault);
  cudaHostAlloc((void **)&CHANSIGMA_HOST,MEMSIZESTATBLK,cudaHostAllocDefault);
  cudaHostAlloc((void **)&CHANOUTLIER_HOST,MEMSIZESTATBLK,cudaHostAllocDefault);
  cudaMalloc((void **)&powFULLFB,MEMSIZEFULLFB);
  cudaMalloc((void **)&powAVE,MEMSIZEPOWER);
  cudaMalloc((void **)&powTOT,MEMSIZEPOWER*NAVE);
  cudaMalloc((void **)&timein0,MEMSIZETIME);
  cudaMalloc((void **)&timein1,MEMSIZETIME);
  cudaMalloc((void **)&utime0,MEMSIZEUTIME);
  cudaMalloc((void **)&utime1,MEMSIZEUTIME);
  cudaMalloc((void **)&freqout0,sizeof(cufftComplex)*NFREQ*NAVE*NBUNDLE);
  cudaMalloc((void **)&freqout1,sizeof(cufftComplex)*NFREQ*NAVE*NBUNDLE);
  cudaMalloc((void **)&CHANAVE,MEMSIZESTATBLK);
  cudaMalloc((void **)&CHANSIGMA,MEMSIZESTATBLK);
  cudaMalloc((void **)&CHANOUTLIER,MEMSIZESTATBLK);

  //tmphdr for ipcio_read, gets converted to hdr
  tmphdr0  = (char *)malloc(HDRLEN*sizeof(char));
  tmphdr1  = (char *)malloc(HDRLEN*sizeof(char));
  //data array for single frame of vdif data, both pols
  data0    = (unsigned char *)malloc(DATALEN*sizeof(unsigned char));
  data1    = (unsigned char *)malloc(DATALEN*sizeof(unsigned char));
  //buffers to hold data to fill-in for skipped frames
  skip0    = (unsigned char *)malloc(DATALEN*sizeof(unsigned char));
  skip1    = (unsigned char *)malloc(DATALEN*sizeof(unsigned char));
  //buffers to hold next good frame of data when skipped frames found
  tmpdata0 = (unsigned char *)malloc(DATALEN*sizeof(unsigned char));
  tmpdata1 = (unsigned char *)malloc(DATALEN*sizeof(unsigned char));
	 	 
  //setup FFT
  fprintf(outflog,"calculating plan...\n");
  if(cufftPlan1d(&plan,TWONCHAN,CUFFT_R2C,NAVE*NBUNDLE) != CUFFT_SUCCESS){
    fprintf(outflog,"Error creating plan!\n");
    return(1);
  }

  //initialize
  NSKIP0     = 0;
  NSKIP1     = 0;
  INDREAD    = 0;
  FLAGEODADA = 1;
  NPARTLEFT  = 0;
  //initialize frame buffers with first vdif frame from DADA buffer
  fprintf(outflog,"initializing pol 0 & 1 single frame buffers...\n");
  if (initframe01() != 0){
    fprintf(outflog,"initframe01 returned in error!\n");
    return(2);
  }
  //initialize skip frame data
  if (initskip01() != 0){
    fprintf(outflog,"initskip01 returned in error!\n");
    return(2);
  }  

  //start time of filterbank
  tstart=getVDIFDMJD(&hdr0,FRAMESPERSEC); //[MJD]

  ///start looping here
  loopcnt=0;
  fprintf(outflog,"begin main loop...\n");
  while(FLAGEODADA){
    //fill time buffer on host with float data from frame buffers
    if(filltimebufferHOST(timein0_host,timein1_host)){ 
      break;
    }
    
    //copy data in host memory to GPU
    cudaMemcpy(utime0,timein0_host,MEMSIZEUTIME,cudaMemcpyHostToDevice);
    cudaMemcpy(utime1,timein1_host,MEMSIZEUTIME,cudaMemcpyHostToDevice);

    //convert unsigned char time arrays to float arrays on GPU
    convertarray<<<NBLOCK_CONVERT,NTHREAD_CONVERT>>>(timein0,utime0);
    convertarray<<<NBLOCK_CONVERT,NTHREAD_CONVERT>>>(timein1,utime1);

    /////generate filterbank
    //FT time data for each pol, in all bundles
    if(cufftExecR2C(plan,timein0,freqout0) != CUFFT_SUCCESS){
      fprintf(stderr,"Error executing plan for pol 0!\n");
      return(1);
    }
    if(cufftExecR2C(plan,timein1,freqout1) != CUFFT_SUCCESS){
      fprintf(stderr,"Error executing plan for pol 1!\n");
      return(1);
    }

    //sum powers in both polarizations before averaging, all bundles
    sumpower<<<NBLOCK_SUM,NTHREAD_SUM>>>(freqout0,freqout1,powTOT);

    //average power, all bundles
    avepower<<<NBLOCK_AVE,NTHREAD_AVE>>>(powTOT,powAVE);

    //copy data in bundles to full filterbank array
    FILL=loopcnt*NCHANOUT*NBUNDLE;
    for(j=0;j<NBUNDLE;j++){
      cudaMemcpy(&powFULLFB[FILL+(j*NCHANOUT)],&powAVE[NSKIP+(j*NFREQ)],MEMSIZEFILTERBANK,cudaMemcpyDeviceToDevice);
    }

    // VLITE data in LSB, do not need to flip band. Highest freq is array index 1
    // Nyquist freq is array index 0
    // only write NCHANOUT, skip MUOS polluted chans and skip Nyquist (included in NSKIP)

    loopcnt++;
    if(loopcnt==NLOOP){
      //subtract BP and cal signal from FULLFB
      subtractbpCAL<<<NBLOCK_SUB,NTHREAD_SUB>>>((cufftReal *)powFULLFB);
      //calc channel statistics
      calcstats<<<NBLOCK_STAT,NTHREAD_STAT>>>((cufftReal *)powFULLFB,CHANAVE,CHANSIGMA,CHANOUTLIER);
      //copy device memory to host
      cudaMemcpy(CHANAVE_HOST,CHANAVE,MEMSIZESTATBLK,cudaMemcpyDeviceToHost);
      cudaMemcpy(CHANSIGMA_HOST,CHANSIGMA,MEMSIZESTATBLK,cudaMemcpyDeviceToHost);
      cudaMemcpy(CHANOUTLIER_HOST,CHANOUTLIER,MEMSIZESTATBLK,cudaMemcpyDeviceToHost);
      //append output file
      outf1 = fopen(STATSFILE,"ab");
      fwrite(&CHANAVE_HOST[0],MEMSIZESTATBLK,1,outf1);
      fwrite(&CHANSIGMA_HOST[0],MEMSIZESTATBLK,1,outf1);
      fwrite(&CHANOUTLIER_HOST[0],MEMSIZESTATBLK,1,outf1);
      fclose(outf1);
      //reset loop counter
      loopcnt=0;
    }

  }//end while loop
  fprintf(outflog,"Looped %d times\n\n",loopcnt);  

  /////////
  //cleanup
  cufftDestroy(plan);

  //close connection to input DADA buffer
  if(ipcio_close(&dadain) < 0){
    fprintf(outflog,"Error closing input DADA buffer!\n");
    return(2);
  }
  if(ipcio_disconnect(&dadain) < 0){
    fprintf(outflog,"Error disconnecting from input DADA buffer!\n");
    return(2);
  }

  //close output log file
  fprintf(outflog,"Processed %f [sec] of data\n",loopcnt*tBUNDLE);
  fprintf(outflog,"\nfinished!\n");
  fclose(outflog);

  return 0;
}//end of main

int initframePOL0(){
  //read frame header
  if(ipcio_read(&dadain,tmphdr0,HDRLEN) != HDRLEN){
    //check if EOD
    FLAGEODADA=!ipcbuf_eod((ipcbuf_t *)(&dadain));//FLAGEODADA=0 means end reached 
    if(!FLAGEODADA){
      fprintf(outflog,"EOD while trying to read header0\n");
      return 1;
    }
    else{
      fprintf(outflog,"ERROR while trying to read header0!!!\n");
      return 2;
    }
  }
  //cast hdr0
  hdr0= *((vdif_header *)(tmphdr0));
  //read frame data -- ipcio_read requires data0 be char *!
  if(ipcio_read(&dadain,(char *)data0,DATALEN) != DATALEN){
    FLAGEODADA=!ipcbuf_eod((ipcbuf_t *)(&dadain));//FLAGEODADA=0 means end reached 
    if(!FLAGEODADA){
      fprintf(outflog,"EOD while trying to read data0\n");
      return(1);
    }
    else{
      fprintf(outflog,"ERROR while trying to read data0!!!\n");
      return(2);
    }
  }
  return 0;
}  

int initframePOL1(){
  //read frame header
  if(ipcio_read(&dadain,tmphdr1,HDRLEN) != HDRLEN){
    //check if EOD
    FLAGEODADA=!ipcbuf_eod((ipcbuf_t *)(&dadain));//FLAGEODADA=0 means end reached 
    if(!FLAGEODADA){
      fprintf(outflog,"EOD while trying to read header1\n");
      return 1;
    }
    else{
      fprintf(outflog,"ERROR while trying to read header1!!!\n");
      return 2;
    }
  }
  //cast hdr0
  hdr1= *((vdif_header *)(tmphdr1));
  //read frame data -- ipcio_read requires data0 be char *!
  if(ipcio_read(&dadain,(char *)data1,DATALEN) != DATALEN){
    FLAGEODADA=!ipcbuf_eod((ipcbuf_t *)(&dadain));//FLAGEODADA=0 means end reached 
    if(!FLAGEODADA){
      fprintf(outflog,"EOD while trying to read data0\n");
      return(1);
    }
    else{
      fprintf(outflog,"ERROR while trying to read data0!!!\n");
      return(2);
    }
  }
  return 0;
}

//check for skipped frames, pol 0
//  check if frame number agrees with expected value
int checkframeskip0(){
  int fnum0;
  fnum0=getVDIFFrameNumber(&hdr0);
  if(fnum0 != EXPECTEDFRAMENUM0){
    //set # skipped frames
    NSKIP0=fnum0-EXPECTEDFRAMENUM0;
    if(NSKIP0<0) NSKIP0+=FRAMESPERSEC;
    fprintf(outflog,"FRAME SKIP 0!!! EXPECTED FRAME NUMBER0= %d, BUT READ %d! (NSKIP0= %d, loopcnt= %d)\n",EXPECTEDFRAMENUM0,fnum0,NSKIP0,loopcnt);
    //copy frame data to tmpdata buffer
    memcpy(&tmpdata0[0],&data0[0],DATALEN);
    //copy skip frame values to data buffer
    memcpy(&data0[0],&skip0[0],DATALEN);
    //update expected frame number
    //EXPECTEDFRAMENUM0=(fnum0+1) % FRAMESPERSEC;    
    //return NSKIP0;
  }
  //update expected frame number
  EXPECTEDFRAMENUM0=(fnum0+1) % FRAMESPERSEC;
  return 0;
}

//check for skipped frames, pol 1
//  check if frame number agrees with expected value
int checkframeskip1(){
  int fnum1;
  fnum1=getVDIFFrameNumber(&hdr1);
  if(fnum1 != EXPECTEDFRAMENUM1){
    //set # skipped frames
    NSKIP1=fnum1-EXPECTEDFRAMENUM1;
    if(NSKIP1<0) NSKIP1+=FRAMESPERSEC;
    fprintf(outflog,"FRAME SKIP 1!!! EXPECTED FRAME NUMBER1= %d, BUT READ %d! (NSKIP1= %d, loopcnt= %d)\n",EXPECTEDFRAMENUM1,fnum1,NSKIP1,loopcnt);
    //copy frame data to tmpdata buffer
    memcpy(&tmpdata1[0],&data1[0],DATALEN);
    //copy skip frame values to data buffer
    memcpy(&data1[0],&skip1[0],DATALEN);
    //update expected frame number
    //EXPECTEDFRAMENUM1=(fnum1+1) % FRAMESPERSEC;    
    //return NSKIP1;
  }
  //update expected frame number
  EXPECTEDFRAMENUM1=(fnum1+1) % FRAMESPERSEC;
  return 0;
}

//read one VDIF frame from DADA for pol 0
int readoneframePOL0(){
  //deal with skipped frames, if necessary
  if(NSKIP0){
    NSKIP0--;
    //have accounted for all skipped frames?
    //if so, copy tmpdata to data & return
    if(NSKIP0==0){
      memcpy(&data0[0],&tmpdata0[0],DATALEN);   
      return 0;
    }
    //if not, still skipped frames. skip0 already in data0 so just return
    return 0;
  }
  else{//no skips, proceed normally
    //read frame header
    if(ipcio_read(&dadain,tmphdr0,HDRLEN) != HDRLEN){
      //check if EOD
      FLAGEODADA=!ipcbuf_eod((ipcbuf_t *)(&dadain));//FLAGEODADA=0 means end reached 
      if(!FLAGEODADA){
	fprintf(outflog,"EOD while trying to read header0!\n");
	return(1);
      }
      else{
	fprintf(outflog,"ERROR while trying to read header0!!!\n");
	return(2);
      }
    }
    //cast hdr0
    hdr0= *((vdif_header *)(tmphdr0));
    //read frame data -- ipcio_read requires data0 be char *!
    if(ipcio_read(&dadain,(char *)data0,DATALEN) != DATALEN){
      FLAGEODADA=!ipcbuf_eod((ipcbuf_t *)(&dadain));//FLAGEODADA=0 means end reached 
      if(!FLAGEODADA){
	fprintf(outflog,"EOD while trying to read data0\n");
	return(1);
      }
      else{
	fprintf(outflog,"ERROR while trying to read data0!!!\n");
	return(2);
      }
    }
    //check for frame skips
    checkframeskip0();
    //if skipped frame, checkframeskip copies data0 into tmpdata0 
    //  and skip0 into data0, so skip values will be returned
    return (0);
  }
}

//read one VDIF frame from DADA for pol 1
int readoneframePOL1(){
  //deal with skipped frames, if necessary
  if(NSKIP1){
    NSKIP1--;
    //have accounted for all skipped frames?
    //if so, copy tmpdata to data & return
    if(NSKIP1==0){
      memcpy(&data1[0],&tmpdata1[0],DATALEN);   
      return 0;
    }
    //if not, still skipped frames. skip1 already in data1 so just return
    return 0;
  }
  else{//no skips, proceed normally
    //read frame header
    if(ipcio_read(&dadain,tmphdr1,HDRLEN) != HDRLEN){
      //check if EOD
      FLAGEODADA=!ipcbuf_eod((ipcbuf_t *)(&dadain));//FLAGEODADA=0 means end reached 
      if(!FLAGEODADA){
	fprintf(outflog,"EOD while trying to read header1!\n");
	return(1);
      }
      else{
	fprintf(outflog,"ERROR while trying to read header1!!!\n");
	return(2);
      }
    }
    //cast hdr1
    hdr1= *((vdif_header *)(tmphdr1));
    //read frame data
    if(ipcio_read(&dadain,(char *)data1,DATALEN) != DATALEN){
      FLAGEODADA=!ipcbuf_eod((ipcbuf_t *)(&dadain));//FLAGEODADA=0 means end reached 
      if(!FLAGEODADA){
	fprintf(outflog,"EOD while trying to read data1\n");
	return(1);
      }
      else{
	fprintf(outflog,"ERROR while trying to read data1!!!\n");
	return(2);
      }
    }
    //check for frame skips
    checkframeskip1();
    //if skipped frame, checkframeskip copies data1 into tmpdata1 
    //  and skip1 into data1, so skip values will be returned
    return (0);
  }
}

//read a frame for each polarization from file
int readtwoframe(){
  int flag;
  //pol 0
  flag=readoneframePOL0();
  if(flag) return(flag);
  //pol 1
  flag=readoneframePOL1();
  if(flag) return(flag);
  return(0);
}

//initialize skip buffers 0 & 1 with values to fill-in for skipped frames
int initskip01()
{
  int i;
  for(i=0;i<DATALEN;i++){
    skip0[i]=(unsigned char)(0);
    skip1[i]=(unsigned char)(0);    
  }
  return 0;
}

//initialize frame buffers 0 & 1 with a VDIF frame from DADA buffer
// skip initial frame if not pol 0
// advance to start of cal signal
int initframe01()
{
  int id0,id1,fnum0,fnum1,ncal=0,flag;
  //read pol 0
  flag=initframePOL0();
  if(flag) return(flag);    
  //check pol 0, if not then skip forward 1 frame
  id0=getVDIFThreadID(&hdr0);
  if(id0 != 0){
    fprintf(outflog,"VDIF Threads misaligned, skipping forward 1 frame\n");
    flag=initframePOL0();
    if(flag) return(flag);    
  }
  //double check
  id0=getVDIFThreadID(&hdr0);
  if(id0 != 0){
    fprintf(outflog,"ERROR, initframe01()! VDIF Thread ID= %d, != 0. Abort!\n",id0);
    return(1);
  }

  //read pol 1
  flag=initframePOL1();
  if(flag) return(flag);    
  //check pol 1
  id1=getVDIFThreadID(&hdr1);
  if(id1 != 1){
    fprintf(outflog,"VDIF Thread ID= %d, != 1. Abort!\n",id1);
    return(1);
  }

  //check agreement of pol frame numbers
  fnum0=getVDIFFrameNumber(&hdr0);
  fnum1=getVDIFFrameNumber(&hdr1);
  if(fnum0 != fnum1){
    fprintf(outflog,"ERROR! polarization frame numbers do not agree! 0: %d  !=  1: %d\n",fnum0,fnum1);
    return(1);
  }

  //find start of cal signal (10 Hz)
  while(1){
    if(fmod((float)fnum0,(float)2560) < 1e-4){//this frame is start of cal signal (OFF), exit loop
      fprintf(outflog,"Skipped %d frames to begin at noise cal cycle\n",ncal);
      break; 
    }
    //otherwise, read next pair of frames
    flag=initframePOL0();
    if(flag) return(flag);
    flag=initframePOL1();
    if(flag) return(flag);
    fnum0=getVDIFFrameNumber(&hdr0);
    ncal++;
  }

  //re-check agreement of pol frame numbers
  fnum1=getVDIFFrameNumber(&hdr1);
  if(fnum0 != fnum1){
    fprintf(outflog,"ERROR! polarization VDIF frame numbers do not agree! 0: %d  !=  1: %d\n",fnum0,fnum1);
    return(1);
  }

  //set expected frame numbers for next read (to check for frame skips)
  EXPECTEDFRAMENUM0=(fnum0+1) % FRAMESPERSEC;
  EXPECTEDFRAMENUM1=(fnum1+1) % FRAMESPERSEC; 

  //fprintf(outflog,"INITIALIZING EXPECTED FRAME NUMBER0= %d\n",EXPECTEDFRAMENUM0);
  //fprintf(outflog,"INITIALIZING EXPECTED FRAME NUMBER1= %d\n",EXPECTEDFRAMENUM1);  
  return 0;
}

//fill host time buffer from DADA via single frame buffers
int filltimebufferHOST(unsigned char *timein0, unsigned char *timein1)
{
  int q,NWHOLE,NTOT,IFILL=0;
  if(NPARTLEFT){
    memcpy(&timein0[IFILL],&data0[NPARTFILL],NPARTLEFT);
    memcpy(&timein1[IFILL],&data1[NPARTFILL],NPARTLEFT);
    IFILL+=NPARTLEFT;
    //refill frame buffers
    if(readtwoframe()) return(1);
  }
  NTOT=NFFT-NPARTLEFT;
  NWHOLE=NTOT/DATALEN;
  NPARTFILL=NTOT % DATALEN;
  NPARTLEFT=DATALEN-NPARTFILL;

  for(q=0;q<NWHOLE;q++){
    memcpy(&timein0[IFILL],&data0[0],DATALEN);
    memcpy(&timein1[IFILL],&data1[0],DATALEN);
    IFILL+=DATALEN;    
    //refill frame buffers
    if(readtwoframe()) return(1);
  }

  if(NPARTFILL){
    memcpy(&timein0[IFILL],&data0[0],NPARTFILL);
    memcpy(&timein1[IFILL],&data1[0],NPARTFILL);
    IFILL+=NPARTFILL;
  }  

  //returns with full or partially filled data0[] & data1[]  
  return(0);
}

//////////////////////
void print_usage(){
  fprintf(stderr,"***************************************************************************\n");
  fprintf(stderr,"This program reads time series data from a DADA buffer with VDIF format data,\n");
  fprintf(stderr,"  generates a filterbank and records statistics of the channel power in 1 second intervals.\n\n");
  fprintf(stderr,"  Usage:\n./vdifDADA2stats:\n");
  fprintf(stderr,"\t             -h:      print this help message.\n");
  fprintf(stderr,"\n");
  fprintf(stderr,"  DEFAULTS:\n");
  fprintf(stderr,"\t             DADA_INPUT_KEY             = %x\n",DADA_DEFAULT_INPUT_KEY);
  fprintf(stderr,"\t             NAME OF OUTPUT FILE        = vdifDADA2stats.txt\n");
  fprintf(stderr,"\t             NUM OF CHANNELS TO AVERAGE = 4\n");
  fprintf(stderr,"\t             NUM OF CHANNELS            = 6250\n");
  fprintf(stderr,"\t             NUM OF BUNDLES             = 32\n");
  fprintf(stderr,"\n");
  return;
}

int parser(int argc, char *argv[]){
  int i=1;

  TWONCHAN=6250*2;
  NAVE=4;
  NBUNDLE=32;
  keyin = DADA_DEFAULT_INPUT_KEY;
  strcpy(OUTNAME,"vdifDADA2stats_");

  while(i<argc){
    if(strings_equal(argv[i],"-h")){
      print_usage();   
      return(3);
    }
    else{
      //unknown argument
      fprintf(stderr,"Error! Unknown argument: %s\n\n",argv[i]);
      print_usage();   
      return(3);
    }
  }

  return(0);
}

//sum polarization powers before averaging
__global__ void sumpower(cufftComplex *freqout0, cufftComplex *freqout1, cufftReal *powTOT){
  int i;
  i=threadIdx.x+(blockIdx.x*blockDim.x);
  if(i < CNTOT_SUM)
    powTOT[i]=((freqout0[i].x*freqout0[i].x)+(freqout0[i].y*freqout0[i].y)+(freqout1[i].x*freqout1[i].x)+(freqout1[i].y*freqout1[i].y))/(cufftReal)(CNFREQ);
  return;
}

//average powers for final filterbank power
__global__ void avepower(cufftReal *powTOT, cufftReal *powAVE){
  int AVE_ind,TOT_ind,i;
  float tmp;
  AVE_ind=threadIdx.x+(blockIdx.x*blockDim.x);
  if(AVE_ind < CNTOT_AVE){
    TOT_ind= ((int)(AVE_ind/CNFREQ)*CSTRIDEAVE) + (AVE_ind % CNFREQ);
    tmp=0.0;
    for(i=0;i<CNAVE;i++)
      tmp+=powTOT[TOT_ind+(i*CNFREQ)];
    powAVE[AVE_ind]=tmp/CNAVE;
  }
  return;
}

//convert unsigned char time array to float
__global__ void convertarray(cufftReal *time, unsigned char *utime){
  int i;
  i=threadIdx.x+(blockIdx.x*blockDim.x);  
  if(i < CNTOT_CONVERT)
    time[i]=(cufftReal)(utime[i]);
  return;
}

__global__ void subtractbpCAL(cufftReal *FULLFB){
  int i,j,istart;
  int row,colmin,colmax,npts;
  float sum,sum2,ave,sig,HI,LO;
  row=(blockIdx.x);
  if((threadIdx.x<CNHALFCYC) && (row < CNCHANOUT)){
    colmin=128*threadIdx.x;
    colmax=colmin+128;
    if(colmax>CMAXNT)
      colmax=CMAXNT;
    istart=row+(colmin*CNCHANOUT);
    npts=colmax-colmin;
    ave=0.0;
    for(j=0;j<npts;j++)
      ave+=FULLFB[istart+(j*CNCHANOUT)];
    ave/=npts;
    sum=0.0;sum2=0.0;
    for(j=0;j<npts;j++){
      sum2+=((FULLFB[istart+(j*CNCHANOUT)]-ave)*(FULLFB[istart+(j*CNCHANOUT)]-ave));
      sum+=(FULLFB[istart+(j*CNCHANOUT)]-ave);
    }
    sig=(sum2-(sum*sum)/npts)/(npts-1);
    
    //recalc ave w/o extreme points
    HI=ave+(3.0*sig);
    LO=ave-(3.0*sig);
    ave=0.0; i=0;
    for(j=0;j<npts;j++)
      if(FULLFB[istart+(j*CNCHANOUT)]<HI && FULLFB[istart+(j*CNCHANOUT)]>LO){
	ave+=FULLFB[istart+(j*CNCHANOUT)];
	i++;
      }
    if(i==0) i=1;
    ave/=i;

    for(j=0;j<npts;j++)
      FULLFB[istart+(j*CNCHANOUT)]-=ave;
  }
  return;
}

__global__ void calcstats(cufftReal *FULLFB, float *CHANAVE, float *CHANSIGMA, float *CHANOUTLIER)
{
  int i,j,row,colmin,colmax,npts,istart;
  float sum,sum2,ave,sig,HI;
  row=(blockIdx.x);
  if((threadIdx.x<CNSTATBLKS) && (row < CNCHANOUT)){
    colmin=CNCOLSTATBLK*threadIdx.x;
    colmax=colmin+CNCOLSTATBLK;
    if(colmax>CMAXNT)
      colmax=CMAXNT;
    istart=row+(colmin*CNCHANOUT);
    npts=colmax-colmin;
    ave=0.0;
    for(j=0;j<npts;j++)
      ave+=FULLFB[istart+(j*CNCHANOUT)];
    ave/=npts;
    sum=0.0;sum2=0.0;
    for(j=0;j<npts;j++){
      sum2+=((FULLFB[istart+(j*CNCHANOUT)]-ave)*(FULLFB[istart+(j*CNCHANOUT)]-ave));
      sum+=(FULLFB[istart+(j*CNCHANOUT)]-ave);
    }
    sig=(sum2-(sum*sum)/npts)/(npts-1);
    
    //count high outliers 
    HI=ave+(5.0*sig);
    i=0;
    for(j=0;j<npts;j++)
      if(FULLFB[istart+(j*CNCHANOUT)]>HI){
	i++;
      }

    CHANAVE[row+(threadIdx.x*CNCHANOUT)]=ave;
    CHANSIGMA[row+(threadIdx.x*CNCHANOUT)]=sig;
    CHANOUTLIER[row+(threadIdx.x*CNCHANOUT)]=(i/npts);
  }
  return;
}


int strings_equal (char *string1, char *string2)
{
  if (!strcmp(string1,string2)) return 1;
  else return 0;
}
