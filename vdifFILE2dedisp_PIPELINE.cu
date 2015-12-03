/*
 * Code to read VDIF data from a DADA ring buffer and write out as FILTERBANK file in
 *  SIGPROC format.
 *
 * Uses CUDA FFT to perform FFTs on GPU
 *
 * DADA buffer contains VDIF data for multiple Frames consisting of a header followed by
 *   a data block.
 *   2 polarizations, frames alternate pols
 *
 *
 * This version of vdif2fil is meant to be called as a step in a pipeline. 
 *    Speed is a factor, so no plotting.
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
#include <unistd.h>
#include <dedisp.h>

//version of this software
#define VERSION "1.0"

#define FRAMELEN 5032   //vdif frame length, bytes
#define HDRLEN 32       //vdif header length, bytes
#define DATALEN 5000    //vdif data length, bytes (5000 samples per frame, 1 sample = 1 byte)
#define FRAMESPERSEC 25600 //# of vdif fames per sec
#define VLITERATE 128e6 //bytes per sec = samples per sec = FRAMESPERSEC*DATALEN
#define ANTENNA "XX"
#define DEVICE 0        //GPU. on furby: 0 = TITAN Black, 1 = GTX 780
#define STRLEN 256

#define VLITEBW 64.0          //VLITE bandwidth, MHz
#define VLITELOWFREQ 320.0    //Low end of VLITE band, MHz
//const float VLITEFREQCENTER=VLITELOWFREQ+(0.5*VLITEBW); //center frequency, MHz
//const float VLITEHIGHFREQ=VLITELOWFREQ+VLITEBW;         //upper frequency, MHz
#define MUOSLOWFREQ 360.0     //Low end of MUOS-polluted bands, MHz
//#define MUOSLOWFREQ 384.0     //set this to write full 320-384 band to output .fil files

//float MAXTIME=30.0;//max amt of time-sampled voltage data to collect before dedispersing [sec]
float MAXTIME=70.0;
float HALFCYCLE=0.05; //half of a 10 Hz cal signal cycle [sec]
int NHALFCYC;         //number of cal signal half cycles in MAXTIME


int TWONCHAN; //2 * NCHAN, for FFT 
int NAVE;     //# of chan samples to average for final chan output
int NBUNDLE;  //# of TWONCHAN*NAVE bundles to process on GPU at once
int NLOOP;    //TWONCHAN*NAVE*NBUNDLE

vdif_header   hdr0, hdr1;    //vdif frame headers, 2 pol
unsigned char *data0,*data1; //vdif frame data, 2 pols
int           INDREAD;       //index of read location in buffers
unsigned char *skip0,*skip1; //vdif frame data to insert for skipped frames
unsigned char *tmpdata0,*tmpdata1; //buffers to hold frame data when skipped frames found

FILE *ft;             //pointer to file with VDIF data
int  FLAGEOF;         //initialized to 1, set to 0 when EOF reached
char VDIFNAME[STRLEN];//name of input file with VDIF data
char OUTNAME[STRLEN]; //base name for output files
FILE *outflog;        //log file pointer

//functions for handling vdif data
int checkframeskip0();
int checkframeskip1();
int initskip01();
int initframePOL0();
int initframePOL1();
int initframe01();
int readoneframePOL0();
int readoneframePOL1();
int readtwoframe();
int filltimebufferHOST(unsigned char *timein0, unsigned char *timein1);
int NPARTFILL,NPARTLEFT;
int EXPECTEDFRAMENUM0,EXPECTEDFRAMENUM1;
int NSKIP0,NSKIP1;

//command line functions
void print_usage();
int parser(int argc, char *argv[]);

//output functions, modified from sigproc.h
void send_coords(double raj, double dej, double az, double za, FILE *output);
void send_double (const char *name, double double_precision, FILE *output);
void send_float(const char *name,float floating_point, FILE *output);
void send_int(const char *name, int integer, FILE *output);
void send_long(const char *name, long integer, FILE *output);
void send_string(const char *string, FILE *output);
int strings_equal (const char *string1, char *string2);

//GPU functions
__global__ void sumpower(cufftComplex *freqout0, cufftComplex *freqout1, cufftReal *powTOT);
__global__ void avepower(cufftReal *powTOT, cufftReal *powAVE);
__global__ void convertarray(cufftReal *time, unsigned char *utime);
__global__ void convertarraybyte(dedisp_byte *powbyte, cufftReal *pow);
__global__ void calcstats(dedisp_float *outDD, int *FRB, float *ave, float *sigma, float *peak);
__global__ void subtractbp(cufftReal *FULLFB);
__global__ void subtractbpCAL(cufftReal *FULLFB);
__global__ void subtractZERO(cufftReal *FULLFB);
__global__ void dedisperse(cufftReal *FULLFB, dedisp_float *outDD, int *OFFSET);

//GPU constants
__constant__ int CSTRIDEAVE;
__constant__ int CNFREQ;
__constant__ int CNAVE;
__constant__ int CNTOT_SUM;
__constant__ int CNTOT_AVE;
__constant__ int CNTOT_CONVERT;
__constant__ int CNTIME_OUTDD;
__constant__ int CNDM;
__constant__ int CNCHANOUT;
__constant__ int CMAXNT;
__constant__ int CNHALFCYC;

static void HandleError(cudaError_t err){
  if(err!=cudaSuccess){
    printf("CUDA ERROR! %s\n",cudaGetErrorString(err));
    exit(0);
  }
}

#define HANDLE_ERROR(err) (HandleError(err))
#define HANDLE_NULL(a) (if (a==NULL){printf("Host memory failure!\n")})

int loopcnt;

int main(int argc, char *argv[])
{
  cudaEvent_t startC,stopC;
  cudaEventCreate(&startC);
  cudaEventRecord(startC,0);
  cudaEventCreate(&stopC);

  int loopcntTOT=0;
  //int FILLFLAG=0;
  int NCHAN;    //# chan across VLITEBW
  int NCHANOUT; //# chan in ouput filterbank file (excluding MUOS band)
  int NFREQ;    //NCHAN + 1, for Nyquist freq
  int NSKIP;    //# chan to skip when writing = NCHAN-NCHANOUT
  double CHANBW;//BW of a single frequency channel, MHz
  char LOGFILE[STRLEN]; //log file name for run-time messages
  char SIGNAME_HEAD[STRLEN],SIGNAME_NOHEADFIL[STRLEN]; //output SIGPROC file names
  double fch1,foffNEG;
  int nifs=1;
  int nbits=8*sizeof(float);
  double tstart,tsamp,tBUNDLE,tchan;
  int j,FILL,i;
  loopcnt=0;
  FILE *outf1,*outf2,*outf3;
  double tFT,tcptohost,tcptodevice,tGPU,tfill,tsum,tave,twrite;
  tFT=tcptohost=tcptodevice=tGPU=tfill=tsum=tave=twrite=0.0;

  int *hostOFFSET,*OFFSET;
  double dt;

  FILE *fin;
  char fname[STRLEN],line[STRLEN];
  cudaEvent_t start,stop;
  cudaEvent_t start1,stop1;
  float time,timems;
  cufftComplex  *freqout0,*freqout1;
  cufftReal     *timein0;
  unsigned char *timein0_host;
  cufftReal     *timein1;
  unsigned char *timein1_host;
  cufftReal     *powTOT,*powAVE,*powAVE_host,*hostFULLFB;
  dedisp_byte *powAVEbyte;
  //cufftReal     *powFULLFB;
  //dedisp_float *powFULLFB;
  dedisp_byte *powFULLFB;
  dedisp_byte *powLEFTOVER;
  unsigned char *utime0,*utime1;
  int NTHREAD_SUB,NBLOCK_SUB;
  int NTHREAD_STAT,NBLOCK_STAT;
  int NTHREAD_SUM,NTHREAD_AVE,NTOT_SUM;
  int NBLOCK_SUM,NBLOCK_AVE,NTOT_AVE;
  int NBLOCK_CONVERT,NTHREAD_CONVERT,NTOT_CONVERT;
  int NBLOCK_DD,NTHREAD_DD;
  int STRIDEAVE;
  size_t MEMSIZEOFFSET;     //size of offset array
  size_t MEMSIZEFILTERBANK; //size of single column of freq channel filterbank data 
  size_t MEMSIZEUTIME;      //size of unsinged char time array to copy to GPU
  size_t MEMSIZETIME;       //size of float time array (conversion done on GPU)
  size_t MEMSIZEPOWER;      //size of power array to copy from GPU
  size_t MEMSIZEFULLFB;     //max size of full filterbank array up to MAXTIME [on GPU]
  size_t MEMSIZELEFTOVER;   //size of end of FB array that must be copied to beginning for next iteration
  int MAXNT;                //max number of time samples per freq chan = MAXTIME/tchan 
  int NTIME_OUTDD;
  int   *FRB,*FRB_host;
  float *ave,*ave_host;
  float *sigma, *sigma_host;
  float *peak,*peak_host;
  dedisp_plan  planDD;
  dedisp_float dm_start,dm_end,pulse_width,dm_tol;
  //dedisp_byte *outDD;
  dedisp_float *outDD;
  dedisp_float *host_outDD;
  dedisp_size  dm_count,max_delay;
  dedisp_size  out_nbits = 32;
  dedisp_size  in_nbits  = 32;
  dedisp_size NDM101;
  dedisp_error err;
  float *DM101_list;
  cufftHandle plan;
  cudaDeviceProp prop;
  cudaError_t cudaerr;
  int device_idx=DEVICE;
  cudaGetDeviceProperties(&prop,device_idx);
  if((err=dedisp_set_device(device_idx)) != DEDISP_NO_ERROR){
    printf("Error setting dedisp device!\n");
    printf("%s\n",dedisp_get_error_string(err));
    return 2;
  }
  printf("%s:\n",prop.name);
  printf("maxThreadsPerBlock= %d\n",prop.maxThreadsPerBlock);
  printf("maxThreadsDim[0]= %d\n",prop.maxThreadsDim[0]);
  printf("maxThreadsDim[1]= %d\n",prop.maxThreadsDim[1]);
  printf("maxThreadsDim[2]= %d\n",prop.maxThreadsDim[2]);
  printf("maxGridSize[0]= %d\n",prop.maxGridSize[0]);
  printf("maxGridSize[1]= %d\n",prop.maxGridSize[1]);
  printf("maxGridSize[2]= %d\n",prop.maxGridSize[2]);

  //create and start timing
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  //cudaEventCreate(&start1);
  //cudaEventCreate(&stop1);
  cudaEventRecord(start,0);


  //check and parse command line
  if (argc < 2) {
    print_usage();
    return(1);
  }
  if(parser(argc,argv)) return(1);


  //set SIGPROC and log file names and open output files:
  // 1 --- SIGPROC header only
  // 2 --- SIGPROC filterbank file, no header
  // log - file for messages
  //
  sprintf(SIGNAME_HEAD,"%s.head",OUTNAME);
  sprintf(SIGNAME_NOHEADFIL,"%s.nohead.fil",OUTNAME);
  sprintf(LOGFILE,"%s.log",OUTNAME);
  outflog = fopen(LOGFILE,"w");
  if (outflog<0) {
    fprintf(stderr,"Unable to open output log file: %s\n",LOGFILE);
    return(2);
  }  
  outf1 = fopen(SIGNAME_HEAD,"wb");
  if (outf1<0) {
    fprintf(stderr,"Unable to open output SIGPROC header file: %s\n",SIGNAME_HEAD);
    return(2);
  }
  //File access is faster if a preexisting file is deleted before opening
  if(access(SIGNAME_NOHEADFIL,F_OK)!=-1)
    remove(SIGNAME_NOHEADFIL);//file exists, delete it
  //
  outf2 = fopen(SIGNAME_NOHEADFIL,"wb");
  if (outf2<0) {
    fprintf(stderr,"Unable to open output SIGPROC headerless filterbank file: %s\n",SIGNAME_NOHEADFIL);
    return(2);
  }
  //////////////////
  if(access("OUTDD.bin",F_OK)!=-1)
    remove("OUTDD.bin");//file exists, delete it
  outf3 = fopen("OUTDD.bin","wb");
  if (outf3<0) {
    fprintf(stderr,"Unable to open output for dedispersion debugging\n");
    return(2);
  }
  ///////////////////

  fprintf(outflog,"Running vdifFILE2fil_PIPELINE version %s\n",VERSION);
  fprintf(outflog,"Running on device: %s\n",prop.name);


  //parameters
  NCHAN  = (int)(0.5*TWONCHAN);
  NFREQ  = NCHAN+1;        //+1 for Nyquist bin
  CHANBW = VLITEBW/NCHAN;  //channel BW, MHz 
  NCHANOUT = (int)((MUOSLOWFREQ-VLITELOWFREQ)/CHANBW)+1; //# of channels written to output files
  if((NCHANOUT % 2) == 1) NCHANOUT++; //make NCHANOUT even
  NSKIP  = (NCHAN-NCHANOUT)+1;    //+1 to skip Nyquist freq
  NLOOP  = NBUNDLE*NAVE*TWONCHAN; //size of each pol time series to FFT
  foffNEG= -1.0*CHANBW; //negative for SIGPROC header because fch1 is highest freq
  fch1   = (VLITELOWFREQ+(NCHANOUT*CHANBW))-(0.5*CHANBW);//center freq of 1st chan in ouput, MHz
  tchan  = (double)(NAVE*TWONCHAN)/VLITERATE;//time between power values of a frequency channel [s]
  tBUNDLE= (double)(NAVE*TWONCHAN*NBUNDLE)/VLITERATE;//amt of time data sent in each transfer to GPU (each "bundle") [s]
  MAXNT  = (int)(MAXTIME/tchan);
  NHALFCYC = MAXNT/128;
  if(MAXNT % 128 != 0) NHALFCYC++;
  MEMSIZEFILTERBANK = sizeof(float)*NCHANOUT;//size of a single column of freq channels (exlcuding MUOS)
  MEMSIZETIME       = sizeof(cufftReal)*NLOOP;
  MEMSIZEUTIME      = sizeof(unsigned char)*NLOOP;
  MEMSIZEPOWER      = sizeof(cufftReal)*NFREQ*NBUNDLE; 
  //MEMSIZEFULLFB=MEMSIZEFILTERBANK*MAXNT;//size of full FB up to MAXTIME
  MEMSIZEFULLFB     = MAXNT*NCHANOUT*(in_nbits/8);
  //write to log
  fprintf(outflog,"NFREQ= %d\n",NFREQ);
  fprintf(outflog,"NCHAN= %d\n",NCHAN);
  fprintf(outflog,"NCHANOUT= %d\n",NCHANOUT);
  fprintf(outflog,"NAVE= %d\n",NAVE);
  fprintf(outflog,"NBUNDLE= %d\n",NBUNDLE);
  fprintf(outflog,"CHANBW= %lf [MHz]  (foffNEG= %lf)\n",CHANBW,foffNEG);
  fprintf(outflog,"fch1= %lf [MHz]\n",fch1);
  fprintf(outflog,"Will memcpy and filterbank time data in %g [ms] bundles.\n",tBUNDLE*1000.0);
  fprintf(outflog,"MAXNT= %d\n",MAXNT);
  fprintf(outflog,"NHALFCYC= %d\n",NHALFCYC);

  //device (GPU) parameters
  NBLOCK_SUB      = NCHANOUT; //for BP+cal subtraction
  NTHREAD_SUB     = NHALFCYC; //for BP+cal subtraction
  NTHREAD_SUM     = 512;
  NTHREAD_AVE     = NTHREAD_SUM;
  NTHREAD_CONVERT = NTHREAD_SUM;
  NTOT_SUM        = NFREQ*NAVE*NBUNDLE;
  NTOT_AVE        = NFREQ*NBUNDLE;
  NTOT_CONVERT    = NLOOP;
  NBLOCK_SUM      = (int)(NTOT_SUM/NTHREAD_SUM)+1;
  NBLOCK_AVE      = (int)ceil(NTOT_AVE/NTHREAD_AVE)+1;
  NBLOCK_CONVERT  = (int)ceil(NTOT_CONVERT/NTHREAD_CONVERT)+1;
  STRIDEAVE       = NAVE*NFREQ;
  fprintf(outflog,"NTOT_SUM= %d   NTOT_AVE= %d  NTOT_CONVERT= %d\n",NTOT_SUM,NTOT_AVE,NTOT_CONVERT);
  fprintf(outflog,"  NTHREAD_SUM= %d  NBLOCK_SUM= %d\n",NTHREAD_SUM,NBLOCK_SUM,NTHREAD_AVE,NBLOCK_AVE);
  fprintf(outflog,"  NTHREAD_AVE= %d  NBLOCK_AVE= %d\n",NTHREAD_SUM,NBLOCK_SUM,NTHREAD_AVE,NBLOCK_AVE);
  fprintf(outflog,"  NTHREAD_CONVERT= %d  NBLOCK_CONVERT= %d\n",NTHREAD_CONVERT,NBLOCK_CONVERT);
  fprintf(outflog,"NTHREAD_SUB= %d  NBLOCK_SUB= %d\n",NTHREAD_SUB,NBLOCK_SUB);

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

  //open input file with VDIF data
  ft = fopen(VDIFNAME,"rb");
  if (ft<0) {
    fprintf(stderr,"Unable to open input VDIF file: %s\n",VDIFNAME);
    return 2;
  }


  //count number of DM values in file
  NDM101=0;
  sprintf(fname,"DM101.txt");
  fprintf(outflog,"counting number of DM trials in %s...\n",fname);
  if((fin=fopen(fname,"r"))==NULL){
    fprintf(outflog,"ERROR opening file %f!\n",fname);
    return 2;
  }
  while((fgets(line,STRLEN,fin))!=NULL)
    NDM101++;
  fprintf(outflog,"found %d DM trials.\n",NDM101);
  rewind(fin);
  fclose(fin);
  //allocate
  MEMSIZEOFFSET=sizeof(int)*NCHANOUT*NDM101;
  HANDLE_ERROR(cudaHostAlloc((void **)&hostOFFSET,MEMSIZEOFFSET,cudaHostAllocDefault));
  HANDLE_ERROR(cudaMalloc((void **)&OFFSET,MEMSIZEOFFSET));
  HANDLE_ERROR(cudaHostAlloc((void **)&hostFULLFB,MEMSIZEFULLFB,cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void **)&timein0_host,MEMSIZEUTIME,cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void **)&timein1_host,MEMSIZEUTIME,cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void **)&powAVE_host,MEMSIZEPOWER,cudaHostAllocDefault));
  ////
  HANDLE_ERROR(cudaMalloc((void **)&powFULLFB,MEMSIZEFULLFB));
  //HANDLE_ERROR(cudaMalloc((void **)&powFULLFB,MAXNT*NCHANOUT*(in_nbits/8)));
  ////
  //HANDLE_ERROR(cudaMalloc((void **)&powAVEbyte,NFREQ*NBUNDLE*in_nbits/8));
  HANDLE_ERROR(cudaMalloc((void **)&powAVEbyte,MEMSIZEPOWER));
  HANDLE_ERROR(cudaMalloc((void **)&powAVE,MEMSIZEPOWER));
  HANDLE_ERROR(cudaMalloc((void **)&powTOT,MEMSIZEPOWER*NAVE));
  HANDLE_ERROR(cudaMalloc((void **)&timein0,MEMSIZETIME));
  HANDLE_ERROR(cudaMalloc((void **)&timein1,MEMSIZETIME));
  HANDLE_ERROR(cudaMalloc((void **)&utime0,MEMSIZEUTIME));
  HANDLE_ERROR(cudaMalloc((void **)&utime1,MEMSIZEUTIME));
  HANDLE_ERROR(cudaMalloc((void **)&freqout0,sizeof(cufftComplex)*NFREQ*NAVE*NBUNDLE));
  HANDLE_ERROR(cudaMalloc((void **)&freqout1,sizeof(cufftComplex)*NFREQ*NAVE*NBUNDLE));
  DM101_list=(float *)malloc(NDM101*sizeof(float));

  //buffers to hold a single frame of vdif data, for each pol
  data0=(unsigned char *)malloc(DATALEN*sizeof(unsigned char));
  data1=(unsigned char *)malloc(DATALEN*sizeof(unsigned char));
  //buffers to hold data to fill-in for skipped frames
  skip0=(unsigned char *)malloc(DATALEN*sizeof(unsigned char));
  skip1=(unsigned char *)malloc(DATALEN*sizeof(unsigned char));
  //buffers to hold next good frame of data when skipped frames found
  tmpdata0=(unsigned char *)malloc(DATALEN*sizeof(unsigned char));
  tmpdata1=(unsigned char *)malloc(DATALEN*sizeof(unsigned char));
	 
  //setup FFT
  fprintf(outflog,"calculating FFT plan...\n");
  if(cufftPlan1d(&plan,TWONCHAN,CUFFT_R2C,NAVE*NBUNDLE) != CUFFT_SUCCESS){
    fprintf(outflog,"Error creating FFT plan!\n");
    return 1;
  }

  //setup dedisp plan
  fprintf(outflog,"calculating dedisp plan...\n");
  if((err=dedisp_create_plan(&planDD,(dedisp_size)NCHANOUT,(dedisp_float)tchan,(dedisp_float)fch1,(dedisp_float)CHANBW)) != DEDISP_NO_ERROR){
    fprintf(outflog,"Error creating dedisp plan!\n");
    fprintf(outflog,"%s\n",dedisp_get_error_string(err));
    return 1;
  }


  //read DM trial list from file
  sprintf(fname,"DM101.txt");
  fprintf(outflog,"reading DM list from %s...\n",fname);
  if((fin=fopen(fname,"r"))==NULL){
    fprintf(outflog,"ERROR opening file %f!\n",fname);
    return 2;
  }
  for(i=0;i<NDM101;i++){
    fgets(line,STRLEN,fin);
    sscanf(line,"%f",&DM101_list[i]);
  }
  //dm_count=NDM101;
  if((err=dedisp_set_dm_list(planDD,DM101_list,NDM101)) != DEDISP_NO_ERROR){
    fprintf(outflog,"Error setting DM list!\n");
    fprintf(outflog,"%s\n",dedisp_get_error_string(err));
    return 1;
  } 
  dm_count=dedisp_get_dm_count(planDD);
  max_delay=dedisp_get_max_delay(planDD);//gets max delay in samples applied during dedispersion
  fprintf(outflog,"dm_count = %d\n",dm_count);
  fprintf(outflog,"max_delay= %d [samples]\n",max_delay);
  cudaMemcpyToSymbol(CNDM,&NDM101,sizeof(int));

  //setup OFFSET array
  // NCHANOUT is fastest changing index. Starts at highest freq
  // NDM is slower index. starts at 0
  //fprintf(outflog,"calculating offset array...\n");
  //for(i=0;i<NDM101;i++){
  //hostOFFSET[i*NCHANOUT]=0;//highest freq chan is the reference
  //for(j=1;j<NCHANOUT;j++){
  //  dt=4.148808e3*DM101_list[i]*(pow(fch1-(j*CHANBW),-2)-pow(fch1,-2));//sec
  //  hostOFFSET[(i*NCHANOUT)+j]=(int)(dt/tchan);
  //}
  ////printf("off= %d\n",hostOFFSET[(i+1)*NCHANOUT-1]);fflush(stdout);
  //}
  //copy OFFSET to device
  //cudaMemcpy(OFFSET,hostOFFSET,MEMSIZEOFFSET,cudaMemcpyHostToDevice);
  //calc max_delay (should check that it is a multiple of NBUNDLE & NHALFCYC)
  //max_delay=(int)((4.148808e3*DM101_list[NDM101-1]*(pow(fch1-((NCHANOUT)*CHANBW),-2)-pow(fch1,-2)))/tchan);//samples

  MEMSIZELEFTOVER=NCHANOUT*max_delay*(in_nbits/8);
  HANDLE_ERROR(cudaMalloc((void **)&powLEFTOVER,MEMSIZELEFTOVER));


  //fprintf(outflog,"Setting gulp size...\n");
  //if((err=dedisp_set_gulp_size(planDD,16384)) != DEDISP_NO_ERROR){
  //fprintf(outflog,"Error setting gulp size!\n");
  //fprintf(outflog,"%s\n",dedisp_get_error_string(err));
  //return 1;
  //} 


  //allocate outDD
  HANDLE_ERROR(cudaHostAlloc((void **)&FRB_host,dm_count*sizeof(int),cudaHostAllocDefault));
  HANDLE_ERROR(cudaMalloc((void **)&FRB,dm_count*sizeof(int)));
  HANDLE_ERROR(cudaHostAlloc((void **)&ave_host,dm_count*sizeof(float),cudaHostAllocDefault));
  HANDLE_ERROR(cudaMalloc((void **)&ave,dm_count*sizeof(float)));
  HANDLE_ERROR(cudaHostAlloc((void **)&sigma_host,dm_count*sizeof(float),cudaHostAllocDefault));
  HANDLE_ERROR(cudaMalloc((void **)&sigma,dm_count*sizeof(float)));
  HANDLE_ERROR(cudaHostAlloc((void **)&peak_host,dm_count*sizeof(float),cudaHostAllocDefault));
  HANDLE_ERROR(cudaMalloc((void **)&peak,dm_count*sizeof(float)));
  //size_t MEMSIZEOUTDD;
  unsigned int MEMSIZEOUTDD;
  NTIME_OUTDD=MAXNT-max_delay;
  //MEMSIZEOUTDD=sizeof(float)*dm_count*NTIME_OUTDD;
  MEMSIZEOUTDD=dm_count*NTIME_OUTDD*(out_nbits/8);
  //
  HANDLE_ERROR(cudaMalloc((void **)&outDD,MEMSIZEOUTDD));
  HANDLE_ERROR(cudaHostAlloc((void **)&host_outDD,MEMSIZEOUTDD,cudaHostAllocDefault));
  //HANDLE_ERROR(cudaMalloc((void **)&outDD,dm_count*NTIME_OUTDD*(out_nbits/8)));
  ///
  cudaMemcpyToSymbol(CNTIME_OUTDD,&NTIME_OUTDD,sizeof(int));
  fprintf(outflog,"MEMSIZEFULLFB= %u\n",MEMSIZEFULLFB);
  fprintf(outflog,"MEMSIZEOUTDD=  %u\n",MEMSIZEOUTDD);
  fprintf(outflog,"MEMSIZEUTIME= %u\n",MEMSIZEUTIME);
  fprintf(outflog,"MEMSIZETIME = %u\n",MEMSIZETIME);
  fprintf(outflog,"MEMSIZEPOWER= %u\n",MEMSIZEPOWER);
  fprintf(outflog,"MEMSIZEFILTERBANK= %u\n",MEMSIZEFILTERBANK);
  fprintf(outflog,"MEMSIZELEFTOVER= %u\n",MEMSIZELEFTOVER);
  fprintf(outflog,"NSKIP= %u\n",NSKIP);
  fprintf(outflog,"NTIME_OUTDD= %d\n",NTIME_OUTDD);

  //initialize
  NSKIP0=0;
  NSKIP1=0;
  INDREAD=0;
  FLAGEOF=1;
  NPARTLEFT=0;
  //initialize frame buffers with first vdif frame from file
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
  tsamp=NAVE*TWONCHAN/VLITERATE;          //[sec]

  //write SIGPROC header to outf1
  send_string("HEADER_START",outf1);
  send_int("barycentric",1,outf1);
  send_int("telescope_id",12,outf1);//SIGPROC doesn't have an alias for the VLA!
  send_int("data_type",1,outf1);
  send_double("fch1",fch1,outf1);
  send_double("foff",foffNEG,outf1);//negative foff, fch1 is highest freq
  send_int("nchans",NCHANOUT,outf1);
  send_int("nbits",nbits,outf1);
  send_double("tstart",tstart,outf1);
  send_double("tsamp",tsamp,outf1);//[sec]
  send_int("nifs",nifs,outf1);
  send_string("HEADER_END",outf1);


  //printf("sizeof dedisp_size= %u\n",sizeof(dedisp_size));


  NTHREAD_STAT=1;
  NBLOCK_STAT=dm_count;


  NTHREAD_DD=NTHREAD_SUM;
  NBLOCK_DD=(int)(NDM101*NTIME_OUTDD/NTHREAD_DD)+1;

  printf("NTHREAD_DD= %d  NBLOCK_DD= %d\n",NTHREAD_DD,NBLOCK_DD);
  fflush(stdout);

  int q=0;
  ///start looping here
  fprintf(outflog,"begin main loop...\n");
  while(FLAGEOF){
    //fill time buffer on host with float data from frame buffers
    //cudaEventRecord(start1,0);
    if(filltimebufferHOST(timein0_host,timein1_host)){ 
      //cudaEventRecord(stop1,0); //not necessary
      break;
    }
    //cudaEventRecord(stop1,0);
    //cudaEventSynchronize(stop1);
    //cudaEventElapsedTime(&time,start1,stop1);
    //tfill+=time;
    

    //copy data in host memory to GPU
    //cudaEventRecord(start1,0);
    cudaMemcpy(utime0,timein0_host,MEMSIZEUTIME,cudaMemcpyHostToDevice);
    cudaMemcpy(utime1,timein1_host,MEMSIZEUTIME,cudaMemcpyHostToDevice);
    //cudaEventRecord(stop1,0);
    //cudaEventSynchronize(stop1);
    //cudaEventElapsedTime(&time,start1,stop1);
    //tcptodevice+=time;

    //convert unsigned char time arrays to float arrays on GPU
    convertarray<<<NBLOCK_CONVERT,NTHREAD_CONVERT>>>(timein0,utime0);
    convertarray<<<NBLOCK_CONVERT,NTHREAD_CONVERT>>>(timein1,utime1);


    /////generate filterbank
    //FT time data for each pol, in all bundles
    //cudaEventRecord(start1,0);
    if(cufftExecR2C(plan,timein0,freqout0) != CUFFT_SUCCESS){
      fprintf(stderr,"Error executing plan for pol 0!\n");
      return(1);
    }
    if(cufftExecR2C(plan,timein1,freqout1) != CUFFT_SUCCESS){
      fprintf(stderr,"Error executing plan for pol 1!\n");
      return(1);
    }
    //cudaEventRecord(stop1,0);
    //cudaEventSynchronize(stop1);
    //cudaEventElapsedTime(&time,start1,stop1);
    //tFT+=time;

    //sum powers in both polarizations before averaging, all bundles
    //cudaEventRecord(start1,0);
    sumpower<<<NBLOCK_SUM,NTHREAD_SUM>>>(freqout0,freqout1,powTOT);
    //cudaEventRecord(stop1,0);
    //cudaEventSynchronize(stop1);
    //cudaEventElapsedTime(&time,start1,stop1);
    //tsum+=time;

    //average power, all bundles
    //cudaEventRecord(start1,0);
    avepower<<<NBLOCK_AVE,NTHREAD_AVE>>>(powTOT,powAVE);
    //cudaEventRecord(stop1,0);
    //cudaEventSynchronize(stop1);
    //cudaEventElapsedTime(&time,start1,stop1);
    //tave+=time;


    //convert powAVE to dedisp_byte
    //printf("memcpy powAVE to powAVEbyte on device (loopcnt= %d, loopcntTOT= %d)...\n",loopcnt,loopcntTOT);
    //HANDLE_ERROR(cudaMemcpy(&powAVEbyte[0],&powAVE[0],MEMSIZEPOWER,cudaMemcpyDeviceToDevice));
    //convertarraybyte<<<NBLOCK_CONVERT,NTHREAD_CONVERT>>>(powAVEbyte,powAVE);


    //copy data in bundles to full filterbank array
    FILL=loopcnt*NCHANOUT*NBUNDLE;
    for(j=0;j<NBUNDLE;j++){
      //printf("memcpy powAVEbyte to powFULLFB on device...\n");     
      //cudaMemcpy(&powFULLFB[FILL+(j*NCHANOUT)],&powAVE[NSKIP+(j*NFREQ)],MEMSIZEFILTERBANK,cudaMemcpyDeviceToDevice);
      //HANDLE_ERROR(cudaMemcpy(&powFULLFB[4*(FILL+(j*NCHANOUT))],&powAVEbyte[4*(NSKIP+(j*NFREQ))],MEMSIZEFILTERBANK,cudaMemcpyDeviceToDevice));
      HANDLE_ERROR(cudaMemcpy(&powFULLFB[4*(FILL+(j*NCHANOUT))],&powAVE[(NSKIP+(j*NFREQ))],MEMSIZEFILTERBANK,cudaMemcpyDeviceToDevice));
    }



    //when powFULLFB filled:
    // 1 dedisp
    // 2 calc ave, std dev, search for peaks
    // 3 copy max_delay columns to start of powFULLFB, loop and refill w/ more data 
    //dedisperse powFULLFB
    //fprintf(outflog,"executing dedisp plan...\n");
    //if(dedisp_execute(planDD,(dedisp_size)NCHANOUT,powFULLFB,32,outDD,32,DEDISP_DEVICE_POINTERS) != DEDISP_NO_ERROR){
    //fprintf(outflog,"Error executing dedisp plan!\n");
    //return 1;
    //}
    


    loopcnt++;
    //is powFULLFB full? if so, dedisperse
    if(loopcnt==(MAXNT/NBUNDLE)){
      //copy leftover end of FB array to holding buffer before BP+cal subtraction
      HANDLE_ERROR(cudaMemcpy(&powLEFTOVER[0],&powFULLFB[4*(MAXNT-max_delay)*NCHANOUT],MEMSIZELEFTOVER,cudaMemcpyDeviceToDevice));

      //subtract bandpass and cal signal 
      // calc and subtract bandpass in each on/off half-cycle to naturally subtract both BP and cal signal
      // 128 columns in each half-cycle (4 bundles)
      subtractbpCAL<<<NBLOCK_SUB,NTHREAD_SUB>>>((cufftReal *)powFULLFB);

      //subtract average of each freq column from channels
      //subtractZERO<<<NBLOCK_SUB,NTHREAD_SUB>>>((cufftReal *)powFULLFB);

      //debugging:
      if(q==4)
	cudaMemcpy(hostFULLFB,(cufftReal *)powFULLFB,MEMSIZEFULLFB,cudaMemcpyDeviceToHost);

      //dedisperse
      fprintf(outflog,"loopcnt= %d, loopcntTOT= %d  (loopcnt*NBUNDLE= %d)\n",loopcnt,loopcntTOT,loopcnt*NBUNDLE);
      fprintf(outflog,"executing dedisp plan...\n");
      ////if((err=dedisp_execute(planDD,(dedisp_size)MAXNT,(dedisp_byte *)powFULLFB,in_nbits,(dedisp_byte *)outDD,out_nbits,DEDISP_DEVICE_POINTERS)) != DEDISP_NO_ERROR){
      if((err=dedisp_execute(planDD,(dedisp_size)MAXNT,powFULLFB,in_nbits,(dedisp_byte *)outDD,out_nbits,DEDISP_DEVICE_POINTERS)) != DEDISP_NO_ERROR){
	fprintf(outflog,"Error executing dedisp plan!\n");
	fprintf(outflog,"%s\n",dedisp_get_error_string(err));
      ////return 1;
      }
      //replace with own dedisp routine
      //fprintf(outflog,"calling dedisperse...\n");
      //dedisperse<<<NBLOCK_DD,NTHREAD_DD>>>((cufftReal *)powFULLFB,outDD,OFFSET);

      //debugging:
      if(q==4)
	cudaMemcpy(host_outDD,outDD,MEMSIZEOUTDD,cudaMemcpyDeviceToHost);
      
      //search for signals
      calcstats<<<NBLOCK_STAT,NTHREAD_STAT>>>(outDD,FRB,ave,sigma,peak);  

      //debugging:
      if(q==4){
	cudaMemcpy(FRB_host,FRB,dm_count*sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(ave_host,ave,dm_count*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(sigma_host,sigma,dm_count*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(peak_host,peak,dm_count*sizeof(float),cudaMemcpyDeviceToHost);
      }
 
      //copy leftover buffer to beginning of powFULLFB
      cudaMemcpy(&powFULLFB[0],&powLEFTOVER[0],MEMSIZELEFTOVER,cudaMemcpyDeviceToDevice);
      //set FILL location accordingly
      FILL=max_delay*NCHANOUT*(in_nbits/8);
      //set loopcnt accordingly
      //loopcnt=FILL/NBUNDLE;
      loopcnt=max_delay/NBUNDLE;

      q++;
    }

    loopcntTOT++;
  
  }//end while loop
  fprintf(outflog,"Looped %d total times, (loopcnt= %d) (q= %d)\n\n",loopcntTOT,loopcnt,q);

  //subtract bandpass
  //subtractbp<<<NBLOCK_AVE,NTHREAD_AVE>>>((cufftReal *)powFULLFB);




  //fprintf(outflog,"executing dedisp plan...\n");
  ////if((err=dedisp_execute(planDD,(dedisp_size)MAXNT,(dedisp_byte *)powFULLFB,in_nbits,(dedisp_byte *)outDD,out_nbits,DEDISP_DEVICE_POINTERS)) != DEDISP_NO_ERROR){
  //if((err=dedisp_execute(planDD,(dedisp_size)MAXNT,powFULLFB,in_nbits,(dedisp_byte *)outDD,out_nbits,DEDISP_DEVICE_POINTERS)) != DEDISP_NO_ERROR){
  //fprintf(outflog,"Error executing dedisp plan!\n");
  //fprintf(outflog,"%s\n",dedisp_get_error_string(err));
  //  //return 1;
  //}
  
  //calc ave, std dev, for each DM trial in outDD
  //calcstats<<<NBLOCK_AVE,NTHREAD_AVE>>>(outDD,FRB,ave,sigma,peak);  
  //copy FRB array to host
  //cudaMemcpy(FRB_host,FRB,dm_count*sizeof(int),cudaMemcpyDeviceToHost);
  //cudaMemcpy(ave_host,ave,dm_count*sizeof(float),cudaMemcpyDeviceToHost);
  //cudaMemcpy(sigma_host,sigma,dm_count*sizeof(float),cudaMemcpyDeviceToHost);
  //cudaMemcpy(peak_host,peak,dm_count*sizeof(float),cudaMemcpyDeviceToHost);

  //write host_outDD to file for debugging
  int k=0;
  for(i=0;i<dm_count;i++){
    fwrite(&DM101_list[i],sizeof(float),1,outf3);
    //for(j=0;j<NTIME_OUTDD;j++)
    fwrite(&host_outDD[(i*NTIME_OUTDD)],sizeof(dedisp_float)*NTIME_OUTDD,1,outf3);
  }
#if DEBUG
  for(i=0;i<dm_count;i++){
    if(DM101_list[i]>20.7 && DM101_list[i]<27.3){
      //printf("%d DM= %f\n",i,DM101_list[i]);
      fprintf(outf3,"%f %f %f %f %d  ",DM101_list[i],ave_host[i],sigma_host[i],peak_host[i],FRB_host[i]);
    }
    for(j=0;j<NTIME_OUTDD;j++){
      if(DM101_list[i]>20.7 && DM101_list[i]<27.3)
	fprintf(outf3,"%f ",host_outDD[k]);
      k++;
    }
    if(DM101_list[i]>20.7 && DM101_list[i]<27.3)
      fprintf(outf3,"\n\n");
  }
#endif
  ////if(FRB_host[i]==1)
    //fprintf(outflog,"  FRB found (%d)! i= %d, DM= %f, ave= %f, sigma= %f, peak= %f\n",FRB_host[i],i,i*0.1,ave_host[i],sigma_host[i],peak_host[i]);

  //copy data in GPU memory to host
  //cudaEventRecord(start1,0);
  //  cudaMemcpy(hostFULLFB,(cufftReal *)powFULLFB,loopcnt*MEMSIZEFILTERBANK*NBUNDLE,cudaMemcpyDeviceToHost);
  //cudaEventRecord(stop1,0);
  //cudaEventSynchronize(stop1);
  //cudaEventElapsedTime(&time,start1,stop1);
  //tcptohost+=time;
  
  //printf("%u\n",MEMSIZEFULLFB);
  //printf("%u\n",loopcnt*NBUNDLE*MEMSIZEFILTERBANK);
  
  //write filterbank to output file
  // VLITE data in LSB, do not need to flip band. Highest freq is array index 1
  // Nyquist freq is array index 0
  // only write NCHANOUT, skip MUOS polluted chans and skip Nyquist (included in NSKIP)
  //
  //cudaEventRecord(start1,0);
  ////for(j=0;j<NBUNDLE;j++){
  ////fwrite(&hostFULLFB[0],loopcnt*MEMSIZEFILTERBANK*NBUNDLE,1,outf2);
  fwrite(&hostFULLFB[0],MEMSIZEFULLFB,1,outf2);
  //////}


  //cleanup
  fclose(ft);
  fflush(outf1);
  fflush(outf2);
  fflush(outf3);
  fclose(outf1);
  fclose(outf2);
  fclose(outf3);
  cufftDestroy(plan);
  dedisp_destroy_plan(planDD);

  fprintf(outflog,"Processed %f [sec] of data\n",loopcntTOT*tBUNDLE);
  //record timing
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time,start,stop);
  //timems=time;//ms
  //time*=1e-3;//s
  //twrite*=1e-3;//s
  //tfill*=1e-3;//s
  //tGPU=1e-3*(tcptodevice+tFT+tsum+tave+tcptohost);//s
  fprintf(outflog,"Total run time = %f [sec]\n",time*1e-3);
  //fprintf(outflog,"   Time to fill time buffer on HOST =  %.3f [s], = %f of total\n",tfill,tfill/time);
  //fprintf(outflog,"   Time to write .fil files         =  %.3f [s], = %f of total\n",twrite,twrite/time);
  //fprintf(outflog,"   Total GPU-related time           =  %.3f [s], = %f of total\n",tGPU,tGPU/time);
  //fprintf(outflog,"   \tTime to memcopy to device     =  %.0f [ms], = %f of total\n",tcptodevice,tcptodevice/timems);
  //fprintf(outflog,"   \tTime to filterbank            =  %.0f [ms], = %f of total\n",tFT,tFT/timems);
  //fprintf(outflog,"   \tTime to sumpower              =  %.0f [ms], = %f of total\n",tsum,tsum/timems);
  //fprintf(outflog,"   \tTime to avepower              =  %.0f [ms], = %f of total\n",tave,tave/timems);
  //fprintf(outflog,"   \tTime to memcopy to host       =  %.0f [ms], = %f of total\n",tcptohost,tcptohost/timems);
  //fprintf(outflog,"\n");

  /////////
  //more cleanup
  /////////
  cudaEventDestroy(start); cudaEventDestroy(stop);
  //cudaEventDestroy(start1); cudaEventDestroy(stop1);

  fprintf(outflog,"\nfinished!\n");fflush(outflog);
  fclose(outflog);
  cudaEventRecord(stopC,0);
  cudaEventSynchronize(stopC);
  cudaEventElapsedTime(&time,startC,stopC);
  fprintf(stdout,"TOTAL TIME = %f [sec]\n",time*1e-3);
  return(0);
}//end of main


int initframePOL0(){
  //read frame header
  if(fread(&hdr0,1,HDRLEN,ft) != HDRLEN){
    //check if EOF 
    if((FLAGEOF=!feof(ft))==0){
      fprintf(outflog,"EOF while trying to read header0\n");
      return 1;
    }
    else{
      fprintf(outflog,"ERROR while trying to read header0!!!\n");
      return 2;
    }
  }
  //read frame data
  if(fread(data0,1,DATALEN,ft) != DATALEN){
    //check if EOF 
    if((FLAGEOF=!feof(ft))==0){
      fprintf(outflog,"EOF while trying to read data0\n");
      return 1;
    }
    else{
      fprintf(outflog,"ERROR while trying to read data0!!!\n");
      return 2;
    }
  }
  return 0;
}  

int initframePOL1(){
  //read frame header
  if(fread(&hdr1,1,HDRLEN,ft) != HDRLEN){
    //check if EOF 
    if((FLAGEOF=!feof(ft))==0){
      fprintf(outflog,"EOF while trying to read header1\n");
      return 1;
    }
    else{
      fprintf(outflog,"ERROR while trying to read header1!!!\n");
      return 2;
    }
  }
  //read frame data
  if(fread(data1,1,DATALEN,ft) != DATALEN){
    //check if EOF 
    if((FLAGEOF=!feof(ft))==0){
      fprintf(outflog,"EOF while trying to read data1\n");
      return 1;
    }
    else{
      fprintf(outflog,"ERROR while trying to read data1!!!\n");
      return 2;
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


//read 1 VDIF frame for pol 0 
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
    if(fread(&hdr0,1,HDRLEN,ft) != HDRLEN){
      //check if EOF 
      if((FLAGEOF=!feof(ft))==0){
	fprintf(outflog,"EOF while trying to read header0\n");
	return 1;
      }
      else{
	fprintf(outflog,"ERROR while trying to read header0!!!\n");
	return 2;
      }
    }
    //read frame data
    if(fread(data0,1,DATALEN,ft) != DATALEN){
      //check if EOF 
      if((FLAGEOF=!feof(ft))==0){
	fprintf(outflog,"EOF while trying to read data0\n");
	return 1;
      }
      else{
	fprintf(outflog,"ERROR while trying to read data0!!!\n");
	return 2;
      }
    }
    //check for frame skips
    checkframeskip0();
    //if skipped frame, checkframeskip copies data0 into tmpdata0 
    //  and skip0 into data0, so skip values will be returned
    return 0;
  }
}  

//read 1 VDIF frame for pol 1
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
    if(fread(&hdr1,1,HDRLEN,ft) != HDRLEN){
      //check if EOF 
      if((FLAGEOF=!feof(ft))==0){
	fprintf(outflog,"EOF while trying to read header1\n");
	return 1;
      }
      else{
	fprintf(outflog,"ERROR while trying to read header1!!!\n");
	return 2;
      }
    }
    //read frame data
    if(fread(data1,1,DATALEN,ft) != DATALEN){
      //check if EOF 
      if((FLAGEOF=!feof(ft))==0){
	fprintf(outflog,"EOF while trying to read data1\n");
	return 1;
      }
      else{
	fprintf(outflog,"ERROR while trying to read data1!!!\n");
	return 2;
      }
    }
    //check for frame skips
    checkframeskip1();
    //if skipped frame, checkframeskip copies data1 into tmpdata1 
    //  and skip1 into data1, so skip values will be returned
    return 0;
  }
}




//read a frame for each pol from file
int readtwoframe(){
  int flag;
  //pol0
  flag=readoneframePOL0();
  if(flag) return(flag);
  //pol1
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


//initialize buffers 0 & 1 with a frame of data from file
//  advance to start of cal signal
//  skip initial frame if not pol 0
int initframe01()
{
  int id0,id1,fnum0,fnum1,ncal=0,flag;
  //read pol 0
  //flag=readoneframePOL0();
  flag=initframePOL0();
  if(flag) return flag;    
  //check pol 0, if not then try next frame
  id0=getVDIFThreadID(&hdr0);
  if(id0 != 0){
    fprintf(outflog,"VDIF Threads misaligned, skipping forward 1 frame\n");
    //flag=readoneframePOL0();
    flag=initframePOL0();
    if(flag) return flag;    
  }
  //double check
  id0=getVDIFThreadID(&hdr0);
  if(id0 != 0){
    fprintf(outflog,"ERROR, initframe01! VDIF Thread ID= %d, != 0. Abort!\n",id0);
    return 1;
  }

  //read pol 1
  //flag=readoneframePOL1();
  flag=initframePOL1();
  if(flag) return flag;    
  //check pol 1
  id1=getVDIFThreadID(&hdr1);
  if(id1 != 1){
    fprintf(outflog,"VDIF Thread ID= %d, != 1. Abort!\n",id1);
    return 1;
  }

  //check agreement of pol frame numbers
  fnum0=getVDIFFrameNumber(&hdr0);
  fnum1=getVDIFFrameNumber(&hdr1);
  if(fnum0 != fnum1){
    fprintf(outflog,"ERROR! polarization VDIF frame numbers do not agree! 0: %d  !=  1: %d\n",fnum0,fnum1);
    return(1);
  }

  //find start of cal signal (10 Hz) FRAMESPERCAL = FRAMESPERSEC/10Hz = 2560
  while(1){
    if(fmod((float)fnum0,(float)2560) < 1e-4){//this frame is start of cal signal (OFF), exit loop
      fprintf(outflog,"Skipped %d frames to begin at noise cal cycle\n",ncal);
      break; 
    }
    //otherwise, read next pair of frames
    flag=initframePOL0();
    if(flag) return flag;
    flag=initframePOL1();
    //flag=readtwoframe();
    if(flag) return flag;
    fnum0=getVDIFFrameNumber(&hdr0);
    //fnum1=getVDIFFrameNumber(&hdr1);
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
  int q,NWHOLE,NTOT,IFILL=0,fnum0,fnum1;
  if(NPARTLEFT){
    memcpy(&timein0[IFILL],&data0[NPARTFILL],NPARTLEFT);
    memcpy(&timein1[IFILL],&data1[NPARTFILL],NPARTLEFT);
    IFILL+=NPARTLEFT;
    //refill frame buffers
    if(readtwoframe()) return(1);
    //check for skipped frames
    //checkframeskip();
  }
  NTOT=NLOOP-NPARTLEFT;
  NWHOLE=NTOT/DATALEN;
  NPARTFILL=NTOT % DATALEN;

  NPARTLEFT=DATALEN-NPARTFILL;

  for(q=0;q<NWHOLE;q++){
    memcpy(&timein0[IFILL],&data0[0],DATALEN);
    memcpy(&timein1[IFILL],&data1[0],DATALEN);
    IFILL+=DATALEN;    
    //refill frame buffers
    if(readtwoframe()) return(1);
    //check for skipped frames
    //checkframeskip();
  }
  
  if(NPARTFILL){
    memcpy(&timein0[IFILL],&data0[0],NPARTFILL);
    memcpy(&timein1[IFILL],&data1[0],NPARTFILL);
    IFILL+=NPARTFILL;
  }  

  //returns with full or partially filled data0[] & data1[]  
  return(0);
}



//modified from send_stuff.c from SIGPROC
////
void send_string(const char *string, FILE *output)
{
  int len;
  len=strlen(string);
  fwrite(&len, sizeof(int), 1, output);
  fwrite(string, sizeof(char), len, output);
}

void send_float(const char *name,float floating_point, FILE *output)
{
  send_string(name,output);
  fwrite(&floating_point,sizeof(float),1,output);
}

void send_double (const char *name, double double_precision, FILE *output)
{
  send_string(name,output);
  fwrite(&double_precision,sizeof(double),1,output);
}

void send_int(const char *name, int integer, FILE *output)
{
  send_string(name,output);
  fwrite(&integer,sizeof(int),1,output);
}

void send_long(const char *name, long integer, FILE *output)
{
  send_string(name,output);
  fwrite(&integer,sizeof(long),1,output);
}

void send_coords(double raj, double dej, double az, double za, FILE *output)
{
  if ((raj != 0.0) || (raj != -1.0)) send_double("src_raj",raj,output);
  if ((dej != 0.0) || (dej != -1.0)) send_double("src_dej",dej,output);
  if ((az != 0.0)  || (az != -1.0))  send_double("az_start",az,output);
  if ((za != 0.0)  || (za != -1.0))  send_double("za_start",za,output);
}

int strings_equal (char *string1, char *string2)
{
  if (!strcmp(string1,string2)) return 1;
  else return 0;
}

//////////////////////
void print_usage(){
  fprintf(stderr,"***************************************************************************\n");
  fprintf(stderr,"This program reads time series data from a file with VDIF format data and outputs\n");
  fprintf(stderr,"  a filterbank file in SIGPROC format.\n\n");
  fprintf(stderr,"  Usage:\n./vdifFILE2fil_PIPELINE -d:\t    use default values (unless other values given)\n");
  fprintf(stderr,"\t             -i (string):   name of input file to read VDIF data from,\n");
  fprintf(stderr,"\t             -o (string):   base name of ouput files to write,\n");
  fprintf(stderr,"\t             -nave (int):   number of time samples to average over,\n");
  fprintf(stderr,"\t             -nchan (int):  number of frequency channels,\n");
  fprintf(stderr,"\t             -nb (int):     number of bundles of size (2*2*nchan*nave) to send in 1 memcpy\n");
  fprintf(stderr,"\n");
  fprintf(stderr,"  DEFAULTS:\n");
  fprintf(stderr,"\t             -i     %s\n","/data/VLITE/0329+54/20150302_151845_B0329+54_ev_0005.out.uw");
  fprintf(stderr,"\t             -o     testDEDISP\n");
  fprintf(stderr,"\t             -nave  4\n");
  fprintf(stderr,"\t             -nchan 6250\n");
  fprintf(stderr,"\t             -nb    32\n");
  fprintf(stderr,"\n");
  return;
}

int parser(int argc, char *argv[]){
  int i=1;
  if(strings_equal(argv[i],"-d")){
    //set defaults
    TWONCHAN=6250*2;
    NAVE=4;
    NBUNDLE=32;
    strcpy(VDIFNAME,"/data/VLITE/0329+54/20150302_151845_B0329+54_ev_0005.out.uw");
    strcpy(OUTNAME,"testDEDISP");
    i++;
  }

  while(i<argc){
    if(strings_equal(argv[i],"-nchan")){
      i++;
      TWONCHAN=2*atoi(argv[i]);
    }
    else if(strings_equal(argv[i],"-nave")){
      i++;
      NAVE=atoi(argv[i]);
    }
    else if(strings_equal(argv[i],"-nb")){
      i++;
      NBUNDLE=atoi(argv[i]);
    }
    else if(strings_equal(argv[i],"-i")){
      i++;
      strcpy(VDIFNAME,argv[i]);
    }
    else if(strings_equal(argv[i],"-o")){
      i++;
      strcpy(OUTNAME,argv[i]);
    }
    else{
      //unknown argument
      fprintf(stderr,"Error! Unknown argument: %s\n\n",argv[i]);
      print_usage();   
      return(3);
    }
    i++;
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

//convert float array to dedisp_byte -- PROBABLY DON'T NEED ANYMORE
__global__ void convertarraybyte(dedisp_byte *powbyte, cufftReal *pow){
  int i;
  int tot=CNCHANOUT*CMAXNT;
  i=threadIdx.x+(blockIdx.x*blockDim.x);  
  if(i < tot)
    //cudaMemcpy(&powbyte[i],&pow[i],4*sizeof(dedisp_byte),cudaMemcpyDeviceToDevice);
    powbyte[i]=(dedisp_byte)(pow[i]);
  return;
}


//search for peaks in dedispersed array
__global__ void calcstats(dedisp_float *outDD, int *FRB, float *ave, float *sigma, float *peak){
  int DM_ind,TOT_ind,i;
  double tave,tsigma;
  double sum,sum2,dout;
  ///////////////////////////////////////////
  DM_ind=threadIdx.x+(blockIdx.x);
  ///////////////////////////////////////////
  if(DM_ind < CNDM){
    FRB[DM_ind]=0;
    peak[DM_ind]=0.0;
    TOT_ind= (int)(DM_ind*CNTIME_OUTDD);
    sum=0.0;
    for(i=0;i<CNTIME_OUTDD;i++)
      sum+=outDD[TOT_ind+i];
    tave=sum/CNTIME_OUTDD;
    sum=sum2=0.0;
    for(i=0;i<CNTIME_OUTDD;i++){
      dout=outDD[TOT_ind+i];
      sum2+=((dout-tave)*(dout-tave));
      sum+=(dout-tave);
    }
    tsigma=(sum2-(sum*sum)/CNTIME_OUTDD)/(CNTIME_OUTDD-1);
    ave[DM_ind]=tave;
    sigma[DM_ind]=sqrt(tsigma);
    for(i=0;i<CNTIME_OUTDD;i++){
      dout=(outDD[TOT_ind+i]-ave[DM_ind])/sigma[DM_ind];
      if(dout > 3.0)
	FRB[DM_ind]=1;
      if(dout > peak[DM_ind])
	peak[DM_ind]=dout;
    }
  }
  return;
}

__global__ void subtractbp(cufftReal *FULLFB){
  int CHAN_ind,i,j;
  float sum,sum2,ave,sig,HI,LO;
  CHAN_ind=threadIdx.x+(blockIdx.x*blockDim.x);
  if(CHAN_ind < CNCHANOUT){
    ave=0.0;
    for(j=0;j<CMAXNT;j++)
      ave+=FULLFB[CHAN_ind+(j*CNCHANOUT)];
    ave/=CNCHANOUT;
    sum=0.0;sum2=0.0;
    for(j=0;j<CMAXNT;j++){
      sum2+=((FULLFB[CHAN_ind+(j*CNCHANOUT)]-ave)*(FULLFB[CHAN_ind+(j*CNCHANOUT)]-ave));
      sum+=(FULLFB[CHAN_ind+(j*CNCHANOUT)]-ave);
    }
    sig=(sum2-(sum*sum)/CMAXNT)/(CMAXNT-1);

    //recalc ave w/o extreme points
    HI=ave+(3.0*sig);
    LO=ave-(3.0*sig);
    ave=0.0; i=0;
    for(j=0;j<CMAXNT;j++)
      if(FULLFB[CHAN_ind+(j*CNCHANOUT)]<HI && FULLFB[CHAN_ind+(j*CNCHANOUT)]>LO){
	ave+=FULLFB[CHAN_ind+(j*CNCHANOUT)];
	i++;
      }
    ave/=i;

    //2nd pass
    //sum=0.0;sum2=0.0; i=0;
    //for(j=0;j<CMAXNT;j++){
    //if(FULLFB[CHAN_ind+(j*CNCHANOUT)]<HI && FULLFB[CHAN_ind+(j*CNCHANOUT)]>LO){
    //	sum2+=((FULLFB[CHAN_ind+(j*CNCHANOUT)]-ave)*(FULLFB[CHAN_ind+(j*CNCHANOUT)]-ave));
    ///	sum+=(FULLFB[CHAN_ind+(j*CNCHANOUT)]-ave);
    //	i++;
    //}
    //}
    //sig=(sum2-(sum*sum)/i)/(i-1);
    //HI=ave+(3.0*sig);
    //LO=ave-(3.0*sig);
    //ave=0.0; i=0;
    //for(j=0;j<CMAXNT;j++)
    //if(FULLFB[CHAN_ind+(j*CNCHANOUT)]<HI && FULLFB[CHAN_ind+(j*CNCHANOUT)]>LO){
    //	ave+=FULLFB[CHAN_ind+(j*CNCHANOUT)];
    //	i++;
    //}
    //ave/=i;


    for(j=0;j<CMAXNT;j++)
      FULLFB[CHAN_ind+(j*CNCHANOUT)]-=ave;
  }
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
    //istart=row;
    npts=colmax-colmin;
    ave=0.0;
    //for(j=0;j<CMAXNT;j++)
    //for(j=colmin;j<colmax;j++)
    for(j=0;j<npts;j++)
      ave+=FULLFB[istart+(j*CNCHANOUT)];
    ave/=npts;
    sum=0.0;sum2=0.0;
    //for(j=0;j<CMAXNT;j++){
    //   for(j=colmin;j<colmax;j++){
    for(j=0;j<npts;j++){
      sum2+=((FULLFB[istart+(j*CNCHANOUT)]-ave)*(FULLFB[istart+(j*CNCHANOUT)]-ave));
      sum+=(FULLFB[istart+(j*CNCHANOUT)]-ave);
    }
    sig=(sum2-(sum*sum)/npts)/(npts-1);
    
    //recalc ave w/o extreme points
    HI=ave+(3.0*sig);
    LO=ave-(3.0*sig);
    ave=0.0; i=0;
    //for(j=0;j<CMAXNT;j++)
    //    for(j=colmin;j<colmax;j++)
    for(j=0;j<npts;j++)
      if(FULLFB[istart+(j*CNCHANOUT)]<HI && FULLFB[istart+(j*CNCHANOUT)]>LO){
	ave+=FULLFB[istart+(j*CNCHANOUT)];
	i++;
      }
    if(i==0) i=1;
    ave/=i;

    //    for(j=colmin;j<colmax;j++)
    for(j=0;j<npts;j++)
      FULLFB[istart+(j*CNCHANOUT)]-=ave;
  }
  return;
}


//subtract average of each frequency column from each frequency channel
__global__ void subtractZERO(cufftReal *FULLFB){
  int i,j,istart;
  int col;
  float sum,sum2,ave,sig,HI,LO;
  col=threadIdx.x+(blockIdx.x*blockDim.x);
  if(col < CMAXNT){
    istart=col*CNCHANOUT;
    ave=0.0;
    for(j=0;j<CNCHANOUT;j++)
      ave+=FULLFB[istart+j];
    ave/=CNCHANOUT;
    sum=0.0;sum2=0.0;

    for(j=0;j<CNCHANOUT;j++){
      sum2+=((FULLFB[istart+j]-ave)*(FULLFB[istart+j]-ave));
      sum+=(FULLFB[istart+j]-ave);
    }
    sig=(sum2-(sum*sum)/CNCHANOUT)/(CNCHANOUT-1);
    
    //recalc ave w/o extreme points
    HI=ave+(3.0*sig);
    LO=ave-(3.0*sig);
    ave=0.0; i=0;
    for(j=0;j<CNCHANOUT;j++)
      if(FULLFB[istart+j]<HI && FULLFB[istart+j]>LO){
	ave+=FULLFB[istart+j];
	i++;
      }
    if(i==0) i=1;
    ave/=i;

    for(j=0;j<CNCHANOUT;j++)
      FULLFB[istart+j]-=ave;
  }
  return;
}



__global__ void dedisperse(cufftReal *FULLFB, dedisp_float *outDD, int *OFFSET){
  int i,DMidx,TIMEidx,idx,off,FBidx;
  //int x=threadIdx.x+(blockIdx.x*blockDim.x);
  //int y=threadIdx.y+(blockIdx.y*blockDim.y);
  //int idx=x+(y*blockDim.x*gridDim.x);

  idx=threadIdx.x+(blockIdx.x*blockDim.x);
  if(idx<(CNTIME_OUTDD*CNDM)){
    DMidx=(int)(idx/CNTIME_OUTDD);
    TIMEidx=idx-(DMidx*CNTIME_OUTDD);

    outDD[idx]=0.0;
    for(i=0;i<CNCHANOUT;i++){
      off=OFFSET[(DMidx*CNCHANOUT)+i];
      FBidx=i+((off+TIMEidx)*CNCHANOUT);
      outDD[idx]+=FULLFB[FBidx];
    }
    outDD[idx]/=CNCHANOUT;
  }
  return;
}

