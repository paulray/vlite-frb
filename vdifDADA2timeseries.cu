/*
 * VLITE-FAST code to read VDIF data from a DADA ring buffer and write out dedispersed time series
 *    files in PRESTO format, 2 files: DM=0 and DM~100
 *
 * Uses CUDA FFT to perform FFTs on GPU
 *
 * DADA buffer contains VDIF data for multiple Frames consisting of a header followed by
 *   a data block.
 *   2 polarizations, frames alternate pols
 *
 *
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
#include "dada_def.h"
#include "ipcio.h"
#include "ipcbuf.h"
//#include <dedisp.h>

//version of this software
#define VERSION "1.0"

#define FRAMELEN 5032   //vdif frame length, bytes
#define HDRLEN 32       //vdif header length, bytes
#define DATALEN 5000    //vdif data length, bytes (5000 samples per frame, 1 sample = 1 byte)
#define FRAMESPERSEC 25600 //# of vdif fames per sec
#define VLITERATE 128e6 //bytes per sec = samples per sec = FRAMESPERSEC*DATALEN
#define ANTENNA "XX"
#define DEVICE 0        //GPU. on difx node: 0 = GTX 780
#define STRLEN 256

#define VLITEBW 64.0          //VLITE bandwidth, MHz
#define VLITELOWFREQ 320.0    //Low end of VLITE band, MHz
//const float VLITEFREQCENTER=VLITELOWFREQ+(0.5*VLITEBW); //center frequency, MHz
//const float VLITEHIGHFREQ=VLITELOWFREQ+VLITEBW;         //upper frequency, MHz
#define MUOSLOWFREQ 360.0     //Low end of MUOS-polluted bands, MHz
//#define MUOSLOWFREQ 384.0     //set this to write full 320-384 band to output .fil files

float MAXTIME=30.0;//max amt of time-sampled voltage data to collect before dedispersing [sec]
float HALFCYCLE=0.05; //half of a 10 Hz cal signal cycle [sec]
int NHALFCYC;         //number of cal signal half cycles in MAXTIME


int TWONCHAN; //2 * NCHAN, for FFT 
int NAVE;     //# of chan samples to average for final chan output
int NBUNDLE;  //# of TWONCHAN*NAVE bundles to process on GPU at once
int NLOOP;    //TWONCHAN*NAVE*NBUNDLE

char *tmphdr0,*tmphdr1;      //frame headers read from DADA into tmp buffers
vdif_header   hdr0, hdr1;    //vdif frame headers, 2 pol
unsigned char *data0,*data1; //vdif frame data, 2 pols
int           INDREAD;       //index of read location in buffers
unsigned char *skip0,*skip1; //vdif frame data to insert for skipped frames
unsigned char *tmpdata0,*tmpdata1; //buffers to hold frame data when skipped frames found

////////////////////
//DADA
////////////////////
ipcio_t dadain; //DADA buffer to read time data from
key_t keyin;    //DADA buffer key to read time data from
#define DADA_DEFAULT_INPUT_KEY  0x0000dada
int FLAGEODADA;    //initialized to 1, set to 0 when EOD in buffer reached
#define NFRAMEDADAINIT 256000 //number of frames of each pol to skip at start of reading DADA buffer

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
int EXPECTEDFRAMESEC0,EXPECTEDFRAMESEC1;
int NSKIP0,NSKIP1;


//functions for writing time series files
int writeprestoheader(char *fname, double mjdstart, int ntimes, double tsamp, float dm, double fchanlo, double bw, int nchan, double chanbw);


//command line functions
void print_usage();
int parser(int argc, char *argv[]);

//functions modified from sigproc.h
int strings_equal (const char *string1, char *string2);

//GPU functions
__global__ void sumpower(cufftComplex *freqout0, cufftComplex *freqout1, cufftReal *powTOT);
__global__ void avepower(cufftReal *powTOT, cufftReal *powAVE);
__global__ void convertarray(cufftReal *time, unsigned char *utime);
__global__ void calcstats(float *outDD, int *FRB, float *ave, float *sigma, float *peak);
__global__ void subtractbp(cufftReal *FULLFB);
__global__ void subtractbpCAL(cufftReal *FULLFB);
__global__ void subtractZERO(cufftReal *FULLFB);
//__global__ void dedisperse(cufftReal *FULLFB, float *outDD, int *OFFSET);
__global__ void dedisperseSTATS(cufftReal *FULLFB, float *outDD, float *sigDD, int *OFFSET);

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


////////////DADA
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

//call this function before reading data from DADA buffer
int initDADA();
/////////////////


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
  ////////////////////////////////////////////////////////////////////
  //for PRESTO files:
  char *INFFILE0;   //base name of info file, DM=0
  char *INFFILE100; //base name of info file, DM~100
  char TIMEFILE0[STRLEN];  //name of time series data file, DM=0   (chan ave)
  char TIMEFILE100[STRLEN];//name of time series data file, DM~100 (chan ave)
  char SIGFILE0[STRLEN];   //name of time series data file, DM=0   (chan sigma)
  char SIGFILE100[STRLEN]; //name of time series data file, DM~100 (chan sigma)
  FILE *outfDM0,*outfDM100,*outfDM0SIG,*outfDM100SIG;
  double fchlo;       //center freq of lowest channel [MHz]
  double bandw;       //output bandwidth [MHz]
  double mjdstart;   //MJD of start of time series
  int ntimes;        //number of time samples written to hourly file
  int hrnext;        //what is the next hour
  time_t timesystem; //system time in local time
  struct tm timeutc; //system time in UTC
  char buffer[STRLEN];
  INFFILE0=(char *)malloc(STRLEN*sizeof(char));
  INFFILE100=(char *)malloc(STRLEN*sizeof(char));
  /////////////////////////////////////////////////////////////////////

  int MAXLOOPCNT;
  int NCHAN;    //# chan across VLITEBW
  int NCHANOUT; //# chan in ouput filterbank file (excluding MUOS band)
  int NFREQ;    //NCHAN + 1, for Nyquist freq
  int NSKIP;    //# chan to skip when writing = NCHAN-NCHANOUT
  double CHANBW;//BW of a single frequency channel, MHz
  char LOGFILE[STRLEN]; //log file name for run-time messages
  char OUTDDFILE[STRLEN]; //output file names
  double fch1,foffNEG;
  int nifs=1;
  int nbits=8*sizeof(float);
  double tstart,tsamp,tBUNDLE,tchan;
  int j,FILL,i;
  loopcnt=0;
  FILE *outf1,*outf2,*outf3;

  int *hostOFFSET,*OFFSET;
  double dt;

  FILE *fin;
  char fname[STRLEN],line[STRLEN];
  cufftComplex  *freqout0,*freqout1;
  cufftReal     *timein0;
  unsigned char *timein0_host;
  cufftReal     *timein1;
  unsigned char *timein1_host;
  cufftReal     *powTOT,*powAVE,*powAVE_host,*hostFULLFB;
  cufftReal     *powFULLFB,*powLEFTOVER;
  unsigned char *utime0,*utime1;
  int NTHREAD_SUB,NBLOCK_SUB,NTOT_SUB;
  int NTHREAD_STAT,NBLOCK_STAT;
  int NTHREAD_SUM,NTHREAD_AVE,NTOT_SUM;
  int NBLOCK_SUM,NBLOCK_AVE,NTOT_AVE;
  int NBLOCK_CONVERT,NTHREAD_CONVERT,NTOT_CONVERT;
  int NBLOCK_DD,NTHREAD_DD,NBLOCK_DD_SQ;
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
  float *outDD,*sigDD;
  float *host_outDD,*host_sigDD;
  int  dm_count,max_delay;
  int NDM101;
  float *DM101_list;
  cufftHandle plan;
  cudaDeviceProp prop;
  cudaError_t cudaerr;
  int device_idx=DEVICE;
  cudaGetDeviceProperties(&prop,DEVICE);


  //check and parse command line
  if (argc < 2) {
    print_usage();
    return(1);
  }
  if(parser(argc,argv)) return(1);


  //set OUTDD and log file names and open output files:
  // log - file for messages
  //
  sprintf(LOGFILE,"%s.log",OUTNAME);
  //File access is faster if a preexisting file is deleted before opening
  if(access(LOGFILE,F_OK)!=-1)
    remove(LOGFILE);//file exists, delete it
  outflog = fopen(LOGFILE,"w");
  if (outflog<0) {
    fprintf(stderr,"Unable to open output log file: %s\n",LOGFILE);
    return(2);
  }  
  ///////////////////

  fprintf(outflog,"Running vdifDADA2timeseries version %s\n",VERSION);
  fprintf(outflog,"Running on device: %s\n",prop.name);


  //parameters
  NCHAN  = (int)(0.5*TWONCHAN);
  NFREQ  = NCHAN+1;        //+1 for Nyquist bin
  CHANBW = VLITEBW/NCHAN;  //channel BW, MHz 
  //NCHANOUT = (int)((MUOSLOWFREQ-VLITELOWFREQ)/CHANBW)+1; //# of channels written to output files
  //if((NCHANOUT % 2) == 1) NCHANOUT++; //make NCHANOUT even
  NCHANOUT=3840;
  ///////////////////
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
  MEMSIZEFULLFB     = MAXNT*NCHANOUT*sizeof(cufftReal); //size of full FB, up to MAXTIME
  MAXLOOPCNT=MAXNT/NBUNDLE;
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
  fprintf(outflog,"MAXLOOPCNT= %d\n",MAXLOOPCNT);

  ///PRESTO:
  bandw=NCHANOUT*CHANBW;
  fchlo=VLITELOWFREQ+(0.5*CHANBW);

  //device (GPU) parameters
  NTHREAD_SUM     = 512;
  NTHREAD_AVE     = NTHREAD_SUM;
  NTHREAD_CONVERT = NTHREAD_SUM;
  NTHREAD_SUB     = NTHREAD_SUM;
  NTOT_SUB        = NCHANOUT*NHALFCYC;//for BP+cal subtraction
  NTOT_SUM        = NFREQ*NAVE*NBUNDLE;
  NTOT_AVE        = NFREQ*NBUNDLE;
  NTOT_CONVERT    = NLOOP;
  NBLOCK_SUM      = (int)(NTOT_SUM/NTHREAD_SUM)+1;
  NBLOCK_AVE      = (int)ceil(NTOT_AVE/NTHREAD_AVE)+1;
  NBLOCK_CONVERT  = (int)ceil(NTOT_CONVERT/NTHREAD_CONVERT)+1;
  NBLOCK_SUB      = (int)(NTOT_SUB/NTHREAD_SUB)+1;
  STRIDEAVE       = NAVE*NFREQ;
  fprintf(outflog,"NTOT_SUM= %d   NTOT_AVE= %d  NTOT_CONVERT= %d  NTOT_SUB= %d\n",NTOT_SUM,NTOT_AVE,NTOT_CONVERT,NTOT_SUB);
  fprintf(outflog,"  NTHREAD_SUM=     %d  NBLOCK_SUM    = %d\n",NTHREAD_SUM,NBLOCK_SUM,NTHREAD_AVE,NBLOCK_AVE);
  fprintf(outflog,"  NTHREAD_AVE=     %d  NBLOCK_AVE    = %d\n",NTHREAD_SUM,NBLOCK_SUM,NTHREAD_AVE,NBLOCK_AVE);
  fprintf(outflog,"  NTHREAD_CONVERT= %d  NBLOCK_CONVERT= %d\n",NTHREAD_CONVERT,NBLOCK_CONVERT);
  fprintf(outflog,"  NTHREAD_SUB=     %d  NBLOCK_SUB    = %d\n",NTHREAD_SUB,NBLOCK_SUB);

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


  fprintf(stdout,"Connecting to DADA buffer...\n");
  fflush(stdout);
  ////////////////DADA
  //DADA ring buffers:
  //connect to existing buffer and open as primary read client
  fprintf(outflog,"DADA buffer at start:\n");
  fprintf(outflog," Connecting to DADA buffer with key: %x ...\n",keyin);
  if(ipcio_connect(&dadain,keyin) < 0){
    fprintf(outflog," Could not connect to DADA buffer!\n");
    return -1;
  }
  else
    fprintf(stdout,"Connected.\n");
  fflush(stdout);
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
  fprintf(outflog," nbufs = %lu\n",ipcbuf_get_nbufs((ipcbuf_t *)(&dadain)));
  fprintf(outflog," bufsz = %lu\n",ipcbuf_get_bufsz(((ipcbuf_t *)(&dadain))));
  fprintf(outflog," nfull = %lu\n",ipcbuf_get_nfull(((ipcbuf_t *)(&dadain))));
  fprintf(outflog," nclear= %lu\n",ipcbuf_get_nclear(((ipcbuf_t *)(&dadain))));
  fprintf(outflog," write count= %lu\n",ipcbuf_get_write_count(((ipcbuf_t *)(&dadain))));
  fprintf(outflog," read count = %lu\n",ipcbuf_get_read_count(((ipcbuf_t *)(&dadain))));
  fprintf(outflog," nreaders = %d\n",ipcbuf_get_nreaders((ipcbuf_t *)(&dadain)));
  fprintf(outflog," last byte written = %lu\n",ipcio_tell(&dadain));
  fprintf(outflog," space left   = %lu [bytes]\n",ipcio_space_left(&dadain));
  fprintf(outflog," percent full = %f\n",ipcio_percent_full(&dadain));

  ////////////// initialize DADA buffer for reading data
  if(initDADA()!=0){
    fprintf(outflog,"initDADA returned in error!\n");
    ipcio_disconnect(&dadain);
    return -1;
  }
  ////////////////////////////


  //number of DMs
  NDM101=2;
  fprintf(stdout,"NDM101= %d\n",NDM101);fflush(stdout);

  //allocate
  MEMSIZEOFFSET=sizeof(int)*NCHANOUT*NDM101;
  HANDLE_ERROR(cudaHostAlloc((void **)&hostOFFSET,MEMSIZEOFFSET,cudaHostAllocDefault));
  HANDLE_ERROR(cudaMalloc((void **)&OFFSET,MEMSIZEOFFSET));
  HANDLE_ERROR(cudaHostAlloc((void **)&hostFULLFB,MEMSIZEFULLFB,cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void **)&timein0_host,MEMSIZEUTIME,cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void **)&timein1_host,MEMSIZEUTIME,cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void **)&powAVE_host,MEMSIZEPOWER,cudaHostAllocDefault));
  HANDLE_ERROR(cudaMalloc((void **)&powFULLFB,MEMSIZEFULLFB));
  HANDLE_ERROR(cudaMalloc((void **)&powAVE,MEMSIZEPOWER));
  HANDLE_ERROR(cudaMalloc((void **)&powTOT,MEMSIZEPOWER*NAVE));
  HANDLE_ERROR(cudaMalloc((void **)&timein0,MEMSIZETIME));
  HANDLE_ERROR(cudaMalloc((void **)&timein1,MEMSIZETIME));
  HANDLE_ERROR(cudaMalloc((void **)&utime0,MEMSIZEUTIME));
  HANDLE_ERROR(cudaMalloc((void **)&utime1,MEMSIZEUTIME));
  HANDLE_ERROR(cudaMalloc((void **)&freqout0,sizeof(cufftComplex)*NFREQ*NAVE*NBUNDLE));
  HANDLE_ERROR(cudaMalloc((void **)&freqout1,sizeof(cufftComplex)*NFREQ*NAVE*NBUNDLE));
  DM101_list=(float *)malloc(NDM101*sizeof(float));

  //tmphdr for ipcio_read, gets converted to hdr
  tmphdr0  = (char *)malloc(HDRLEN*sizeof(char));
  tmphdr1  = (char *)malloc(HDRLEN*sizeof(char));
  //buffers to hold a single frame of vdif data, for each pol
  data0=(unsigned char *)malloc(DATALEN*sizeof(unsigned char));
  data1=(unsigned char *)malloc(DATALEN*sizeof(unsigned char));
  //buffers to hold data to fill-in for skipped frames
  skip0=(unsigned char *)malloc(DATALEN*sizeof(unsigned char));
  skip1=(unsigned char *)malloc(DATALEN*sizeof(unsigned char));
  //buffers to hold next good frame of data when skipped frames found
  tmpdata0=(unsigned char *)malloc(DATALEN*sizeof(unsigned char));
  tmpdata1=(unsigned char *)malloc(DATALEN*sizeof(unsigned char));
	 

  fprintf(stdout,"Setting up FFT plan...\n");
  fflush(stdout);
  //setup FFT
  fprintf(outflog,"calculating FFT plan...\n");
  if(cufftPlan1d(&plan,TWONCHAN,CUFFT_R2C,NAVE*NBUNDLE) != CUFFT_SUCCESS){
    fprintf(outflog,"Error creating FFT plan!\n");
    return 1;
  }


  //set DMs
  DM101_list[0]=0.0;
  DM101_list[1]=119.3;//119.3 gives a dt of 1 sec across lower 39.3216 MHz band
  dm_count=NDM101;
  cudaMemcpyToSymbol(CNDM,&NDM101,sizeof(int));




  //setup OFFSET array
  // NCHANOUT is fastest changing index. Starts at highest freq
  // NDM is slower index. starts at 0
  fprintf(stdout,"Setting up offset array...\n");
  fprintf(outflog,"calculating offset array...\n");
  for(i=0;i<NDM101;i++){
    hostOFFSET[i*NCHANOUT]=0;//highest freq chan is the reference
    for(j=1;j<NCHANOUT;j++){
      dt=4.148808e3*DM101_list[i]*(pow(fch1-(j*CHANBW),-2)-pow(fch1,-2));//sec
      hostOFFSET[(i*NCHANOUT)+j]=(int)(dt/tchan);
    }
  }
  //copy OFFSET to device
  cudaMemcpy(OFFSET,hostOFFSET,MEMSIZEOFFSET,cudaMemcpyHostToDevice);
  //calc max_delay (should check that it is a multiple of NBUNDLE & NHALFCYC ?)
  max_delay=(int)((4.148808e3*DM101_list[NDM101-1]*(pow(fch1-((NCHANOUT)*CHANBW),-2)-pow(fch1,-2)))/tchan);//samples
  fprintf(outflog,"dm_count= %d  max_delay= %d\n",dm_count,max_delay);

  MEMSIZELEFTOVER=NCHANOUT*max_delay*sizeof(float);
  HANDLE_ERROR(cudaMalloc((void **)&powLEFTOVER,MEMSIZELEFTOVER));



  //allocate outDD
  unsigned int MEMSIZEOUTDD;
  NTIME_OUTDD=MAXNT-max_delay;
  fprintf(stdout,"NTIME_OUTDD= %d\n",NTIME_OUTDD);

  MEMSIZEOUTDD=dm_count*NTIME_OUTDD*sizeof(float);
  HANDLE_ERROR(cudaMalloc((void **)&outDD,MEMSIZEOUTDD));
  HANDLE_ERROR(cudaHostAlloc((void **)&host_outDD,MEMSIZEOUTDD,cudaHostAllocDefault));
  HANDLE_ERROR(cudaMalloc((void **)&sigDD,MEMSIZEOUTDD));
  HANDLE_ERROR(cudaHostAlloc((void **)&host_sigDD,MEMSIZEOUTDD,cudaHostAllocDefault));
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


  NTHREAD_STAT=1;
  NBLOCK_STAT=dm_count;
  dim3 thrdsDD(16,16);
  NTHREAD_DD=256;
  NBLOCK_DD_SQ=((int)sqrt(NDM101*NTIME_OUTDD/NTHREAD_DD))+1;
  dim3 blksDD(NBLOCK_DD_SQ,NBLOCK_DD_SQ);
  fprintf(outflog,"NTHREAD_STAT= %d  NBLOCK_STAT= %d\n",NTHREAD_STAT,NBLOCK_STAT);
  fprintf(outflog,"NTHREAD_DD  = %d  NBLOCK_DD_SQ  = %d\n",NTHREAD_DD,NBLOCK_DD_SQ);



  //initialize
  NSKIP0=0;
  NSKIP1=0;
  INDREAD=0;
  FLAGEODADA=1;
  NPARTLEFT=0;
  //initialize frame buffers with first vdif frame from DADA buffer
  fprintf(stdout,"initializing frame buffers...\n");
  fprintf(outflog,"initializing pol 0 & 1 single frame buffers...\n");
  fflush(outflog);
  int tmpflag;
  tmpflag=initframe01();
  while(tmpflag != 0){
    if (tmpflag ==2){
      fprintf(outflog,"initframe01 returned in error code 2!\n");
      return(2);
    }
    tmpflag=initframe01();
  }

  fprintf(stdout,"initializing skip frame data...\n");
  fflush(stdout);
  //initialize skip frame data
  if (initskip01() != 0){
    fprintf(outflog,"initskip01 returned in error!\n");
    return(2);
  }  


  //start time of filterbank
  tstart=getVDIFDMJD(&hdr0,FRAMESPERSEC); //[MJD]
  tsamp=NAVE*TWONCHAN/VLITERATE;          //[sec]
  fprintf(outflog,"tstart= %.15lf   tsamp= %le\n",tstart,tsamp);
  fprintf(outflog," frame full second= %d   frame number= %d\n",getVDIFFullSecond(&hdr0),getVDIFFrameNumber(&hdr0));
  fprintf(stdout,"tstart= %.15lf\n",tstart); fflush(stdout);


  ////////////////////////////////////////
  //initialize for PRESTO files:
  mjdstart=tstart;
  ntimes=0;
  timesystem=time(NULL);
  timeutc= *gmtime(&timesystem);
  hrnext=timeutc.tm_hour + 1;
  if(hrnext==24) hrnext=0;
  //set info file names	
  strftime(buffer,STRLEN,"%F_%T",&timeutc);
  sprintf(INFFILE0,"DM%.1f_%s",DM101_list[0],buffer);
  sprintf(INFFILE100,"DM%.1f_%s",DM101_list[1],buffer);
  //set and open time series files
  sprintf(TIMEFILE0,"%s.dat",INFFILE0);
  sprintf(TIMEFILE100,"%s.dat",INFFILE100);
  sprintf(SIGFILE0,"%s_SIGMA.dat",INFFILE0);
  sprintf(SIGFILE100,"%s_SIGMA.dat",INFFILE100);
  //File access is faster if a preexisting file is deleted before opening
  if(access(TIMEFILE0,F_OK)!=-1)   remove(TIMEFILE0);
  if(access(TIMEFILE100,F_OK)!=-1) remove(TIMEFILE100);
  if(access(INFFILE0,F_OK)!=-1)    remove(INFFILE0);
  if(access(INFFILE100,F_OK)!=-1)  remove(INFFILE100);
  if(access(SIGFILE0,F_OK)!=-1)    remove(SIGFILE0);
  if(access(SIGFILE100,F_OK)!=-1)  remove(SIGFILE100);
  ///
  if((outfDM0     =fopen(TIMEFILE0,"wb"))  <0){fprintf(stderr,"Unable to open %s\n",TIMEFILE0);  return(2);}
  if((outfDM100   =fopen(TIMEFILE100,"wb"))<0){fprintf(stderr,"Unable to open %s\n",TIMEFILE100);return(2);}
  if((outfDM0SIG  =fopen(SIGFILE0,"wb"))   <0){fprintf(stderr,"Unable to open %s\n",SIGFILE0);   return(2);}
  if((outfDM100SIG=fopen(SIGFILE100,"wb")) <0){fprintf(stderr,"Unable to open %s\n",SIGFILE100); return(2);}
  //close time series files
  fclose(outfDM0);
  fclose(outfDM100);
  fclose(outfDM0SIG);
  fclose(outfDM100SIG);
  //////////////////////////////////////


  //reset log file to be an hourly log file
  fclose(outflog);
  sprintf(LOGFILE,"%s_%s.log",OUTNAME,buffer);
  outflog = fopen(LOGFILE,"w");
  if (outflog<0) {
    fprintf(stderr,"Unable to open output log file: %s\n",LOGFILE);
    return(2);
  }


  int q=0;
  ///////////////////////////////////////////////////////////////////////////////////////////
  ///start looping here
  fprintf(outflog,"begin main loop...\n");
  fprintf(stdout,"begin main loop...\n");fflush(stdout);
  fflush(outflog);
  while(FLAGEODADA){
    //fill time buffer on host with float data from frame buffers
    if(filltimebufferHOST(timein0_host,timein1_host)){ 
      break;
    }
    

    //copy data in host memory to GPU
    HANDLE_ERROR(cudaMemcpy(utime0,timein0_host,MEMSIZEUTIME,cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(utime1,timein1_host,MEMSIZEUTIME,cudaMemcpyHostToDevice));


    //convert unsigned char time arrays to float arrays on GPU
    convertarray<<<NBLOCK_CONVERT,NTHREAD_CONVERT>>>(timein0,utime0);
    convertarray<<<NBLOCK_CONVERT,NTHREAD_CONVERT>>>(timein1,utime1);


    /////generate filterbank
    //FFT time data for each pol, in all bundles
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
      HANDLE_ERROR(cudaMemcpy(&powFULLFB[(FILL+(j*NCHANOUT))],&powAVE[(NSKIP+(j*NFREQ))],MEMSIZEFILTERBANK,cudaMemcpyDeviceToDevice));
    }


    //when powFULLFB filled:
    // 1 dedisp
    // 2 copy outDD to host and write to output file
    // 3 copy max_delay columns to start of powFULLFB, loop and refill w/ more data 
    loopcnt++;
    //is powFULLFB full? if so, dedisperse
    if(loopcnt==MAXLOOPCNT){
      //copy leftover end of FB array to holding buffer before BP+cal subtraction
      HANDLE_ERROR(cudaMemcpy(&powLEFTOVER[0],&powFULLFB[NTIME_OUTDD*NCHANOUT],MEMSIZELEFTOVER,cudaMemcpyDeviceToDevice));

      //subtract bandpass and cal signal 
      // calc and subtract bandpass in each on/off half-cycle to naturally subtract both BP and cal signal
      // 128 columns in each half-cycle (4 bundles)
      subtractbpCAL<<<NBLOCK_SUB,NTHREAD_SUB>>>((cufftReal *)powFULLFB);

      //dedisperse
      // *** need to replace with efficient dedispersion routines!
      fprintf(outflog,"calling dedisperseSTATS...\n");
      dedisperseSTATS<<<blksDD,thrdsDD>>>((cufftReal *)powFULLFB,outDD,sigDD,OFFSET); //needs to be (cufftReal *)powFULLFB ???

      //////////////////////////////////////////////////////////////////////////////
      //copy outDD to host and write to file
      cudaMemcpy(host_outDD,outDD,MEMSIZEOUTDD,cudaMemcpyDeviceToHost);
      cudaMemcpy(host_sigDD,sigDD,MEMSIZEOUTDD,cudaMemcpyDeviceToHost);
      for(i=0;i<dm_count;i++){
	if(i==0){
	  outfDM0=fopen(TIMEFILE0,"ab");
	  fwrite(&host_outDD[0],NTIME_OUTDD*sizeof(float),1,outfDM0);
	  fclose(outfDM0);	  
	  outfDM0SIG=fopen(SIGFILE0,"ab");
	  fwrite(&host_sigDD[0],NTIME_OUTDD*sizeof(float),1,outfDM0SIG);
	  fclose(outfDM0SIG);	  
	}
	else{
	  outfDM100=fopen(TIMEFILE100,"ab");
	  fwrite(&host_outDD[NTIME_OUTDD],NTIME_OUTDD*sizeof(float),1,outfDM100);
	  fclose(outfDM100);
	  outfDM100SIG=fopen(SIGFILE100,"ab");
	  fwrite(&host_sigDD[NTIME_OUTDD],NTIME_OUTDD*sizeof(float),1,outfDM100SIG);
	  fclose(outfDM100SIG);
	}
      }
      ntimes+=NTIME_OUTDD;
      //////////////////////////////////////////////////////////////////////////////

 
      //copy leftover buffer to beginning of powFULLFB
      cudaMemcpy(&powFULLFB[0],&powLEFTOVER[0],MEMSIZELEFTOVER,cudaMemcpyDeviceToDevice);

      //set FILL location accordingly
      FILL=max_delay*NCHANOUT;
      //set loopcnt accordingly
      loopcnt=max_delay/NBUNDLE;


      //check time. has next hour started? 
      //  if so: write PRESTO info file, open new time series files & reset ntimes, update mjdstart
      timesystem=time(NULL);
      timeutc= *gmtime(&timesystem);
      if(timeutc.tm_hour == hrnext){
	//write info files
	writeprestoheader(INFFILE0,mjdstart,ntimes,tsamp,DM101_list[0],fchlo,bandw,NCHANOUT,CHANBW);
	writeprestoheader(INFFILE100,mjdstart,ntimes,tsamp,DM101_list[1],fchlo,bandw,NCHANOUT,CHANBW);

	//set new info file names	
	strftime(buffer,STRLEN,"%F_%T",&timeutc);
	sprintf(INFFILE0,"DM%.1f_%s",DM101_list[0],buffer);
	sprintf(INFFILE100,"DM%.1f_%s",DM101_list[1],buffer);

	//set and open new time series files
	sprintf(TIMEFILE0,"%s.dat",INFFILE0);
	sprintf(TIMEFILE100,"%s.dat",INFFILE100);
	sprintf(SIGFILE0,"%s_SIGMA.dat",INFFILE0);
	sprintf(SIGFILE100,"%s_SIGMA.dat",INFFILE100);
	if((outfDM0     =fopen(TIMEFILE0,"wb"))  <0){fprintf(stderr,"Unable to open %s\n",TIMEFILE0);  return(2);}
	if((outfDM100   =fopen(TIMEFILE100,"wb"))<0){fprintf(stderr,"Unable to open %s\n",TIMEFILE100);return(2);}
	if((outfDM0SIG  =fopen(SIGFILE0,"wb"))   <0){fprintf(stderr,"Unable to open %s\n",SIGFILE0);   return(2);}
	if((outfDM100SIG=fopen(SIGFILE100,"wb")) <0){fprintf(stderr,"Unable to open %s\n",SIGFILE100); return(2);}
	//close time series files
	fclose(outfDM0);
	fclose(outfDM100);
	fclose(outfDM0SIG);
	fclose(outfDM100SIG);

	//close old log file and open new
	fprintf(outflog,"Log file closed at %s\n",buffer);
	fclose(outflog);
	sprintf(LOGFILE,"%s_%s.log",OUTNAME,buffer);
	outflog = fopen(LOGFILE,"w");
	if (outflog<0) {
	  fprintf(stderr,"Unable to open output log file: %s\n",LOGFILE);
	  return(2);
	}
	fprintf(outflog,"New log file opened at %s\n",buffer);

	//update variables
	mjdstart=mjdstart+((double)ntimes*tsamp/86400.0);
	ntimes=0;
	hrnext++;
	if(hrnext==24) hrnext=0;
      }


      fflush(outflog);
      q++;
    }
  
  }//end while loop


  //close connection to input DADA buffer
  if(ipcio_close(&dadain) < 0){
    fprintf(outflog,"Error closing input DADA buffer!\n");
    return(2);
  }
  if(ipcio_disconnect(&dadain) < 0){
    fprintf(outflog,"Error disconnecting from input DADA buffer!\n");
    return(2);
  }


  fprintf(outflog,"\nfinished!\n");fflush(outflog);
  fclose(outflog);
  return(0);
}//end of main


//initialize DADA buffer for reading by skipping first NFRAMEDADAINIT frames of each pol
int initDADA()
{
  int i;
  for(i=0;i<NFRAMEDADAINIT;i++){
    /////////////////////////////////////////////////////////
    //read a frame header (assumed pol 0 but may not be)
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
    //read a frame data -- ipcio_read requires data0 be char *!
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
    //////////////////////////////////////////////////////
    //read a frame header (assumed pol 1 but may not be)
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
    //read a frame data -- ipcio_read requires data0 be char *!
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
    
  }
  return 0;
}


int initframePOL0(){
  //read frame header
  //fprintf(outflog,"    reading header...\n");fflush(outflog);
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
      fprintf(outflog,"EOD while trying to read data1\n");
      return(1);
    }
    else{
      fprintf(outflog,"ERROR while trying to read data1!!!\n");
      return(2);
    }
  }
  return 0;
}


//check for skipped frames, pol 0
//  check if frame number agrees with expected value
int checkframeskip0(){
  int fnum0,fsec0;
  fnum0=getVDIFFrameNumber(&hdr0);
  fsec0=getVDIFFullSecond(&hdr0);
  if((fnum0 != EXPECTEDFRAMENUM0) || (fsec0 != EXPECTEDFRAMESEC0)){
    //set # skipped frames
    NSKIP0=fnum0-EXPECTEDFRAMENUM0+((fsec0-EXPECTEDFRAMESEC0)*FRAMESPERSEC);
    fprintf(outflog,"FRAME SKIP 0!!! EXPECTED FRAME0 SEC NUMBER= %d %d, BUT READ %d %d! (NSKIP0= %d, loopcnt= %d)\n",EXPECTEDFRAMESEC0,EXPECTEDFRAMENUM0,fsec0,fnum0,NSKIP0,loopcnt);
    //copy frame data to tmpdata buffer
    memcpy(&tmpdata0[0],&data0[0],DATALEN);
    //copy skip frame values to data buffer
    memcpy(&data0[0],&skip0[0],DATALEN);
  }
  //update expected frame number and second
  EXPECTEDFRAMENUM0=(fnum0+1) % FRAMESPERSEC;
  EXPECTEDFRAMESEC0=fsec0; 
  if(EXPECTEDFRAMENUM0==0)
    EXPECTEDFRAMESEC0++;
  return 0;
}

//check for skipped frames, pol 1
//  check if frame number agrees with expected value
int checkframeskip1(){
  int fnum1,fsec1;
  fnum1=getVDIFFrameNumber(&hdr1);
  fsec1=getVDIFFullSecond(&hdr1);
  if((fnum1 != EXPECTEDFRAMENUM1) || (fsec1 != EXPECTEDFRAMESEC1)){
    //set # skipped frames
    NSKIP1=fnum1-EXPECTEDFRAMENUM1+((fsec1-EXPECTEDFRAMESEC1)*FRAMESPERSEC);
    fprintf(outflog,"FRAME SKIP 1!!! EXPECTED FRAME1 SEC NUMBER= %d %d, BUT READ %d %d! (NSKIP1= %d, loopcnt= %d)\n",EXPECTEDFRAMESEC1,EXPECTEDFRAMENUM1,fsec1,fnum1,NSKIP1,loopcnt);
    //copy frame data to tmpdata buffer
    memcpy(&tmpdata1[0],&data1[0],DATALEN);
    //copy skip frame values to data buffer
    memcpy(&data1[0],&skip1[0],DATALEN);
  }
  //update expected frame number and second
  EXPECTEDFRAMENUM1=(fnum1+1) % FRAMESPERSEC;
  EXPECTEDFRAMESEC1=fsec1; 
  if(EXPECTEDFRAMENUM1==0)
    EXPECTEDFRAMESEC1++;
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
  int id0,id1,fnum0,fnum1,ncal=0,flag,fsec0,fsec1;
  //read pol 0
  flag=initframePOL0();
  if(flag) return flag;    
  //check pol 0, if not then try next frame
  id0=getVDIFThreadID(&hdr0);
  if(id0 != 0){
    fprintf(outflog,"id0= %d, VDIF Threads misaligned, skipping forward 1 frame\n",id0);fflush(outflog);
    flag=initframePOL0();
    if(flag) return flag;    
  }
  //double check
  id0=getVDIFThreadID(&hdr0);
  if(id0 != 0){
    fprintf(outflog,"ERROR, initframe01! VDIF Thread ID0= %d, != 0. Abort!\n",id0);fflush(outflog);
    return 1;
  }

  //read pol 1
  flag=initframePOL1();
  if(flag) return flag;    
  //check pol 1
  id1=getVDIFThreadID(&hdr1);
  if(id1 != 1){
    fprintf(outflog,"VDIF Thread ID1= %d, != 1. Abort!\n",id1);
    return 1;
  }

  //check agreement of pol frame numbers and seconds
  fnum0=getVDIFFrameNumber(&hdr0);
  fsec0=getVDIFFullSecond(&hdr0);
  fnum1=getVDIFFrameNumber(&hdr1);
  fsec1=getVDIFFullSecond(&hdr1);
  if((fnum0 != fnum1) || (fsec0 != fsec1)){
    fprintf(outflog,"ERROR! polarization VDIF frame seconds or numbers do not agree! 0: %d %d  !=  1: %d %d\n",fsec0,fnum0,fsec1,fnum1);
    return(2);
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
    if(flag) return flag;
    fnum0=getVDIFFrameNumber(&hdr0);
    ncal++;
  }

  //re-check agreement of pol frame numbers and seconds
  fsec0=getVDIFFullSecond(&hdr0);
  fnum1=getVDIFFrameNumber(&hdr1);
  fsec1=getVDIFFullSecond(&hdr1);
  if((fnum0 != fnum1) || (fsec0 != fsec1)){
    fprintf(outflog,"ERROR! polarization VDIF frame seconds or numbers do not agree! 0: %d %d  !=  1: %d %d\n",fsec0,fnum0,fsec1,fnum1);
    return(2);
  }

  //set expected frame numbers and secs for next read (to check for frame skips)
  EXPECTEDFRAMENUM0=(fnum0+1) % FRAMESPERSEC;
  EXPECTEDFRAMENUM1=(fnum1+1) % FRAMESPERSEC; 
  EXPECTEDFRAMESEC0=fsec0;
  EXPECTEDFRAMESEC1=fsec1; 
  if(EXPECTEDFRAMENUM0==0)
    EXPECTEDFRAMESEC0++;
  if(EXPECTEDFRAMENUM1==0)
    EXPECTEDFRAMESEC1++;

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
  }
  
  if(NPARTFILL){
    memcpy(&timein0[IFILL],&data0[0],NPARTFILL);
    memcpy(&timein1[IFILL],&data1[0],NPARTFILL);
    IFILL+=NPARTFILL;
  }  

  //returns with full or partially filled data0[] & data1[]  
  return(0);
}



//from SIGPROC send_stuff.c
int strings_equal (char *string1, char *string2)
{
  if (!strcmp(string1,string2)) return 1;
  else return 0;
}


//////////////////////
void print_usage(){
  fprintf(stderr,"***************************************************************************\n");
  fprintf(stderr,"This program reads time series data from a DADA buffer with VDIF format data and\n");
  fprintf(stderr,"  generates a filterbank, dedisperses at select DM trials, and outputs time series data.\n\n");
  fprintf(stderr,"  Usage:\n./vdifDADA2timeseries -d:\t    use default values (unless other values given)\n");
  fprintf(stderr,"\t             -h:            print this help message.\n");
  fprintf(stderr,"\t             -k (hex):      hexadecimal key of DADA buffer to read VDIF data from,\n");
  fprintf(stderr,"\t             -o (string):   base name of ouput files to write,\n");
  fprintf(stderr,"\t             -nave (int):   number of time samples to average over,\n");
  fprintf(stderr,"\t             -nchan (int):  number of frequency channels,\n");
  fprintf(stderr,"\t             -nb (int):     number of bundles of size (2*2*nchan*nave) to send in 1 memcpy\n");
  fprintf(stderr,"\n");
  fprintf(stderr,"  DEFAULTS:\n");
  fprintf(stderr,"\t             -k     %x\n",DADA_DEFAULT_INPUT_KEY);
  fprintf(stderr,"\t             -o     CRABtestDEDISP_DADA\n");
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
    keyin = DADA_DEFAULT_INPUT_KEY;
    strcpy(OUTNAME,"testTIMESERIES");
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
    else if(strings_equal(argv[i],"-k")){
      i++;
      sscanf(argv[i],"%x",&keyin);
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


//search for peaks in dedispersed array
__global__ void calcstats(float *outDD, int *FRB, float *ave, float *sigma, float *peak){
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

    for(j=0;j<CMAXNT;j++)
      FULLFB[CHAN_ind+(j*CNCHANOUT)]-=ave;
  }
  return;
}


__global__ void subtractbpCAL(cufftReal *FULLFB){
  int i,j,istart,tid;
  int row,col,colmin,colmax,npts;
  float sum,sum2,ave,sig,HI,LO;

  tid=threadIdx.x+(blockIdx.x*blockDim.x);
  col=(int)(tid/CNCHANOUT);
  row=tid-(col*CNCHANOUT);
  if((col<CNHALFCYC) && (row<CNCHANOUT)){
    colmin=128*col;
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



__global__ void dedisperseSTATS(cufftReal *FULLFB, float *outDD, float *sigDD, int *OFFSET){
  int i,DMidx,TIMEidx,off,FBidx;
  int x=threadIdx.x+(blockIdx.x*blockDim.x);
  int y=threadIdx.y+(blockIdx.y*blockDim.y);
  float sum,sum2,ave,sigma;
  int idx=x+(y*blockDim.x*gridDim.x);

  if(idx<(CNTIME_OUTDD*CNDM)){
    DMidx=(int)(idx/CNTIME_OUTDD);
    TIMEidx=idx-(DMidx*CNTIME_OUTDD);

    sum=0.0;
    for(i=0;i<CNCHANOUT;i++){
      off=OFFSET[(DMidx*CNCHANOUT)+i];
      FBidx=i+((off+TIMEidx)*CNCHANOUT);
      sum+=FULLFB[FBidx];
    }
    ave=sum/CNCHANOUT;
    sum=sum2=0.0;
    for(i=0;i<CNCHANOUT;i++){
      off=OFFSET[(DMidx*CNCHANOUT)+i];
      FBidx=i+((off+TIMEidx)*CNCHANOUT);
      sum+=(FULLFB[FBidx]-ave);
      sum2+=((FULLFB[FBidx]-ave)*(FULLFB[FBidx]-ave));
    }
    sigma=(sum2-(sum*sum))/(CNCHANOUT*(CNCHANOUT-1));
    outDD[idx]=ave;
    sigDD[idx]=sqrt(sigma);
  }

  return;
}

int writeprestoheader(char *fbasename, double mjdstart, int ntimes, double tsamp, float dm, double fchanlo, double bw, int nchan, double chanbw)
{
  FILE *fout;
  char fname[STRLEN];
  sprintf(fname,"%s.inf",fbasename);
  if((fout=fopen(fname,"w"))==NULL){
    fprintf(outflog,"Error! Couldn't open file %s!\n",fname);
    return 1;
  }
  fprintf(fout," Data file name without suffix          = %s\n",fbasename);
  fprintf(fout," Telescope used                         = VLA\n");
  fprintf(fout," Instrument used                        = VLITE-FAST\n");
  fprintf(fout," Object being observed                  = Unknown\n");
  fprintf(fout," J2000 Right Ascension (hh:mm:ss.ssss)  = 00:00:00.0000\n");
  fprintf(fout," J2000 Declination     (dd:mm:ss.ssss)  = 00:00:00.0000\n");
  fprintf(fout," Data observed by                       = vdifDADA2timeseries\n");
  fprintf(fout," Epoch of observation (MJD)             = %.15lf\n",mjdstart);
  fprintf(fout," Barycentered?                          = 1\n");
  fprintf(fout," Number of bins in the time series      = %d\n",ntimes);
  fprintf(fout," Width of each time series bin (sec)    = %.9lf\n",tsamp);
  fprintf(fout," Any breaks in the data? (1=yes, 0=no)  = 0\n");
  fprintf(fout," Type of observation (EM band)          = Radio\n");
  fprintf(fout," Beam diameter (arcsec)                 = 3600\n");
  fprintf(fout," Dispersion measure (cm-3 pc)           = %f\n",dm);
  fprintf(fout," Central freq of low channel (MHz)      = %f\n",fchanlo);
  fprintf(fout," Total bandwidth (MHz)                  = %f\n",bw);
  fprintf(fout," Number of channels                     = %d\n",nchan);
  fprintf(fout," Channel bandwidth (MHz)                = %f\n",chanbw);
  fprintf(fout," Data analyzed by                       = vdifDADA2timeseries\n");
  fprintf(fout," Any additional notes:\n    \n\n");
  fclose(fout);
  return 0;
}


