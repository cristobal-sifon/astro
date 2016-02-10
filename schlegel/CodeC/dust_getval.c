/******************************************************************************/
/*
 * NAME:
 *   dust_getval
 *
 * PURPOSE:
 *   Read values from BH files or our dust maps.
 *
 *   Either the coordinates "gall" and "galb" must be set, or these coordinates
 *   must exist in the file "infile".  Output is written to standard output
 *   or the file "outfile".
 *
 * CALLING SEQUENCE:
 *   dust_getval gall galb map=map ipath=ipath interp=interp noloop=noloop \
 *    infile=infile outfile=outfile verbose=verbose
 *
 * OPTIONAL INPUTS:
 *   gall:       Galactic longitude(s) in degrees
 *   galb:       Galactic latitude(s) in degrees
 *   map:        Set to one of the following (default is "Ebv"):
 *               I100: 100-micron map in MJy/Sr
 *               X   : X-map, temperature-correction factor
 *               T   : Temperature map in degrees Kelvin for n=2 emissivity
 *               Ebv : E(B-V) in magnitudes
 *               mask: Mask values
 *   infile:     If set, then read "gall" and "galb" from this file
 *   outfile:    If set, then write results to this file
 *   interp:     Set this flag to "y" to return a linearly interpolated value
 *               from the 4 nearest pixels.
 *               This is disabled if map=='mask'.
 *   noloop:     Set this flag to "y" to read entire image into memory
 *               rather than reading pixel values for one point at a time.
 *               This is a faster option for reading a large number of values,
 *               but requires reading up to a 64 MB image at a time into
 *               memory.  (Actually, the smallest possible sub-image is read.)
 *   verbose:    Set this flag to "y" for verbose output, printing pixel
 *               coordinates and map values
 *   ipath:      Path name for dust maps; default to path set by the
 *               environment variable $DUST_DIR/maps, or to the current
 *               directory.
 *
 * EXAMPLES:
 *   Read the reddening value E(B-V) at Galactic (l,b)=(12,+34.5),
 *   interpolating from the nearest 4 pixels, and output to the screen:
 *   % dust_getval 12 34.5 interp=y
 *
 *   Read the temperature map at positions listed in the file "dave.in",
 *   interpolating from the nearest 4 pixels, and output to file "dave.out".
 *   The path name for the temperature maps is "/u/schlegel/".
 *   % dust_getval map=T ipath=/u/schlegel/ interp=y \
 *     infile=dave.in outfile=dave.out 
 *
 * DATA FILES FOR SFD MAPS:
 *   SFD_dust_4096_ngp.fits
 *   SFD_dust_4096_sgp.fits
 *   SFD_i100_4096_ngp.fits
 *   SFD_i100_4096_sgp.fits
 *   SFD_mask_4096_ngp.fits
 *   SFD_mask_4096_sgp.fits
 *   SFD_temp_ngp.fits
 *   SFD_temp_sgp.fits
 *   SFD_xmap_ngp.fits
 *   SFD_xmap_sgp.fits
 *
 * DATA FILES FOR BH MAPS:
 *   hinorth.dat
 *   hisouth.dat
 *   rednorth.dat
 *   redsouth.dat
 *
 * REVISION HISTORY:
 *   Written by D. Schlegel, 19 Jan 1998, Durham
 *   5-AUG-1998 Modified by DJS to read a default path from an environment
 *              variable $DUST_DIR/map.
 */
/******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "interface.h"
#include "subs_inoutput.h"
#include "subs_common_string.h"
#include "subs_fits.h"
#include "subs_asciifile.h"
#include "subs_memory.h"
#include "subs_lambert.h"

/******************************************************************************/
void main
  (int      argc,
   char  *  ppArgv[],
   char  *  ppEnvp[])
{
   int      ic;
   int      imap;
   int      ii;
   int      qInterp;
   int      qVerbose;
   int      qNoloop;
   int      iGal;
   int      nGal;
   int      nCol;
   int      bitval[8];
   float    tmpl;
   float    tmpb;
   float *  pGall = NULL;
   float *  pGalb = NULL;
   float *  pMapval;
   float *  pData;
   FILE  *  pFILEout;
   static char pPrivW[] = "w";

   /* Declarations for keyword input values */
   int      ienv;
   int      nkey;
   char  *  pTemp;
   char  *  pKeyname;
   char  *  pKeyval;
   char  *  pMapName;
   char  *  pInFile = NULL;
   char  *  pOutName = NULL;
   char  *  pIPath   = NULL;
   char     pDefPath[] = "./";
   char     pDefMap[] = "Ebv";
   const char pDUST_DIR[] = "DUST_DIR";
   const char     pEq[] = "=";
   const char pText_mask[] = "mask";
   const char pText_mapdir[] = "/maps/";

   /* Declarations for command-line keyword names */
   const char pText_ipath[] = "ipath";
   const char pText_infile[] = "infile";
   const char pText_outfile[] = "outfile";
   const char pText_map[] = "map";
   const char pText_interp[] = "interp";
   const char pText_noloop[] = "noloop";
   const char pText_verbose[] = "verbose";

   /* Bit mask string values */
   char ppBitname[][8] = {
    "       " , "       ",
    "       " , "       ",
    "OK     " , "asteroi",
    "OK     " , "glitch ",
    "OK     " , "source ",
    "OK     " , "no_list",
    "OK     " , "big_obj",
    "OK     " , "no_IRAS" };

   /* Declarations for data file names */
   char     pFileN[MAX_FILE_NAME_LEN];
   char     pFileS[MAX_FILE_NAME_LEN];
   struct   mapParms {
      char *   pName;
      char *   pFile1;
      char *   pFile2;
   } ppMapAll[] = {
     { "Ebv" , "SFD_dust_4096_ngp.fits", "SFD_dust_4096_sgp.fits" },
     { "I100", "SFD_i100_4096_ngp.fits", "SFD_i100_4096_sgp.fits" },
     { "X"   , "SFD_xmap_ngp.fits"     , "SFD_xmap_sgp.fits"      },
     { "T"   , "SFD_temp_ngp.fits"     , "SFD_temp_sgp.fits"      },
     { "mask", "SFD_mask_4096_ngp.fits", "SFD_mask_4096_sgp.fits" }
   };
   const int nmap = sizeof(ppMapAll) / sizeof(ppMapAll[0]);

   /* Set defaults */
   pIPath = pDefPath;
   pMapName = pDefMap;
   qInterp = 0; /* no interpolation */
   qVerbose = 0; /* not verbose */
   qNoloop = 0; /* do not read entire image into memory */

   /* Override default path by value in the environment variable DUST_DIR */
   for (ienv=0; ppEnvp[ienv] != 0; ienv++) {
      if (strcmp(pDUST_DIR,strtok(ppEnvp[ienv],pEq))==0 ) {
         pIPath = strcat( strtok(NULL,pEq), pText_mapdir );
      }
   }

   nkey = 0;
   for (ic=1; ic < argc; ic++) {
      /* Check if this argument is a keyword */
      if ((pTemp=strchr(ppArgv[ic],'=')) != NULL) {
         nkey++;
         pKeyname = ppArgv[ic];
         pKeyval = pTemp + 1;
         pTemp[0] = '\0'; /* replace equals with NULL to terminate string */

         if (strcmp(pKeyname,pText_infile) == 0) pInFile = pKeyval;

         if (strcmp(pKeyname,pText_outfile) == 0) pOutName = pKeyval;

         if (strcmp(pKeyname,pText_map) == 0) pMapName = pKeyval;

         if (strcmp(pKeyname,pText_ipath) == 0) pIPath = pKeyval;

         if (strcmp(pKeyname,pText_interp) == 0) {
            if (strchr(pKeyval,'y') != NULL || strchr(pKeyval,'Y') != NULL)
             qInterp = 1; /* do interpolation */
         }

         if (strcmp(pKeyname,pText_noloop) == 0) {
            if (strchr(pKeyval,'y') != NULL || strchr(pKeyval,'Y') != NULL)
             qNoloop=1; /* read entire image into memory */
         }

         if (strcmp(pKeyname,pText_verbose) == 0) {
            if (strchr(pKeyval,'y') != NULL || strchr(pKeyval,'Y') != NULL)
             qVerbose = 1; /* do interpolation */
         }

      }
   }

   /* If no input coordinate file, then read coordinates from either
    * the command line or by query */
   if (pInFile != NULL) {
      asciifile_read_colmajor(pInFile, 2, &nGal, &nCol, &pData);
      pGall = pData;
      pGalb = pData + nGal;
   } else {
      if (argc-nkey > 2) {
         sscanf(ppArgv[1], "%f", &tmpl);
         sscanf(ppArgv[2], "%f", &tmpb);
      } else {
         printf("Galactic longitude (degrees):\n");
         scanf("%f", &tmpl);
         printf("Galactic latitude (degrees):\n");
         scanf("%f", &tmpb);
      }
      nGal = 1;
      pGall = ccvector_build_(nGal);
      pGalb = ccvector_build_(nGal);
      pGall[0] = tmpl;
      pGalb[0] = tmpb;
   }

   /* Determine the file names to use */
   for (imap=0; imap < nmap; imap++) {
      if (strcmp(pMapName,ppMapAll[imap].pName) == 0) {
         sprintf(pFileN, "%s/%s", pIPath, ppMapAll[imap].pFile1);
         sprintf(pFileS, "%s/%s", pIPath, ppMapAll[imap].pFile2);
      }
   }

   /* Disable interpolation if reading the mask */
   if (strcmp(pMapName,pText_mask) == 0) qInterp = 0;

   /* Read values from FITS files in Lambert projection */
   pMapval = lambert_getval(pFileN, pFileS, nGal, pGall, pGalb,
    qInterp, qNoloop, qVerbose);

   /* If no output file, then output to screen */
   if (pOutName != NULL) pFILEout = fopen(pOutName, pPrivW);
   else pFILEout = stdout;
   for (iGal=0; iGal < nGal; iGal++) {
      fprintf(pFILEout, "%8.3f %7.3f", pGall[iGal], pGalb[iGal]);
      if (strcmp(pMapName,pText_mask) == 0) {
         /* Translate mask bits */
         for (ii=0; ii < 8; ii++)
          bitval[ii] = ( (int)pMapval[iGal] & (int)pow(2,ii) ) > 0;
         fprintf(pFILEout, "  %1dhcons", bitval[0]+2*bitval[1]);
         for (ii=2; ii < 8; ii++) {
            fprintf(pFILEout, " %7s", ppBitname[bitval[ii]+2*ii]);
         }
         fprintf(pFILEout, "\n");
      } else {
         fprintf(pFILEout, " %12.5f\n", pMapval[iGal]);
      }
   }
   if (pOutName != NULL) fclose(pFILEout);

   if (pInFile != NULL) {
      ccfree_((void **)&pData);
   } else {
      ccfree_((void **)&pGall);
      ccfree_((void **)&pGalb);
   }
}
/******************************************************************************/
