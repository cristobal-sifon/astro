/******************************************************************************/
/*
 * NAME:
 *   predict_sync
 *
 * PURPOSE:
 *   Read predictions of synchtron emission and return sky intensity.
 *
 * CALLING SEQUENCE:
 *   predict_sync gall galb nu=nu infile=infile \
 *    outfile=outfile interp=interp \
 *    noloop=noloop verbose=verbose ipath=ipath units=units
 *
 * OPTIONAL INPUTS:
 *   gall:       Galactic longitude(s) in degrees
 *   galb:       Galactic latitude(s) in degrees
 *   nu:         Frequency in GHz.  If this is a vector, it must be the same
 *               dimension as GALL and GALB.  If this is a scalar, then it
 *               applies to all positions GALL, GALB.
 *   infile:     If set, then read "gall" and "galb" from this file.  If "nu"
 *               is not set, then "nu" is read as the 3rd column of this same
 *               file.
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
 *               environment variable $DUST_DIR/map, or to the current
 *               directory.
 *   units:      Units for output values:
 *               'MJy'   : MJy/sr (default)
 *               'microK' : brightness (antenna) temperature [micro-Kelvin]
 *               'thermo' : thermodynamic temperature [micro-Kelvin]
 *                          assuming T(CMB) = 2.73 K
 *
 * COMMENTS:
 *   These predictions are based upon the following radio surveys:
 *     408 MHz - Haslam et al. 1981, A&A, 100, 209
 *     1420 MHz - Reich & Reich 1986, A&AS, 63, 205
 *     2326 MHz - Jonas, Baart, & Nicolson 1998 MNRAS, 297, 977
 *   A detailed description of this data product may be found in:
 *   "Extrapolation of the Haslam 408 MHz survey to CMBR Frequencies",
 *   by D. P. Finkbeiner, & M. Davis, in preparation.
 *
 *   Either the coordinates GALL, GALB and NU must be set, or their values
 *   must exist in the file INFILE.  Output is written to the variable VALUE
 *   and/or the file OUTFILE.
 *
 * EXAMPLES:
 *   Read the predicted synchtrotron emission at Galactic (l,b)=(12,+34.5)
 *   interpolating from the nearest 4 pixels, and return result in VALUE.
 *   > value = predict_sync(12, 34.5, nu=3000, /interp)
 *
 * DATA FILES:
 *   Haslam_clean_ngp.fits
 *   Haslam_clean_sgp.fits
 *   Synch_Beta_ngp.fits
 *   Synch_Beta_sgp.fits
 *
 * REVISION HISTORY:
 *   Written by D. Schlegel, 10 Apr 1999, Princeton
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
#include "subs_predict.h"

/******************************************************************************/
void main
  (int      argc,
   char  *  ppArgv[],
   char  *  ppEnvp[])
{
   int      ic;
   int      qInterp;
   int      qVerbose;
   int      qNoloop;
   int      iGal;
   int      nGal;
   int      nCol;
   float    tmpl;
   float    tmpb;
   float *  pGall = NULL;
   float *  pGalb = NULL;
   float *  pNu = NULL;
   float *  pInu;
   float *  pData;

   FILE  *  pFILEout;
   static char pPrivW[] = "w";

   /* Declarations for keyword input values */
   int      ienv;
   int      nkey;
   float    nuval = 0.0;
   char  *  pTemp;
   char  *  pKeyname;
   char  *  pKeyval;
   char  *  pUnitsName;
   char  *  pInFile = NULL;
   char  *  pOutName = NULL;
   char  *  pIPath   = NULL;
   char     pString1[13];
   char     pDefPath[] = "./";
   const char pDUST_DIR[] = "DUST_DIR";
   const char     pEq[] = "=";
   const char pText_mapdir[] = "/maps/";

   /* Declarations for command-line keyword names */
   const char pText_ipath[] = "ipath";
   const char pText_infile[] = "infile";
   const char pText_outfile[] = "outfile";
   const char pText_nu[] = "nu";
   const char pText_units[] = "units";
   const char pText_interp[] = "interp";
   const char pText_noloop[] = "noloop";
   const char pText_verbose[] = "verbose";
   char     pText_MJy[] = "MJy";

   /* Set defaults */
   pIPath = pDefPath;
   pUnitsName = pText_MJy;
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

         if (strcmp(pKeyname,pText_nu) == 0)
          sscanf(pKeyval, "%f", &nuval);

         if (strcmp(pKeyname,pText_infile) == 0) pInFile = pKeyval;

         if (strcmp(pKeyname,pText_outfile) == 0) pOutName = pKeyval;

         if (strcmp(pKeyname,pText_units) == 0) pUnitsName = pKeyval;

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
      if (nuval == 0.0) {
         asciifile_read_colmajor(pInFile, 3, &nGal, &nCol, &pData);
         pGall = pData;
         pGalb = pData + nGal;
         pNu = pData + 2*nGal;
      } else {
         asciifile_read_colmajor(pInFile, 2, &nGal, &nCol, &pData);
         pGall = pData;
         pGalb = pData + nGal;
      }
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

   /* If "nu" was specified on the command line, then set frequency of
      all points to this value; this will supersede any in an input file. */
   if (nuval != 0.0) {
      if (pNu == NULL) pNu = ccvector_build_(nGal);
      for (iGal=0; iGal < nGal; iGal++) pNu[iGal] = nuval;
   }

   pInu = predict_sync(nGal, pGall, pGalb, pNu, pIPath,
    pUnitsName, qInterp, qNoloop, qVerbose);

   /* If no output file, then output to screen */
   if (pOutName != NULL) pFILEout = fopen(pOutName, pPrivW);
   else pFILEout = stdout;
   sprintf(pString1, "Inu(%s)", pUnitsName);
   fprintf(pFILEout, " l(deg)  b(deg)  nu(GHz)      %-12.12s\n", pString1);
   fprintf(pFILEout, " ------- ------- ------------ ------------\n");
   for (iGal=0; iGal < nGal; iGal++) {
      fprintf(pFILEout, "%8.3f %7.3f %12.5e %12.5e\n",
       pGall[iGal], pGalb[iGal], pNu[iGal], pInu[iGal]);
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
