/******************************************************************************/
/*
 * Program to read values from Lambert files
 *
 * Syntax: lamgetval FileNorth FileSouth qInterp gal_lon gal_lat
 *
 * D Schlegel -- Durham -- ANSI C
 * Jan 1998  DJS  Created
 */
/******************************************************************************/
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "interface.h"
#include "subs_inoutput.h"
#include "subs_common_string.h"
#include "subs_fits.h"
#include "subs_lambert.h"

/******************************************************************************/
void main
  (int      argc,
   char  *  argv[],
   char  *  envp[])
{
   int      qInterp;
   int      qNoloop;
   int      qVerbose;
   int      nGal;
   float    tmpl;
   float    tmpb;
   float    pGall[1];
   float    pGalb[1];
   float *  pMapval;
   char     pFileN[MAX_FILE_NAME_LEN];
   char     pFileS[MAX_FILE_NAME_LEN];

   if (argc > 1) {
      strcpy(pFileN, argv[1]);
   } else {
      printf("Northern FITS file:\n");
      scanf("%s", pFileN);
   }

   if (argc > 2) {
      strcpy(pFileS, argv[2]);
   } else {
      printf("Southern FITS file:\n");
      scanf("%s", pFileS);
   }

   if (argc > 3) {
      sscanf(argv[3], "%d", &qInterp);
   } else {
      printf("Interpolate (0=no,1=yes)?\n");
      scanf("%d", &qInterp);
   }

   if (argc > 4) {
      sscanf(argv[4], "%f", &tmpl);
   } else {
      printf("Galactic longitude (degrees):\n");
      scanf("%f", &tmpl);
   }

   if (argc > 5) {
      sscanf(argv[5], "%f", &tmpb);
   } else {
      printf("Galactic latitude (degrees):\n");
      scanf("%f", &tmpb);
   }

   nGal = 1;
   pGall[0] = tmpl;
   pGalb[0] = tmpb;
   qNoloop = 0;
   qVerbose = 0;
   pMapval = lambert_getval(pFileN, pFileS, nGal, pGall, pGalb,
    qInterp, qNoloop, qVerbose);

   printf("%f\n", pMapval[0]);
}
/******************************************************************************/
