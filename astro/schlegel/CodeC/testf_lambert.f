      program testf_lambert
      implicit none

      integer*4 qInterp
      integer*4 qNoloop
      integer*4 qVerbose
      integer*4 ngal
      real      tmpl
      real      tmpb
      real      pGall(1)
      real      pGalb(1)
      real      pOutput(1)
      character*80 pFileN
      character*80 pFileS

      print *, 'Galactic longitude (degrees);'
      read *, tmpl
      print *, 'Galactic latitude (degrees);'
      read *, tmpb
      print *, 'Northern FITS file: '
      read *, pFileN
      print *, 'Southern FITS file: '
      read *, pFileS
      print *, 'Interpolate (0=no,1=yes)?'
      read *, qInterp

      ngal = 1
      pGall(1) = tmpl
      pGalb(1) = tmpb
      qVerbose = 0
      qNoloop = 0
      call fort_lambert_getval(pFileN, pFileS, ngal, pGall, pGalb,
     & qInterp, qNoloop, qVerbose, pOutput)
      print *, pOutput(1)

      end
