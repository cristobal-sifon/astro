      program testf_thermal
      implicit none

      integer*4 qInterp
      integer*4 qNoloop
      integer*4 qVerbose
      integer*4 ngal
      integer*4 modelNum
      real      tmpl
      real      tmpb
      real      tmpnu
      real      pGall(1)
      real      pGalb(1)
      real      pNu(1)
      real      pOutput(1)
      character*80 pIPath
      character*10 pResName
      character*10 pUnitsName

      print *, 'Galactic longitude (degrees):'
      read *, tmpl
      print *, 'Galactic latitude (degrees):'
      read *, tmpb
      print *, 'Frequency (GHz):'
      read *, tmpnu
      print *, 'Path name:'
      read (*,'(80a)'), pIPath
      print *, 'Model number?'
      read *, modelNum
      print *, 'Interpolate (0=no,1=yes)?'
      read *, qInterp

c      pIPath = '/u/schlegel/dustpub/maps'
      pResName = 'I4096'
      pUnitsName = 'MJy'

      ngal = 1
      pGall(1) = tmpl
      pGalb(1) = tmpb
      pNu(1) = tmpnu
      qVerbose = 0
      qNoloop = 0
      call fort_predict_thermal(ngal, pGall, pGalb, pNu, pIPath,
     & pResName, pUnitsName, modelNum, qInterp, qNoloop, qVerbose,
     & pOutput)

      print *, pOutput(1)

      end
