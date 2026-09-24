C     ************************ INIT COND *********************
C     Jan 2026 - adapted from trajtou
C
C     ********************************************************
C       generates initial condtions (cartesian coordinates) for the
C       diatom (v,j)
C    
C       Includes all necessarry routines
C       needs intrep.dat and potasym.dat
C       needs inputfile.dat generally prepared in the job script
C       (init_aimd.dat)
C     ********************************************************
C     NSTAR: -if NSTAR.GE.1, we start a new set of calculations:
C             all output files are new and the first trajectory is
C             trajectory NSTAR. 
C            -NSTAR=0: continuation of a previous calculation. Results 
C             are appended to existing output files. Parameters are 
C             those of the previous calculation. 
C            -if NSTAR.LE.-1, start calculation from |NSTAR|. No
C             output except into the file DETAILS.RES. Parameters are 
C             those of a previous calculation. Used to test given 
C             trajectories. 
C     NTIR: number of trajectories in the present calculation.
C     EPERP: initial normal translation energy (eV)
C     NVPARTIR:=1, parallel velocity given in (VX,VY)
C              =2, random distribution of parallel velocity directions
C     THETAV: angle of initial velocity with respect to normal (deg) 
C               0<thetav<90 - (only used when NVPARTIR=2)
C     VX,VY: parallel components of velocity (in au)
C                (only used when NVPARTIR=1)
C     EVIROT: ro-vibrational kinetic energy (in au).
C        If zero, the system is at rest at the bottom of the effective
C        potential well including the rotational energy. This
C        is used for "classical" calculations. 
C     JROT: rotational quantum-number
C     NXYTIR: =1 => impact parameter given in (XZER,YZER).
C             =2 => random distribution of impact parameters.
C     XZER,YZER: center of mass impact parameter (units of disatat) 
C                (only used when NXYTIR=1). Cartesian coordinates.
C     IFIFIX: =1 => fixed phi given in FIFIX 
C             =2 => random distribution of phi.
C     FIFIX: fixed PHI angle (deg).
C                (only used when IFIFIX=1)
C     NDIM: dimension of calculations.
C           = 6 --> 6D
C           = 4 (X,Y) are fixed at their initial value for the 
C               calculation of the potential.
C           = 3 (X,Y,phi) are fixed at their initial value for the
C               calculation of the potential.
C           Note: for reduced dimensions, the value of (X,Y,phi)
C           is not fixed in time but p_X, p_Y, p_phi are constant.
C     MA: mass of atom A (unit proton mass)
C     MB: mass of atom B (unit proton mass)
C     ZIN: initial distance of center of mass to surface (Angstroms)
C     EPST: precision for time integration of internal motion
C     HIN: initial time increment (au) for internal motion
C     PRECIT: precision on determination of turning points.
C     ----------------------------------------------------------------
C     Other data required:
C     ====================
C     The asymptotic potential for the isolated molecule must be given
C     in POTASYM.DAT
C     ----------------------------------------------------------------
C     Potential energy surface:
C     ----------------------------------------------------------------
C     Reduced dimensions:
C     ==================
C     Case I: NDIM=4
C     ------- ------
C     The coordinates (X,Y) of the center of mass are constant for the
C     evaluation of the potential and forces.
C     Their value depends on NXYTIR.
C        NXYTIR=1 => (XZER,YZER) used for all trajectories. 
C        NXYTIR=2 => random value of (X,Y) for each trajectory.
C
C     Case II: NDIM=3
C     -------  ------
C     The coordinates (X,Y) of the center of mass and the azimuthal
C     angle PHI are constant for the evaluation of the potential and
C     forces.
C     The value of (X,Y) depends on NXYTIR.
C        NXYTIR=1 => (XZER,YZER) used for all trajectories. 
C        NXYTIR=2 => random value of (X,Y) for each trajectory.
C     The value of PHI depends on IFIFIX:
C        IFIFIX=1 => FIFIX used for all trajectories.
C        IFIFIX=2 => random value of PHI for each trajectory.
C        XCMFIX,YCMFIX,FIFIX unused when NDIM=6
C
C     ------------------------------------------------------------
C     Definition of some variables:
C     ============================
C     EINTIN : Internal energy
C     VEREPROJ: Minimum energy of asymptotic potential.
C     ETRANS : total translational energy of center of mass
C
C     Signification of the 24 components of z:
C     z(1)=Z_A; z(2)=X_A; z(3)=Y_A; z(4)=Z_B; z(5)=X_B; z(6)=Y_B
C     z(7)=P_(Z_A);  z(8)=P_(X_A); z(9)=P_(Y_A);
C     z(10)=P_(Z_B); z(11)=P_(X_B);  z(12)=P_(Y_B)
c===========
C     COMMONs bringing information into TRAJTOU:
C     -----------------------------------------
C     DATLIM: information on boundaries of PES definition.
C     INVIB: contains the results of INMOL3 used for the MC sampling
C            on the initial vibrational motion. 
C     COMMONs transfering information from TRAJTOU to PES routines
C     --------------------------------------------------------------
C     MASS: transfers masses used in the determination of forces (they
C           depend on the center of mass position).
C     Other COMMONs
C     -------------
C     PESDEF: transfers the PES name from POT6D to PESCHECK.

C=====================================================================
      PROGRAM INITAIMD
      IMPLICIT DOUBLE PRECISION (a-h,o-z)
C
C     IDP: maximum number of points in asymptotic potential
C     IDT: maximum number of times for molecule vibration.
      PARAMETER (idp=999,idt=900)

C     Constants for transformation into atomic units
      PARAMETER(evua=27.2,psua=4.14e4,ptmass=1836.,angua=0.529177249D0)
      PARAMETER (boltzman=1.381D-23*6.242D+18/evua)
      PARAMETER (pi=3.141592653589793D0)
      PARAMETER (iuni=2)
C     idum sets the sampling in RAN0 

      CHARACTER*1 tit(72)
      LOGICAL fin,classic,lstick
      DIMENSION z(12),zinit(12)

      DIMENSION a(idp),aa(idp),aaa(idp)
      DOUBLE PRECISION mt,mu,ma,mb,mamt,mbmt,parinput,einput
      COMMON/INVIB/at(idt),rt(idt),pt(idt),crt(idt),cpt(idt),nt
      COMMON/POTSPL/ras(idp),vas(idp),cvas(idp),nas
      picons=pi
      iunit=iuni
      OPEN(1,file='init_aimd.dat',status='old')
      OPEN(11,FILE='details.res',STATUS='unknown')
        READ(1,*) nstar
        WRITE(11,*)'nstar=',nstar
        READ(1,*) ntir
        WRITE(11,*)'ntir=',ntir 
        IF(nstar.GE.1) THEN
          nstart=nstar
          ninit=nstar
          nstold=nstart-1
          lstick=.true.
          READ(1,*) einput
          WRITE(11,*)'Initial energy=',einput,' eV'
          READ(1,*) nvpartir
          WRITE(11,*)'nvpartir=',nvpartir 
          READ(1,*) thetav
          WRITE(11,*)'thetav=',thetav,' deg' 
          IF(thetav.LT.0.D0.OR.thetav.GT.90.D0) THEN
	     WRITE(*,*) 'THETAV outside interval [0,90]'
	     STOP
	  END IF   
          READ(1,*) parinput
          WRITE(11,*)'parinput=',parinput,' au or deg'
          READ(1,*) vy
          WRITE(11,*)'vy=',vy,' au'
          READ(1,*) evirot
          WRITE(11,*)'evirot=',evirot,' au' 
          READ(1,*) jrot
          WRITE(11,*)'jrot=',jrot 
          READ(1,*) nxytir
          WRITE(11,*)'nxytir=',nxytir 
          READ(1,*) xzer
          WRITE(11,*)'xzer=',xzer,' disatat'
          READ(1,*) yzer
          WRITE(11,*)'yzer=',yzer,' disatat' 
          READ(1,*) ififix
          WRITE(11,*)'ififix=',ififix
          READ(1,*) fifix
          WRITE(11,*)'fifix=',fifix,' deg'
          READ(1,*) ndim
          WRITE(11,*)'ndim=',ndim
          READ(1,*) ma
          WRITE(11,*)'ma=',ma 
          READ(1,*) mb
          WRITE(11,*)'mb=',mb 
          READ(1,*) zin
          WRITE(11,*)'zin=',zin,' Ang.' 
          READ(1,*) epst
          WRITE(11,*)'epst=',epst 
          READ(1,*) hin
          WRITE(11,*)'hin=',hin,' au'
          READ(1,*) precit
          WRITE(11,*)'precit=',precit 
          READ(1,*) xelem
          WRITE(11,*)'xelem=', xelem,' disatat'
          READ(1,*) yelem
          WRITE(11,*)'yelem=', yelem,' disatat'
	ENDIF 

	  OPEN(10,FILE='intrep.dat',STATUS='OLD') 
          READ(10,*)disatat,isatat
          CLOSE(UNIT=10) 
         read(1,*)idum
         write(11,*)'idum',idum
      CLOSE(1)
      CLOSE(11)

      IF(evirot.EQ.0.D0) THEN
        classic=.true.
        WRITE(*,*) 'Classical calculation'
      ELSE
        classic=.false.
        WRITE(*,*) 'Quasi-classical calculation'
      END IF   

      IF(.NOT.((ndim.EQ.3.OR.ndim.EQ.4).OR.ndim.EQ.6) ) THEN
	WRITE(*,*)'ERROR: incorrect value of NDIM'
	STOP
      END IF

      IF(isatat.EQ.233) THEN
        WRITE(*,*) 'Interatomic distance:',disatat,' ANGSTROEMS'
        disunit=angua
      ELSE IF(isatat.EQ.437) THEN
        WRITE(*,*) 'Interatomic distance:',disatat,' ATOMIC UNITS'
        disunit=1.D0
      ELSE
        WRITE(*,*) 'Improper units for DISATAT'
      STOP
        END IF
C                                     Asymptotic (molecular) potential
      OPEN(2,file='potasym.dat',status='old')
        READ(2,fmt='(72a1)') tit
	READ(2,*) nas
        IF(nas.GE.idp) THEN
	  WRITE(*,*) 'IDP too small'
	  STOP
        END IF
	READ(2,*) (ras(n),vas(n),n=1,nas)
      CLOSE(UNIT=2)    
      WRITE(*,*) 'Asymptotic potential:'
      WRITE(*,fmt='(72a1)') tit
      CALL DSPLIN(nas,ras,vas,cvas,0.D0,0,0.D0,0,a,aa,aaa)
      reproj=(ras(1)+ras(nas))/2.D0

C     ----------------------------------------------------------------
C                                                Switch to local units
      fifix=fifix*pi/180.D0

      IF(nxytir.EQ.1) THEN
        xzer=xzer*disatat/disunit
        yzer=yzer*disatat/disunit
      END IF
      zccmin=zccmin/angua
      zin=zin/angua
      eperp=eperp/evua
      tsup=tsup*psua
      h0=h0*psua
      ma=ma*ptmass
      mb=mb*ptmass
      mt=ma+mb
      mu=ma*mb/mt
      mamt=ma/mt
      mbmt=mb/mt
      ms=ms*ptmass
C                                                Initial radial motion
      CALL INMOL3(0,mu,evirot,jrot,epst,hin,precit,reproj,vreproj,
     1            rsm,rout,tnu,idt,idp)

      vnew=POTASYM(reproj)
      IF(classic) THEN
        eintin=vnew+jrot*(jrot+1)/(2.D0*mu*reproj**2)
      ELSE
        eintin=evirot+vnew
      END IF
      vin=DSQRT(2.D0*eperp/mt)
      totmom=DSQRT(DFLOAT(jrot*(jrot+1)))

      IF (nvpartir.EQ.2) THEN
        eperp=einput
        vpar=vin*TAN(pi*thetav/180.D0)
        etrans=eperp+mt*vpar*vpar/2.D0
      ELSE IF (nvpartir.EQ.2) THEN
        eperp=einput
        etrans=eperp+mt*(vx*vx+vy*vy)/2.D0
      ELSE IF (nvpartir.EQ.3) THEN
        etrans=einput
        eperp=einput*COS(thetav*pi/180.D0)**2/evua   ! ua
        vin=DSQRT(2.D0*eperp/mt)                     ! perp velocity
        epar=einput*SIN(thetav*pi/180.D0)**2/evua    ! ua
        vpar=SQRT(2.D0*epar/mt)                      
        vphi=parinput
        vx=vpar*COS(vphi*pi/180.D0)
        vy=vpar*SIN(vphi*pi/180.D0)
      END IF

	et=etrans+eintin
      IF(nstar.LT.0) THEN	
	OPEN(11,FILE='details.res',STATUS='OLD',ACCESS='APPEND')
	WRITE(11,*)'Initial translational energy:',etrans,' ua'
	WRITE(11,*)'Initial internal energy:',eintin,' ua'
	WRITE(11,*)'Initial total energy (with as. pot.):    ',et,
     1             ' ua'
	CLOSE(UNIT=11)
      END IF

C         
C                       END PROLOGUE
C=====================================================================
C                       START TRAJECTORIES
C     ----------------------------------------------------------------
C                                                Loop on trajectories

cpl---< added to parallelize APT code
      ntidinit=(idum-1)*ntir+1
      ntidmax=ntidinit+ntir-1
cpl      DO 157 ncoup=1,ntir+nstart-1
cpl--->
      OPEN(55,FILE='INIT-AIMD.res',STATUS='unknown')
      DO 157 ncoup=ntidinit,ntidmax
      OPEN(11,FILE='trajnbr',STATUS='unknown')
        WRITE(11,*) ncoup
      CLOSE(11)
      zcmmin=zin
C                     Toss over impact parameter and parallel velocity

C                             If required, determine impact-parameter
C         Always toss to ensure that random values are the same in
C         successive calculations (for comparison purposes) 
      ai=RAN0(idum)  ! change by pl for parallel code
      bi=RAN0(idum)  ! change by pl for parallel code 
      IF(nxytir.EQ.2) THEN
        xzer=ai*(disatat/disunit)*xelem
        yzer=bi*(disatat/disunit)*yelem
      END IF       

      IF(ndim.EQ.3.OR.ndim.EQ.4) THEN
        xcmfix=xzer
        ycmfix=yzer
      END IF
C         If required, determine parallel velocity
C         Always toss on xhi to ensure that random values are the same
C         in successive calculations (for comparison purposes) 
      xhi=2.D0*pi*RAN0(idum)
      IF (nvpartir.EQ. 2) THEN
        vpar=vin*TAN(pi*thetav/180.D0)
        vx=vpar*DCOS(xhi)
        vy=vpar*DSIN(xhi)
      END IF
        etrans=eperp+mt*(vx*vx+vy*vy)/2.D0
C     -----------------------------------------------------------------
C                                      Toss over molecular coordinates

      delta=RAN0(idum)
      thetin=DACOS(1.D0-2.D0*RAN0(idum))
      phiin=RAN0(idum)*2.D0*pi
      eta=2.D0*pi*RAN0(idum)

      IF(ififix.EQ.2) THEN
       fifix=phiin
      ELSE IF(ififix.EQ.1) THEN
       phiin=fifix
      END IF
C     -------------------------------------------
      IF(ndim.EQ.3) THEN
        IF(phiin.GE.0.D0.AND.phiin.LE.pi) THEN
         fiinflag=1.D0
        ELSE
         fiinflag=-1.D0
        END IF
      END IF

C        If ncoup<nstart, the above part is used to ensure that the
C        random set is consistent with the nstart first values.
      IF (ncoup.le.nstart-1) GO TO 157

       IF(nstar.LT.0) THEN	
        OPEN(11,FILE='details.res',STATUS='OLD',ACCESS='APPEND')
        WRITE(11,*) 'TRAJECTORY: ',ncoup
        CLOSE(UNIT=11)
      END IF
        pth=-totmom*DSIN(eta)
        pphi=totmom*DCOS(eta)*DSIN(thetin)
C     See comment on COMMON CASO3D
	pfiin=pphi
        IF(classic) THEN
          rin=reproj
          pr=0.D0
        ELSE
	  rin=DPL(nt,at,rt,crt,delta*tnu)
	  pr=DPL(nt,at,pt,cpt,delta*tnu)
        END IF
	CALL SPHCAR(pr,pth,pphi,rin,thetin,phiin,px,py,pz)  
C     -------------------------------------------
C                Set initial conditions

	    z(1)=zin-mbmt*rin*DCOS(thetin)
	    z(4)=zin+mamt*rin*DCOS(thetin)
	    z(2)=xzer-mbmt*rin*DSIN(thetin)*DCOS(phiin)
	    z(5)=xzer+mamt*rin*DSIN(thetin)*DCOS(phiin)
	    z(3)=yzer-mbmt*rin*DSIN(thetin)*DSIN(phiin)
	    z(6)=yzer+mamt*rin*DSIN(thetin)*DSIN(phiin)
	    z(7)=-ma*vin-pz
	    z(10)=-mb*vin+pz
	    z(8)=ma*vx-px
	    z(11)=mb*vx+px
	    z(9)=ma*vy-py
	    z(12)=mb*vy+py
        DO i=1,12
          zinit(i)=z(i)
        END DO

        write(99,*) DSQRT((z(4)-z(1))**2+(z(5)-z(2))**2+(z(6)-z(3))**2)


C --- Converting au to A ---
        DO i=1,6
          zinit(i)=zinit(i)*angua
        END DO

C --- Converting P in au to v in A/ps
        DO i=7,12
          zinit(i)=zinit(i)/ma                ! Converting P to v
          zinit(i)=zinit(i)*2.187691263d4   ! Convert v au -> A/ps
        END DO
C     ----------------------------------------------------------------
       
       WRITE(55,*) ncoup,zinit

C     
 157  CONTINUE
      CLOSE(55)
      stop
      END

**********************************************************************
C *** SUBROUTINES ***
C************************ C A R S P H ********************************
      SUBROUTINE CARSPH(px,py,pz,r,theta,phi,pr,pth,pphi)              
      IMPLICIT DOUBLE PRECISION(a-h,o-z)
C                            Calculate spherical component of momentum
C                                            from cartesian components
      stheta=SIN(theta)
      sphi=SIN(phi)
      ctheta=COS(theta)
      cphi=COS(phi)
C     
      pr=px*stheta*cphi+stheta*sphi*py+ctheta*pz
      pth=r*(ctheta*cphi*px+ctheta*sphi*py-stheta*pz)
      pphi=r*stheta*(cphi*py-sphi*px)
C     
      END
C************************ S P H C A R ********************************
      SUBROUTINE SPHCAR(pr,pth,pphi,r,theta,phi,px,py,pz)              
      IMPLICIT DOUBLE PRECISION(a-h,o-z)
C                           Calculate cartesian components of momentum
C                                            from spherical components
      stheta=SIN(theta)
      sphi=SIN(phi)
      ctheta=COS(theta)
      cphi=COS(phi)
C     
      px=cphi*(stheta*pr+ctheta*pth/r)-sphi*pphi/(r*stheta)
      py=sphi*(stheta*pr+ctheta*pth/r)+cphi*pphi/(r*stheta)
      pz=ctheta*pr-stheta*pth/r
C     
      END
C************************ P O T A S Y M ******************************
      DOUBLE PRECISION FUNCTION POTASYM(r) 
      IMPLICIT DOUBLE PRECISION(a-h,o-z)
      PARAMETER (idp=999)

      COMMON/POTSPL/ras(idp),vas(idp),cvas(idp),nas 
C                                       Asymptotic numerical potential
      IF(r.GE.ras(1).AND.r.LE.ras(nas)) THEN
        CALL SPLINT(ras,vas,cvas,nas,r,pot,basu)
        potasym=pot
      ELSE
        WRITE(*,*)'ERROR in POTASYM'
        STOP
      END IF

      END 

C**************************** D I F S Y 2 ***************************
      SUBROUTINE DIFSY2(n,x,y,h0,eps,s,f,fin)
C 
C       Version 0.1     29/1/97
C
C       Traduction du programme ALGOL de Bulirsch and Stoer, 
C     Numerische Mathematik 8(1966)1. Double precision.
C       DIFSY2 calcule un pas d'integration H inferieur ou egal a H0 
C     pour un systeme de N equations differentielles ordinaires du 
C     premier ordre de forme:
C     dY/dX=F(X,Y)
C     dont le membre de droite doit etre donne par un sous-programme
C     commencant par:
C     SUBROUTINE F(x,y,dy)
C     REAL*8 y(n),dy(n)
C     et dont le nom est passe comme argument (f) dans la sequence
C     d'appel de DIFSY2.
C       Le programme prend pour pas H le premier des nombres H0, H0/2,
C     H0/4 pour lequel 9 etapes d'extrapolation suffisent a obtenir le
C     resultat avec la precision requise. Si H est different de H0, le
C     parametre FIN prend la valeur .TRUE.
C       X et Y sont les valeurs initiales. Apres execution de DIFSY2, 
C     les valeurs initiales de X et Y sont remplacees par X+H et 
C     Y(X+H). H0 peut egalement etre modifie (changement de pas 
C     automatique). La nouvelle valeur de H0 est la meilleure valeur 
C     estimee pour le pas suivant.
C       Le tableau S et la constante EPS commandent la precision des 
C     valeurs calculees. Le calcul est arrete lorsque deux valeurs 
C     successives de Y(i) different au plus de EPS*S(i). EPS ne doit 
C     pas etre inferieur a 10**(-D+3) ou D est le nombre de chiffres 
C     decimaux dans la machine utilisee.
C       Il est conseille de faire S(i)=0. pour le premier pas
C     d'integration. A la sortie de DIFSY2, S(i) est devenu 
C     S(i)=MAX[S(i),Y(i,xsi)] 
C     pour xsi compris dans l'intervalle [X,X+H].
C     *
      PARAMETER (id=2)
      REAL*8 d2,d4,d6,dp2
      PARAMETER(d2=16.D0/9.D0,d4=64.D0/9.D0,d6=256.D0/9.D0,
     1          dp2=9.D0/4.D0)
C
      INTEGER r,sr,ncoup
      LOGICAL bo,bh,fin,konv
      REAL*8 d(7),dt(id,7),dy(id),dz(id),s(id),y(id),ya(id),yg(8,id),
     1       yh(8,id),yl(id),ym(id)
      REAL*8 a,b1,b,c,eps,fc,g,h0,ta,u,v,x
C
      IF(n.GT.id) THEN
        WRITE(*,*) 'ID too small in DIFSY2'
	STOP
      END IF

      CALL F(x,y,dz)
      bh=.FALSE.
      fin=.FALSE.
      DO i=1,n
        ya(i)=y(i)
      END DO

C                                                          Iterations
   20 CONTINUE
      a=h0+x
      fc=1.5D0
      bo=.FALSE.
      m=1
      r=2
      sr=3
      jjp1=0
C                                          Extrapolations successives
      DO 170 jp1=1,10
        IF(bo) THEN
          d(2)=d2
          d(4)=d4
          d(6)=d6
        ELSE
          d(2)=dp2
          d(4)=9.D0
          d(6)=36.D0
        END IF
        konv=.FALSE.
        IF(jp1.GT.3) konv=.TRUE.
        IF(jp1.GT.7) THEN
          lp1=7
          fc=0.6D0*fc
          d(7)=64.D0
        ELSE
          lp1=jp1
          d(lp1)=DFLOAT(m**2)
        END IF
        m=2*m
        g=h0/DFLOAT(m)
        b=2.D0*g
        IF(bh.AND.jp1.LT.9) THEN
          DO i=1,n
            ym(i)=yh(jp1,i)
            yl(i)=yg(jp1,i)
          END DO
        ELSE
C                                                      Calcul de T(H)
          kk=(m-2)/2
          m=m-1
          DO  i=1,n
            yl(i)=ya(i)
            ym(i)=ya(i)+g*dz(i)
          END DO
          DO k=1,m
            CALL F(x+DFLOAT(k)*g,ym,dy)
            DO i=1,N
              u=yl(i)+b*dy(i)
              yl(i)=ym(i)
              ym(i)=u
              u=DABS(u)
              IF(u.GT.s(i))  s(i)=u
            END DO  
            IF(k.NE.kk.OR.k.EQ.2) GO TO 100
            jjp1=1+jjp1
            DO i=1,n
              yh(jjp1,i)=ym(i)
              yg(jjp1,i)=yl(i)
            END DO  
 100        CONTINUE
          END DO  
        END IF   
        CALL F(a,ym,dy)
        DO 160 i=1,n
          v=dt(i,1)
          ta=(ym(i)+yl(i)+g*dy(i))/2.D0
C                                 Calcul de la valeur extrapolee T(0)
          c=ta
          dt(i,1)=ta
          IF(lp1.GE.2) THEN
            DO kp1=2,lp1
              b1=d(kp1)*v
              b=b1-c
              u=v
              IF(b.NE.0.D0) THEN
                b=(c-v)/b
                u=c*b
                c=b1*b
              END IF
              v=dt(i,kp1)
              dt(i,kp1)=u
              ta=u+ta
            END DO  
          END IF
          IF(DABS(y(i)-ta).GT.eps*s(i)) konv=.FALSE.
          y(i)=ta
 160    CONTINUE
        IF(konv) GO TO 180
        d(3)=4.D0
        d(5)=16.D0
        bo=.NOT.bo
        m=r
        r=sr
        sr=m*2
 170  CONTINUE
C
C     Plus de 9 iterations etaient necessaires. On recommence en
C     divisant le pas par 2.
C
      bh=.NOT.bh
      fin=.TRUE.
      h0=h0/2.D0
      GO TO 20
C 
 180  h0=fc*h0
      x=a

      RETURN
      END


C**************************** D P L *********************************
      FUNCTION DPL(n,x,y,m,t)
C
C        Version 3.0  - 15/1/97       Author: A. SALIN
C
C        Calculation of a function by interpolation. Double precision.
C        See comments in SUBROUTINE DSPLIN
C
      IMPLICIT REAL*8 (a-h,o-z)
c
      DIMENSION x(n),y(n)
      REAL*8 m(n)
      LOGICAL order
c
      COMMON/KODSPL/kod,k,iw,iligne
c
    1 FORMAT(' DPL: extrapolation - ',1PE15.8,' <',E15.8)
    2 FORMAT(' DPL: extrapolation - ',1PE15.8,' >',E15.8)
c
      order=x(2).GT.x(1)
      IF(k.LE.1.OR.k.GT.n) THEN
	klo=0
	khi=n+1
	GO TO 100
      END IF
C
      inc=1
      klo=k-1
      IF(t.GT.x(klo).EQV.order) THEN
 10     khi=klo+inc
	IF(khi.GT.n) THEN
	  khi=n+1
	ELSE IF(t.GT.x(khi).EQV.order) THEN
	  klo=khi
	  inc=inc+inc
	  GO TO 10
	END IF
      ELSE
	khi=klo
 20     klo=khi-inc
	IF(klo.LT.1) THEN
	  klo=0
	ELSE IF(t.LT.x(klo).EQV.order) THEN
	  khi=klo
	  inc=inc+inc
	  GO TO 20
	END IF
      END IF
C
 100  CONTINUE
      IF(khi-klo.GT.1) THEN
	km=(khi+klo)/2
	IF(t.GT.x(km).EQV.order) THEN
	  klo=km
	ELSE
	  khi=km
	END IF
	GO TO 100
      END IF
	k=klo+1
C
      IF(t.EQ.x(1)) k=2
      IF(k.LE.1) THEN
	e=x(2)-x(1)
	DPL=((y(2)-y(1))/e-m(2)*e/6.D0)*(t-x(1))+y(1)
	kod=1
	IF(iw.GT.0.AND.iligne.GT.0) THEN
	  IF(order) THEN
	    WRITE(iw,1) t,x(1)
	  ELSE
	    WRITE(iw,2) t,x(1)
	  END IF
	  iligne=iligne-1
	END IF
      ELSE IF(k.GT.n) THEN
	e=x(n)-x(n-1)
	DPL=((y(n)-y(n-1))/e+m(n-1)*e/6.D0)*(t-x(n))+y(n)
	kod=2
	IF(iw.GT.0.AND.iligne.GT.0) THEN
	  IF(order) THEN
	    WRITE(iw,2) t,x(n)
	  ELSE
	    WRITE(iw,1) t,x(n)
	  END IF
	  iligne=iligne-1
	END IF
      ELSE
	f=x(k)-t
	g=t-x(k-1)
	e=f+g
	DPL=(-g*f*(m(k-1)*(f+e)+m(k)*(g+e))+6.D0*(g*y(k)+f*y(k-1)))/
     1      (6.D0*e)
	kod=0
      END IF
      END

C**************************** D P L P *******************************
      FUNCTION DPLP(n,x,y,m,t)
C
C        Version 3.0  - 17/1/97       Author: A. SALIN
C
C        Calculation of the derivative of a function by interpolation.
C        Double precision. See comments in SUBROUTINE DSPLIN.
C        If t is outside the interval over which the function is
C        defined, the derivative is extrapolated linearly from its
C        value for the first two (last two) points of the interval.
C
      IMPLICIT REAL*8 (a-h,o-z)
c
      DIMENSION x(n),y(n)
      REAL*8 m(n)
      LOGICAL order
c
      COMMON/KODSPL/kod,k,iw,iligne
c
    1 FORMAT(' DPLP: extrapolation - ',1PE15.8,' <',E15.8)
    2 FORMAT(' DPLP: extrapolation - ',1PE15.8,' >',E15.8)
c
      order=x(2).GT.x(1)
      IF(k.LE.1.OR.k.GT.n) THEN
        klo=0
        khi=n+1
        GO TO 100
      END IF
C
      inc=1
      klo=k-1
      IF(t.GT.x(klo).EQV.order) THEN
 10     khi=klo+inc
        IF(khi.GT.n) THEN
          khi=n+1
        ELSE IF(t.GT.x(khi).EQV.order) THEN
          klo=khi
          inc=inc+inc
          GO TO 10
        END IF
      ELSE
        khi=klo
 20     klo=khi-inc
        IF(klo.LT.1) THEN
          klo=0
        ELSE IF(t.LT.x(klo).EQV.order) THEN
          khi=klo
          inc=inc+inc
          GO TO 20
        END IF
      END IF
C
 100  CONTINUE
      IF(khi-klo.GT.1) THEN
        km=(khi+klo)/2
        IF(t.GT.x(km).EQV.order) THEN
          klo=km
        ELSE
          khi=km
        END IF
        GO TO 100
      END IF
        k=klo+1
C
      IF(t.EQ.x(1)) k=2
      IF(k.LE.1) THEN
        e=x(2)-x(1)
        g=t-x(1)
        f=x(2)-t
        DPLP=(y(2)-y(1))/e+(m(2)*(2*g-f)+m(1)*(g-2*f))/6.D0
        kod=1
        IF(iw.GT.0.AND.iligne.GT.0) THEN
          IF(order) THEN
            WRITE(iw,1) t,x(1)
          ELSE
            WRITE(iw,2) t,x(1)
          END IF
          iligne=iligne-1
        END IF
      ELSE IF(k.GT.n) THEN
        e=x(n)-x(n-1)
        g=t-x(n-1)
        f=x(n)-t
        DPLP=(y(n)-y(n-1))/e+(m(n)*(2*g-f)+m(n-1)*(g-2*f))/6.D0
        kod=2
        IF(iw.GT.0.AND.iligne.GT.0) THEN
          IF(order) THEN
            WRITE(iw,2) t,x(n)
          ELSE
            WRITE(iw,1) t,x(n)
          END IF
          iligne=iligne-1
        END IF
      ELSE
        f=x(k)-t
        g=t-x(k-1)
        e=f+g
        e2=e*e
        DPLP=(-m(k-1)*(3.D0*f*f-e2)+m(k)*(3.D0*g*g-e2)
     1       +6.D0*(y(k)-y(k-1)))/(6.D0*e)
        kod=0
      END IF
      END

C**************************** D S P L I N *************************** 
C        Interpolation package. Given a set of n couples x(i),y(i), 
C     DSPLIN defines a cubic spline function F(x) such that: 
C        1)- F(x(i)) = y(i) 
C        2)- F(x), F'(x), F''(x) are continuous in the interval 
C     [x(1),x(n)]. 
C        Required subroutine: TRIDIA. 
C 
      BLOCK DATA BLKSPL 
      COMMON/KODSPL/kod,k,iw,iligne 
      DATA iw,iligne,k/6,100,2/ 
      END 
C******************************************************************** 
C 
      SUBROUTINE DSPLIN(n,x,y,cm,cm1,ic1,cmn,icn,alpha,beta,b) 
C 
C        Author: A. Salin   Version 3.2  20/1/86 - 10/6/97      
C 
C     ****SUBROUTINE DSPLIN: calculates the vector F''(x(i)) which 
C     defines the cubic spline function F(x). 
C          N= number of pivots x(i). 
C          X= array of abcissae. Should be stored by increasing or     
C     decreasing order. 
C          Y= array of values of Y(i). 
C          CM= array of F''(x(i)) with dimension equal to that of X 
C     and Y.       
C          IC1,CM1,ICN,CMN: define the condition at x(1) and x(n) 
C     If IC1=0, function is the same in first two intervals. 
C     If IC1=1, second derivative at x(1) given in CM1 
C     If IC1=2, first derivative at x(1) given in CM1 
C     IF IC1>2, zero second derivative at x(1) 
C          Similar definitions for ICN and CMN around x(n) 
C          ALPHA, BETA, B are working arrays of dimension N at least. 
C     For cyclic spline use the program TRISPL. 
C 
C     ****DPLCOF: the interpolated function at R is: 
C       F=C(1)*CM(K-1)+C(2)*CM(K)+C(3)*Y(K-1)+C(4)*Y(K) 
C       where C is obtained from: 
C       CALL DPLCOF(n,x,r,c,k) 
C       The vector C should be of dimension 4. 
C 
C     ****Functions DPL,DPLP,DPLP2: calculate respectively the function 
C     F(x), its first or second derivative:  
C             N= number of pivots 
C             X,Y,M= same as X,Y,CM in DSPLIN. 
C             T= value for which the function (or its derivative) 
C     must be determined. 
C     WARNING: when T is outside the interval [X(1),X(N)], F(X)  
C     *******  (resp. F', F") is determined by linear extrapolation 
C              using the value of F (resp. F', F") for the first two   
C              (last two) points in the interval. 
C 
C     ****Subroutine DINITI et DINTSP: calculate the integral of the 
C     function F(x) from X(1) to T. The subroutine DINITI should first 
C     be called before the first CALL DINTSP concerning a given 
C     function F(x). DINITI calculates the array (CI) of values of the 
C     integral of F(x) from X(1) to all X(i). 
C 
C        Arguments of DINITI: 
C             N,X,Y,CM: as in subroutine DSPLIN. 
C             CI: array of dimension at least N.   
C        Arguments of DINTSP: 
C             N,X,Y,CM,CI: as in DINITI 
C             T: value of the upper limit of the integration (X(1).LT.T. 
C     LE.X(N)). 
C             B: value of the integral. 
C       The calculations in DINTSP are nearly as rapid as for the deter 
C     mination of the function F(x) by DPL. 
C 
C     Parameters of COMMON/KODSPL/: 
C     ----------------------------------------------------- 
C       -KOD= after execution of one subroutine of the package, KOD 
C     takes the value: 
C              *0: no error 
C              *-1: values of X(i) are not stored in correct order. 
C              *1: T outside the interval [X(1),X(N)]. Linear extrapo- 
C     lation done using X(1) and X(2). 
C              *2: T outside the interval [X(1),X(N)]. Linear extrapo- 
C     lation done using X(n-1) and X(n). 
C       -K= after the execution of DPL, DPLP or DPLP2, K is such that 
C     T lies in the interval [X(k-1),X(k)]. 
C       -IW= error messages are printed on unit IW unless IW.LE.0 
C     Default: 6. 
C       -ILIGNE: when T is outside the interval [X(1),X(n)], DPL, DPLP 
C     and DPLP2 print a message if ILIGNE.GT.0. The initial value of 
C     ILIGNE (defined by DATA) is 100. For every extrapolation, the 
C     value is decreased by 1. 
C 
C     *************************************************************** 
C 
      IMPLICIT REAL*8 (a-h,o-z) 
      PARAMETER (six=6.D0,douze=12.D0,zero=0.D0) 
C 
      DIMENSION x(n),y(n),cm(n) 
      DIMENSION alpha(n),beta(n),b(n) 
C      
      COMMON/KODSPL/kod,k,iw,iligne 
C 
   91 FORMAT('Error in the data for DSPLIN',/,'Check abcissae:', 
     1       1PD15.8,1X,' and ',D15.8,/,'Program stopped') 
C 
C                                           Contour condition at X(1) 
      IF(ic1.EQ.0) THEN 
        s1=zero 
        fac1=(x(2)-x(1))*(1.D0+(x(2)-x(1))/(x(3)-x(2)))/six 
        fab1=-(x(2)-x(1))**2/(six*(x(3)-x(2))) 
      ELSE IF(ic1.EQ.1) THEN 
        s1=-cm1*(x(2)-x(1))/six 
        fac1=zero 
        fab1=zero 
      ELSE IF(ic1.EQ.2) THEN 
        s1=0.5D0*(cm1-(y(2)-y(1))/(x(2)-x(1))) 
        fac1=-(x(2)-x(1))/douze 
        fab1=zero 
      ELSE 
C        WRITE(iw,*) 'DSPLIN: unknown condition - use default' 
        s1=zero 
        fac1=zero 
        fab1=zero 
      END IF 
C                                           Contour condition at X(n) 
      IF(icn.EQ.0) THEN 
        sn=zero 
        facn=(x(n)-x(n-1))*(1.D0+(x(n)-x(n-1))/(x(n-1)-x(n-2)))/six 
        fabn=-(x(n)-x(n-1))**2/(six*(x(n-1)-x(n-2))) 
      ELSE IF(icn.EQ.1) THEN 
        sn=-cmn*(x(n)-x(n-1))/six 
        facn=zero 
        fabn=zero 
      ELSE IF(icn.EQ.2) THEN 
        sn=-0.5D0*(cmn-(y(n)-y(n-1))/(x(n)-x(n-1))) 
        facn=-(x(n)-x(n-1))/douze 
        fabn=zero 
      ELSE 
C        WRITE(iw,*) 'DSPLIN: unknown condition - use default' 
        sn=zero 
        facn=zero 
        fabn=zero 
      END IF 
C 
      kod=0 
      c=x(2)-x(1) 
      e=(y(2)-y(1))/c 
      DO i=3,n 
        i2=i-2 
        a=c 
        c=x(i)-x(i-1) 
        IF(a*c.LE.zero) THEN    
          kod=-1 
          IF(iw.GT.0) WRITE(iw,91) x(i-1),x(i) 
          STOP 
          END IF 
        alpha(i2)=(a+c)/3.D0 
        beta(i2)=c/six 
        cm(i2)=beta(i2) 
        d=e 
        e=(y(i)-y(i-1))/c 
        b(i2)=e-d 
      END DO 
C 
      b(1)=b(1)+s1 
      b(n-2)=b(n-2)+sn 
      alpha(1)=alpha(1)+fac1 
      alpha(n-2)=alpha(n-2)+facn 
      cm(n-3)=cm(n-3)+fabn 
      beta(1)=beta(1)+fab1 
C                                            Solve tridiagonal system 
      CALL TRIDIA(alpha,cm,beta,b,cm(2),n-2) 
C      
      IF(ic1.EQ.0) THEN 
        cm(1)=cm(2)*(1.D0+(x(2)-x(1))/(x(3)-x(2))) 
     1        -cm(3)*(x(2)-x(1))/(x(3)-x(2)) 
      ELSE IF(ic1.EQ.1) THEN 
        cm(1)=cm1 
      ELSE IF(ic1.EQ.2) THEN 
        cm(1)=-six*s1/(x(2)-x(1))-cm(2)/2.D0 
      ELSE 
        cm(1)=zero 
      END IF 
C       
      IF(icn.EQ.0) THEN 
        cm(n)=cm(n-1)*(1.D0+(x(n)-x(n-1))/(x(n-1)-x(n-2))) 
     1        -cm(n-2)*(x(n)-x(n-1))/(x(n-1)-x(n-2)) 
      ELSE IF(icn.EQ.1) THEN 
        cm(n)=cmn 
      ELSE IF(icn.EQ.2) THEN 
        cm(n)=-six*sn/(x(n)-x(n-1))-cm(n-1)/2.D0 
      ELSE 
        cm(n)=zero 
      END IF 
C 
      k=2 
      RETURN 
      END 

C**************************** I N M O L 3 ***************************
      SUBROUTINE INMOL3(numpot,mu,evirot,jrot,eps,hin,precis,re,vre,
     1                  rinner,router,tnu,idtc,idpc)

C       Classical ro-vibrationnal motion. Calculation of turning
C     points and period.
C
C          Author: A. Salin        Version 2.0   19/01/2001
C       *
C      
C     Included subroutines: POTEFF, DPOTEFF, ZERRIN, FVIBIN, PERIOD. 
C     Library routines: SAKEN, DSPLIN, TRISPL, DPL, DPLP, DIFSY2.
C
C       INPUT:
C     NUMPOT: if 0, numerical potential otherwise Morse potential
C     MU: reduced mass of vibrating system
C     EVIROT: ro-vibrational kinetic energy (in au).
C         If zero, the system is at rest at the bottom of the
C         potential well (including the rotational energy). This
C         is used for "classical" calculations. 
C     JROT: rotation quantum number
C     EPS: precision on integration of vibration motion.
C     HIN: time step (in au) (used for integration and interpolation
C          to find tnu). 
C     PRECIS: precision on turning points
C     RE: initial guess of equilibrium distance.
C     IDTC: dimensions in COMMON/INVIB/
C     IDPC: dimensions in COMMON/POTSPL/
C
C       OUTPUT:
C  EVIROT non zero: 
C     RE,VRE: equilibrium distance and corresponding potential 
C             energy (au) in the absence of rotation.
C     RINNER, ROUTER: inner and outer turning points (au)
C     TNU: vibration period (au)
C     Through Common INVIB information is transferred that allows a
C     Monte-Carlo sampling. 
C  EVIROT zero: 
C     RE, VRE: equilibrium distance and corresponding potential 
C              energy (au) in the presence of rotation.
C     RINNER, ROUTER, TNU are undefined. Common INVIB unused.
C
      IMPLICIT REAL*8(a-h,o-z)
      PARAMETER(idt=900,idp=999)
C          
C     IDT: maximum number of time steps in integration.
C     IDP: should be as IDPC
C
      PARAMETER(pi=3.141592653589793D0)
C     *
      CHARACTER*1 tit(72)
      REAL*8 mu,mas
      LOGICAL fin,classic
      DIMENSION z(2),s(2),a(idt),aa(idt),aaa(idt),a4(idt),a5(idt)
C     *
      EXTERNAL FVIBIN,ZERRIN,PERIOD,DPOTEFF
C
      COMMON/INICOM/eint,ej2m,mas,npot
      COMMON/INVIB/at(idt),r(idt),p(idt),cr(idt),cp(idt),nt
      COMMON/POTSPL/ras(idp),vas(idp),cvas(idp),nas
C     *
      IF(evirot.EQ.0.D0) THEN
        classic=.true.
      ELSE
        classic=.false.
        IF(idt.NE.idtc) THEN
	  WRITE(*,*) 'IDT incorrect in INMOL3'
	  STOP
        END IF
      END IF

      npot=numpot
      mas=mu
      IF(numpot.EQ.0) THEN
        IF(idp.NE.idpc) THEN
          WRITE(*,*) 'IDP incorrect in INMOL3'
	  STOP
        END IF
        rtop=ras(nas)
        rbot=ras(1)
      ELSE
        rbot=0.D0
        rtop=1.d20
      END IF

      OPEN(1,file='inmol3.res',status='unknown')

C--------------------------------------------------------------------
C                      Look for equilibrium distance without rotation
      IF(.NOT.classic) THEN
        ej2m=0.D0
        remin=MAX(re,rbot)
 121    CONTINUE
	  IF(DPOTEFF(remin).GE.0.D0) THEN
            IF(remin.GT.rbot) THEN
   	      remin=MAX(remin*0.9D0,rbot)
	      GO TO 121
	    ELSE 
	      WRITE(*,*) 'No minimum in initial potential'
	      STOP
	    END IF  
	  END IF
        rsup=MIN(1.1D0 * remin,rtop)
 122    CONTINUE
	  IF(DPOTEFF(rsup).LE.0.D0) THEN
	    IF(rsup.LT.rtop) THEN
	      rsup=MIN(1.5*(rsup-remin) + remin,rtop)
	      GO TO 122
	    ELSE
	      WRITE(*,*) 'No minimum in initial potential'
	      STOP
	    END IF    
	  END IF
        dr=DABS(remin-rsup)/10.D0
        CALL SAKEN(remin,rsup,dr,precis,DPOTEFF,re,kod,6)
        vre=POTEFF(re)
        WRITE(1,*) 'Bottom of well:',re,' E:',vre
        eint=evirot+vre
      END IF
C--------------------------------------------------------------------
C             From now on, rotation is included      
      ej2m=jrot*(jrot+1)/(2.D0*mu)
C--------------------------------------------------------------------
C                         Look for equilibrium distance with rotation
      remin=MAX(re,rbot)
 21   CONTINUE
	IF(DPOTEFF(remin).GE.0.D0) THEN
	  IF(remin.GT.rbot) THEN
	    remin=MAX(remin*0.9D0,rbot)
	    GO TO 21
	  ELSE
	    WRITE(*,*) 'No minimum in effective initial potential'
	    STOP
	  END IF    
	END IF
	rsup=MIN(1.1D0 * remin,rtop)
 22   CONTINUE
	IF(DPOTEFF(rsup).LE.0.D0) THEN
	  IF(rsup.LT.rtop) THEN
	    rsup=MIN(1.5*(rsup-remin) + remin,rtop)
	    GO TO 22
	  ELSE
	    WRITE(*,*) 'No minimum in effective initial potential'
	    STOP
	  END IF
	END IF      
	dr=DABS(remin-rsup)/10.D0
      CALL SAKEN(remin,rsup,dr,precis,DPOTEFF,rerot,kod,6)
      WRITE(1,*) 'Equilibrium distance (with rotation):',rerot
C                                                End "classical" case
      IF(classic) THEN
        re=rerot
        vre=POTEFF(re)
        RETURN
      END IF
C                                        Look for outer turning point
      rsup=MIN(1.1D0 * rerot,rtop)
 1    CONTINUE
	IF(ZERRIN(rsup).LE.0.D0) THEN
          IF(rsup.LT.rtop) THEN
	    rsup=MIN(1.5*(rsup-rerot) + rerot,rtop)
  	    GO TO 1
	  ELSE
	    WRITE(*,*) 'No outer turning point for initial energy'
	    STOP
	  END IF    
	END IF
	dr=DABS(rsup-rerot)/10.D0
      CALL SAKEN(rerot,rsup,dr,precis,ZERRIN,router,kod,6)
 2    CONTINUE      
      IF(ZERRIN(router).GT.0.D0) THEN 
	router=router-precis*DABS(router)
	GO TO 2
      END IF
      WRITE(1,fmt='(''Outer turning point:'',1pe12.4,'' au'',/)') 
     1                                                        router

C                                        Look for inner turning point
      rsup=MAX(rerot-0.5D0*(router-rerot),rbot)
 79   CONTINUE
	IF(ZERRIN(rsup).LE.0.D0) THEN
	  IF(rsup.GT.rbot) THEN
	    rsup=MAX(1.5*(rsup-rerot) + rerot,rbot)
	    GO TO 79
	  ELSE         
	    WRITE(*,*) 'No inner turning point for initial energy'
	    STOP
	  END IF    
	END IF
	dr=DABS(rsup-rerot)/10.D0
      CALL SAKEN(rerot,rsup,dr,precis,ZERRIN,rinner,kod,6)
 3    CONTINUE      
      IF(ZERRIN(rinner).GT.0.D0) THEN 
	rinner=rinner+precis*DABS(rinner)
	GO TO 3
      END IF
      WRITE(1,fmt='(''Inner turning point:'',1pe12.4,'' au'',/)') 
     1                                                         rinner
C--------------------------------------------------------------------
C     *                      Start integration at outer turning point
      t=0.d0
      z(1)=router
      z(2)=0.D0
      nt=1
      at(1)=0.D0
      r(1)=router
      p(1)=0.D0
      WRITE(1,fmt='(''Initial total energy:'',1pe12.4,'' au'')') 
     1                                                 POTEFF(router) 
      nc=0
      ht=hin
C       *                             Integrate differential equation
 11   CONTINUE
	s(1)=0.D0
	s(2)=0.D0
	IF(ht.GT.hin) ht=hin
	nt=nt+1
	  IF(nt.GT.idt) THEN
	    write(6,*) 'IDT too small in INMOL3'
	    STOP
	  END IF
	CALL DIFSY2(2,t,z,ht,eps,s,FVIBIN,fin)
	WRITE(1,*) t,z(1),z(2)
	at(nt)=t
	r(nt)=z(1)
	p(nt)=z(2)
	IF(nt.EQ.2) GO TO 11
	IF((r(nt)-rerot)*(r(nt-1)-rerot).LT.0.D0) nc=nc+1
C                    Stop after passing three times potential minimum
	IF(nc.EQ.3) GO TO 12
c                 Note time interval over which the first period ends
	IF((r(nt)-rerot).GT.0.D0.AND.p(nt)*p(nt-1).LT.0.D0)  THEN
	  ntr=nt
	  WRITE(1,*) 'ntr ',ntr
	END IF
	GO TO 11
C
 12   CONTINUE
C                                         Check conservation of energy
      et=z(2)**2/(2.d0*mu)+POTEFF(z(1))
      WRITE(1,fmt ='(''Final total energy:'',1pe12.4,'' au'',/)') et
C                                                     Calculate period
      dp=-DPOTEFF(router)      
      dpnt=-DPOTEFF(r(nt))
      drnt=p(nt)/mas
      CALL DSPLIN(nt,at,r,cr,0.D0,2,drnt,2,a,aa,aaa)
      CALL DSPLIN(nt,at,p,cp,dp,2,dpnt,2,a,aa,aaa)
      CALL SAKEN(at(ntr-1),at(ntr),1.D0,precis,PERIOD,tnu,kod,6)
      WRITE(1,fmt='(''Period:'',1pe12.4,'' au'')') tnu

C                                      Prepare interpolation over time

      nt=ntr
      at(nt)=tnu
      r(nt)=router
      p(nt)=0.D0
      CALL TRISPL(nt,at,r,cr,a,aa,aaa,a4,a5)
      CALL TRISPL(nt,at,p,cp,a,aa,aaa,a4,a5)
C     
      CLOSE(1)
      END
C**************************** Z E R R I N ***************************
      REAL*8 FUNCTION ZERRIN(r)
      IMPLICIT REAL*8 (a-h,o-z)
C                                             Search for turning point
      REAL*8 mu
      COMMON/INICOM/eint,ej2m,mu,npot
C       
      ZERRIN=POTEFF(r)-eint
      RETURN
      END
C**************************** F V I B I N ***************************
      SUBROUTINE FVIBIN(t,z,dz)
      IMPLICIT REAL*8 (a-h,o-z)
C                                                  Hamilton equations
      DIMENSION z(2),dz(2)
C       
      REAL*8 mu
      COMMON/INICOM/eint,ej2m,mu,npot
C       
      dz(1)=z(2)/mu
      dz(2)=-DPOTEFF(z(1))
      RETURN
      END
C**************************** P E R I O D ***************************
      REAL*8 FUNCTION PERIOD(t)
      IMPLICIT REAL*8(a-h,o-z)
C                                                   Search for period
C                   Give the value of p by interpoltation for given t
      PARAMETER(idt=900)
C     *
      COMMON/INVIB/at(idt),r(idt),p(idt),cr(idt),cp(idt),nt
C     *
      period=DPL(nt,at,p,cp,t)
      RETURN
      END
C**************************** P O T E F F ***************************
      REAL*8 FUNCTION POTEFF(r)
      IMPLICIT REAL*8(a-h,o-z)
      PARAMETER(idp=999)
C                                                           Potential      
      REAL*8 mu
      COMMON/INICOM/eint,ej2m,mu,npot
      COMMON/INPOT/d,alp,re
      COMMON/POTSPL/ras(idp),vas(idp),cvas(idp),nas
C                                    
      IF(npot.EQ.0) THEN
        POTEFF=DPL(nas,ras,vas,cvas,r)+ej2m/(r*r)
      ELSE	                 
        arg=alp*(r-re)/2.D0
        POTEFF=d*(DEXP(-arg)*2.d0*DSINH(arg))**2+ej2m/(r*r)
      END IF
      RETURN 
      END
C**************************** D P O T E F F *************************
      REAL*8 FUNCTION DPOTEFF(r)
      IMPLICIT REAL*8(a-h,o-z)
      PARAMETER(idp=999)
C                                             Derivative of potential
      REAL*8 mu
      COMMON/INICOM/eint,ej2m,mu,npot
      COMMON/INPOT/d,alp,re
      COMMON/POTSPL/ras(idp),vas(idp),cvas(idp),nas
C
      IF(npot.EQ.0) THEN
        DPOTEFF=DPLP(nas,ras,vas,cvas,r)-2.D0*ej2m/r**3
      ELSE	
        arg=alp*(r-re)/2.D0
        DPOTEFF=d*alp*4.d0*(DEXP(-3.D0*arg)*DSINH(arg))-2.D0*ej2m/r**3
      END IF	
      RETURN 
      END
cxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
      REAL*8 FUNCTION ran0(idum)
      INTEGER idum,IA,IM,IQ,IR,MASK
      REAL*8 AM
      PARAMETER (IA=16807,IM=2147483647,AM=1.D0/IM,IQ=127773,IR=2836,
     *MASK=123459876)
      INTEGER k
      idum=ieor(idum,MASK)
      k=idum/IQ
      idum=IA*(idum-k*IQ)-IR*k
      if (idum.lt.0) idum=idum+IM
      ran0=AM*idum
      idum=ieor(idum,MASK)
      return
      END

C**************************** S A K E N *****************************
C        SAKEN   VERSION 6.2   31/10/75 - 5/1/00      A. SALIN
C
C        Zero of a function
C
C        Double precision
C
C        Start with an upper (RSUP) and lower (RINF) bound of the
C        root.
C        Firstly, the interval [RINF,RSUP] is decreased such that
C        |RSUP-RINF| < DR. Then the root is found with precision EPS
C        by successive interpolations using Aitken scheme.
C
C        Constraints:
C             The first part assumes that the root is not located at
C             an extremum of the function.
C        The function is given in a FUNCTION.
C
C        RINF: lower bound on R.
C        RSUP: upper bound on R.
C        DR: interval between RINF and RSUP before starting
C        interpolation.
C        EPS: required precision on R.
C        F: name of FUNCTION procedure defining the function.
C        R: set to the root position.
C        KOD:
C                  On input:
C             KOD=0 : do interpolation even if the root is not
C        between RINF and RSUP after decreasing the interval to DR.
C             KOD=-1 : message written if the root is not in the
C        defined interval.
C                  On output:
C             KOD=0 : convergence obtained.
C             KOD=1 : no single root between RINF and RSUP
C             KOD=2 : interpolation has not converged
C        ICONS: unit for error messages
C
      SUBROUTINE SAKEN(rinf,rsup,dr,eps,F,r,kod,icons)
      PARAMETER (nmax=20)
      REAL*8 rsup,rinf,dr,eps,r1,r2,r3,y1,y2,y3,a,ext,x,r,xx,F,prod
      DIMENSION a(nmax),x(nmax)
C
      r1=rinf
      r2=rsup
      y1=F(r1)
      IF(y1.EQ.0.d0) THEN
        r=r1
        kod=0
        RETURN
      END IF
      y2=F(r2)
      IF(y2.EQ.0.d0) THEN
        r=r2
        kod=0
        RETURN
      END IF
C                                                     Reduce interval
   11 CONTINUE
      IF(DABS(r2-r1).GT.DABS(dr))  THEN
        r3=(r1+r2)/2.D0
        y3=F(r3)
        IF(y3.EQ.0.d0) THEN
          r=r3
          kod=0
          RETURN
        END IF
        IF((y3*y1).LE.0.D0) THEN
          y2=y3
          r2=r3
        ELSE
          y1=y3
          r1=r3
        END IF
      GO TO 11
      END IF
C                                              Check location of root
      IF(y1*y2.GT.0.D0) THEN
        IF(kod.EQ.-1) THEN
          WRITE(icons,3) r1,y1,r2,y2
    3 FORMAT('SAKEN: Root not between RINF and RSUP',/,' R1=',
     1       1PD13.6,3X,'Y1=',D13.6,3X,'R2=',D13.6,3X,'Y2=',D13.6)
          kod=1
          RETURN
        ELSE IF(kod.NE.0) THEN
          kod=1
          RETURN
        ELSE
          kod=1
        END IF
      END IF
C      WRITE(*,*)'En SAKEN, comienza la interpolacion'
C                                                       Interpolation
      a(1)=r1
      x(1)=y1
      ext=r1
      x(2)=y2
      r=(r2-r1)/(y2-y1)
      a(2)=r
      prod=-y1
      r=ext+prod*r
      IF(DABS(r-ext).LE.(DABS(r)*eps)) THEN
        IF(kod.NE.1) kod=0
        RETURN
      END IF
      ext=r
      DO n=3,nmax
        nm=n-1
        x(n)=F(r)
        xx=x(n)
        DO i=1,nm
          r=(r-a(i))/(xx-x(i))
        END DO
        a(n)=r
        prod=-prod*x(nm)
        r=ext+prod*r
      IF(DABS(r-ext).LE.(DABS(r)*eps)) THEN
        IF(kod.NE.1) kod=0
        RETURN
      END IF
        ext=r
      END DO
C                                            Convergence not achieved
      WRITE(icons,2) nmax
    2 FORMAT('SAKEN: more than ',i2,' points required')
      kod=2
      RETURN
      END

C************************ S P L I N T ********************************
      SUBROUTINE SPLINT(xa,ya,cya,n,x,y,yp)  
      IMPLICIT REAL*8(a-h,o-z)  

C     Calculates a function Y and its derivative YP at X by spline
C     interpolation. The N data are XA (nodes) and YA (function at
C     nodes). CYA is calculated by DSPLIN

      DIMENSION xa(n),ya(n),cya(n)  

      klo=1  
      khi=n

1     CONTINUE
      IF (khi-klo.GT.1) THEN  
        k=(khi+klo)/2  
        IF(xa(k).GT.x)THEN  
          khi=k  
        ELSE  
          klo=k  
        END IF  
        GO TO 1  
      END IF  

      h=xa(khi)-xa(klo)  
      a=(xa(khi)-x)/h  
      b=(x-xa(klo))/h  
      y=a*ya(klo)+b*ya(khi)+  
     1      ((a**3-a)*cya(klo)+(b**3-b)*cya(khi))*(h*h)/6.D0  
      yp=(ya(khi)-ya(klo))/h+
     1    h*(cya(khi)*(3.D0*b*b-1.D0)-cya(klo)*(3.D0*a*a-1.D0))/6.D0

      END  


C     *********************** T R I C Y C L E ***********************
      SUBROUTINE TRICYCLE(alpha,beta,gamma,cbot,ctop,b,x,n,z)
C
C     Solution of tridiagonal cyclic system
C     Required subroutine: TRIDIA
C
C         A. SALIN    Version 1    9/8/98
C
C     ALPHA: main diagonal (destroyed in TRICYCLE)
C     BETA: subdiagonal
C     GAMMA: superdiagonal
C     CBOT: element of bottom left corner of matrix
C     CTOP: element of top right corner of matrix
C     B: righthand side (destroyed in TRICYCLE)
C     X: solution
C     N: dimension of system
C     Z: scratch vector of dimension N at least

      IMPLICIT REAL*8 (a-h,o-z)
      DIMENSION alpha(n),beta(n),gamma(n),b(n),x(n),z(n)
C
      alp=-alpha(1)
      alpha(1)=alpha(1)-alp
      alpha(n)=alpha(n)-cbot*ctop/alp
      DO i=1,n
        z(i)=alpha(i)
      END DO
      CALL TRIDIA(alpha,beta,gamma,b,x,n)
C
      DO i=1,n
        alpha(i)=z(i)
      END DO
      b(1)=alp
      b(n)=cbot
      DO i=2,n-1
        b(i)=0.D0
      END DO
      CALL TRIDIA(alpha,beta,gamma,b,z,n)
      fact=(x(1)+ctop*x(n)/alp)/(1.D0+z(1)+ctop*z(n)/alp)
      DO i=1,n
        x(i)=x(i)-fact*z(i)
      END DO

      END

C**************************** T R I D I A *************************** 
      SUBROUTINE TRIDIA(alpha,beta,gamma,b,x,n) 
C 
C        Solution of tridiagonal systems 
C 
C     ALPHA: main diagonal (destroyed in TRIDIA). 
C     BETA: subdiagonal (first element has index 1). 
C     GAMMA: superdiagonal (first element has index 1). 
C     B: right-hand side (destroyed in TRIDIA). 
C     X: solution. 
C     N: dimension of system. 
C 
      IMPLICIT REAL*8 (a-h,o-z) 
      DIMENSION alpha(n),beta(n),gamma(n),b(n),x(n) 
C 
      DO i=2,n 
        rap = beta(i-1) / alpha(i-1) 
        alpha(i) = alpha(i) - rap*gamma(i-1) 
        b(i) = b(i) - rap*b(i-1) 
      END DO 
      x(n) = b(n) / alpha(n) 
       
      DO j=2,n 
        i = n-j+1 
        x(i) = ( b(i) - gamma(i)*x(i+1) ) / alpha(i) 
      END DO 
      RETURN 
      END


C     ************************ T R I S P L ***************************

C     Spline interpolation of a periodic function with period
C     [x(n)-x(1)] - i.e. y(n)=y(1).

C     See general comments in DSPLIN.
C     The vectors ALPHA, BETA, GAMMA, B, BZ are scratch vectors of
C     dimension at list N.
C     Required subroutine: TRIDIA, TRICYCLE
C
C        Author: A. Salin   Version 1  09/06/98
C
C      BLOCK DATA BLKTRI
C      COMMON/KODSPL/kod,k,iw,iligne
C      DATA iw,iligne,k/6,100,2/
C      END
C     ***************************************************************
      SUBROUTINE TRISPL(n,x,y,cm,alpha,beta,gamma,b,bz)

      IMPLICIT REAL*8 (a-h,o-z)
      PARAMETER (six=6.D0,trois=3.D0,zero=0.D0)
C
      DIMENSION x(n),y(n),cm(n)
      DIMENSION alpha(n),beta(n),gamma(n),b(n),bz(n)
C
      COMMON/KODSPL/kod,k,iw,iligne
C
   91 FORMAT('Error in the data for TRISPL',/,'Check abcissae:',
     1       1PD15.8,1X,' and ',D15.8,/,'Program stopped')
C
      kod=0
      c=x(2)-x(1)
      e=(y(2)-y(1))/c
      DO i=3,n
        i2=i-2
        a=c
        c=x(i)-x(i-1)
        IF(a*c.LE.zero) THEN
          kod=-1
          IF(iw.GT.0) WRITE(iw,91) x(i-1),x(i)
          STOP
        END IF
        alpha(i2)=(a+c)/trois
        beta(i2)=c/six
        gamma(i2)=beta(i2)
        d=e
        e=(y(i)-y(i-1))/c
        b(i2)=e-d
      END DO
C     
      a=c
      c=x(2)-x(1)
      alpha(n-1)=(a+c)/trois
      beta(n-1)=c/six
      gamma(n-1)=beta(n-1)
      d=e
      e=(y(2)-y(1))/c
      b(n-1)=e-d
      cbot=beta(n-1)
      ctop=cbot
C
C                                            Solve tridiagonal system
      CALL TRICYCLE(alpha,beta,gamma,cbot,ctop,b,cm(2),n-1,bz)
C
      cm(1)=cm(n)
C
      k=2
      END
