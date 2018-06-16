#include "filter.h"
#include <float.h>

#ifndef RXPANOUTIL

#define RXPANOUTIL 1


#ifdef WIN32

#elif defined __APPLE__
#else
    
#include<sys/time.h> 
#include <unistd.h>

#endif
//---------for time stat------------

#ifdef WIN32
typedef DWORD os_TIME;
#else
typedef struct timeval os_TIME;
#endif
void os_GetTime1(os_TIME* time)
{
#ifdef WIN32
    *time = GetTickCount();
#else
    struct timezone tz;
    gettimeofday(time, &tz);
#endif
}

int os_TimeDiff1(os_TIME* time1, os_TIME* time2)
{
#ifdef WIN32
    return *time1 - *time2;
#else
    return (int)((double)time1->tv_sec*1000 + ((double)time1->tv_usec)*1e-3 -
                 (double)time2->tv_sec*1000 - ((double)time2->tv_usec)*1e-3);
#endif
}

#define TIMETRACE(TEXT, CODE) { os_TIME t1,t2; os_GetTime1(&t1); CODE; \
        os_GetTime1(&t2); \
		printf("%s took %f seconds.\n",TEXT,os_TimeDiff1(&t2,&t1)/1000.0); }

int getCPUCount()
{
#ifdef WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors;
#elif defined(HW_NCPU) || defined(__APPLE__)
    // BSD and OSX like system
    int mib[2];
    int numCPUs = 1;
    size_t len = sizeof(numCPUs);

    mib[0] = CTL_HW;
    mib[1] = HW_NCPU;
    sysctl(mib, 2, &numCPUs, &len, 0, 0);
    return numCPUs;

#elif defined(_SC_NPROCESSORS_ONLN)
    // Linux and Solaris
    long nProcessorsOnline = sysconf(_SC_NPROCESSORS_ONLN);
    return nProcessorsOnline;
#else
    return 1;
#endif
}

#ifdef _WINDOWS
unsigned long long getTotalMemory()
{
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
#ifndef _WIN64
    // when compiled as 32 bit version, we can only use about 2 GB
    // even if we have more memory available on a 64 bit system
    return std::min<unsigned long long>(status.ullTotalPhys, 1500*1024*1024);
#else
    return status.ullTotalPhys;
#endif
};
#elif defined __APPLE__
unsigned long long utils::getTotalMemory()
{
    SInt32 ramSize;
    if(Gestalt(gestaltPhysicalRAMSizeInMegabytes, &ramSize)==noErr)
    {
        return ramSize * 1024 * 1024;
    }
    else
    {
        // if query was not successful return 1 GB, 
        // return 0 would result in crash in calling function
        return 1024*1024*1024;
    }
};
#else
unsigned long long getTotalMemory()
{
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}
#endif

//-----------------------for multithread seal------------------------

#define C_FACTOR        100.0
typedef struct env
{
	AlignInfo optInfo;
	double distanceComponents[2] ;

	double initialAvgFov;   // these three for fov stabilization
	double avgfovFromSAP;
	int needInitialAvgFov;
	double fcnPanoHuberSigma; // sigma for Huber M-estimator. 0 disables M-estimator
	int fcnPanoNperCP; // number of functions per control point, 1 or 2
	FILE* adjustLogFile;
};
typedef struct env env;
void  SetInvMakeParams_dist( struct fDesc *stack, struct MakeParams *mp, Image *im , Image *pn, int color )
{

    int     i;
    double    a,b;              // field of view in rad
    double      tx,ty,tpara;

    a =  DEG_TO_RAD( im->hfov );  // field of view in rad   
    b =  DEG_TO_RAD( pn->hfov );

    mp->im = im;
    mp->pn = pn;

    SetMatrix( DEG_TO_RAD( im->pitch ), 
               0.0, 
               DEG_TO_RAD( im->roll ), 
               mp->mt, 
               1 );

    // dangelo: added mercator, sinusoidal and stereographic projection
    switch (pn->format)
        {
        case _rectilinear:
            mp->distance        = (double) pn->width / (2.0 * tan(b/2.0));
            break;
        case _equirectangular:
        case _fisheye_ff:
        case _fisheye_circ:
        case _panorama:
        case _lambert:
        case _mercator:
        case _millercylindrical:
        case _sinusoidal:
        case _mirror:
            // horizontal pixels per rads
            mp->distance        = ((double) pn->width) / b;
            break;
        case _lambertazimuthal:
            tpara = 1;
            lambertazimuthal_erect(b/2.0, 0.0, &tx, &ty, & tpara);
            mp->distance = pn->width/(2.0*tx);
            break;
        case _panini:
            tpara = 1;
            panini_erect(b/2.0, 0.0, &tx, &ty, & tpara);
            mp->distance = pn->width/(2.0*tx);
            break;
        case _equipanini:
            tpara = 1;
            equipanini_erect(b/2.0, 0.0, &tx, &ty, & tpara);
            mp->distance = pn->width/(2.0*tx);
            break;
        case _panini_general:
		  // call setup_panini_general() to set distanceparam
			setup_panini_general( mp );
		  // should abort now if it returns NULL
            break;
        case _architectural:
            tpara = 1;
            arch_erect(b/2.0, 0.0, &tx, &ty, & tpara);
            mp->distance = pn->width/(2.0*tx);
            break;
        case _stereographic:
            tpara = 1;
            stereographic_erect(b/2.0, 0.0, &tx, &ty, & tpara);
            mp->distance = pn->width/(2.0*tx);
            break;
        case _trans_mercator:
            tpara = 1;
            transmercator_erect(b/2.0, 0.0, &tx, &ty, & tpara);
            mp->distance = pn->width/(2.0*tx);
            break;
        case _albersequalareaconic:
            mp->distance = 1.0;
            //albersequalareaconic_erect(1.924913116, -PI/2.0, &tx, &ty, mp);  //b/2.0
            albersequalareaconic_distance(&tx, mp);
            mp->distance = pn->width/(2.0*tx);
            break;
        case _equisolid:
            mp->distance  = (double) pn->width / (4.0 * sin(b/4.0));
            break;
        case _orthographic:
            mp->distance  = (double) pn->width / (2.0 * sin(b/2.0));
            break;
        case _thoby:
            mp->distance  = (double) pn->width / (2.0 * THOBY_K1_PARM * sin(b * THOBY_K2_PARM/2.0));
            break;
        case _biplane:
            biplane_distance(pn->width,b,mp);
            break;
        case _triplane:
            triplane_distance(pn->width,b,mp);
            break;    default:
            // unknown
            PrintError ("SetInvMakeParams: Unsupported panorama projection");
            // no way to report an error back to the caller...
            mp->distance = 1;
            break;
        }

    // calculate final scaling factor, that reverses the mp->distance
    // scaling and applies the required output scaling factor
    switch (im->format)
        {
        case _rectilinear:
            // calculate distance for this projection
            mp->scale[0] = (double) im->width / (2.0 * tan(a/2.0)) / mp->distance;
            break;
        case _equirectangular:
        case _panorama:
        case _fisheye_ff:
        case _fisheye_circ:
        case _mercator:
        case _sinusoidal:
            mp->scale[0] = ((double) im->width) / a / mp->distance;
            break;
        case _equisolid:
        case _mirror:
            mp->scale[0] = (double) im->width / (4.0 * sin(a/4.0)) / mp->distance;
            break;
        case _orthographic:
            mp->scale[0] = (double) im->width / (2.0 * sin(a/2.0)) / mp->distance;
            break;
        case _thoby:
            mp->scale[0] = (double) im->width / (2.0 * THOBY_K1_PARM * sin(a * THOBY_K2_PARM/2.0)) / mp->distance;
            break;
        case _stereographic:
            mp->scale[0] = (double) im->width / (4.0 * tan(a/4.0)) / mp->distance;
            break;
        default:
            PrintError ("SetInvMakeParams: Unsupported input image projection");
            // no way to report an error back to the caller...
            mp->scale[0] = 1;
            break;
        }
    mp->scale[1]    = mp->scale[0];

    mp->shear[0]  = im->cP.shear_x / im->height;
    mp->shear[1]  = im->cP.shear_y / im->width;
  

  mp->tilt[0] = DEG_TO_RAD(im->cP.tilt_x);
  mp->tilt[1] = DEG_TO_RAD(im->cP.tilt_y);
  mp->tilt[2] = DEG_TO_RAD(im->cP.tilt_z);
  mp->tilt[3] = im->cP.tilt_scale;
  
  mp->trans[0] = im->cP.trans_x;
  mp->trans[1] = im->cP.trans_y;
  mp->trans[2] = im->cP.trans_z;

  mp->test[0] = im->cP.test_p0;
  mp->test[1] = im->cP.test_p1;
  mp->test[2] = im->cP.test_p2;
  mp->test[3] = im->cP.test_p3;

  //  panoAdjustPrintMakeParams("Inverse 20",mp,im);

  mp->scale[0] = 1.0 / mp->scale[0];
  mp->scale[1]  = mp->scale[0];
  mp->horizontal  = -im->cP.horizontal_params[color];
  mp->vertical  = -im->cP.vertical_params[color];
  for(i=0; i<4; i++)
    mp->rad[i]  = im->cP.radial_params[color][i];
  mp->rad[5] = im->cP.radial_params[color][4];
  
  switch( im->cP.correction_mode & 3 )
    {
    case correction_mode_radial: mp->rad[4] = ((double)(im->width < im->height ? im->width : im->height) ) / 2.0;break;
    case correction_mode_vertical: 
    case correction_mode_deregister: mp->rad[4] = ((double) im->height) / 2.0;break;
    }

  mp->rot[0]    = mp->distance * PI;                // 180 in screenpoints
  mp->rot[1]    = im->yaw *  mp->distance * PI / 180.0;       //    rotation angle in screenpoints

  mp->perspect[0] = (void*)(mp->mt);
  mp->perspect[1] = (void*)&(mp->distance);

  //  panoAdjustPrintMakeParams("Invert 30",mp,im);

  i = 0;  // Stack counter
    

  if( im->cP.shear )
  {
    SetDesc( stack[i],shearInv,      mp->shear   ); i++;
  }
    
  if ( im->cP.horizontal )
  {
    SetDesc(stack[i],horiz,       &(mp->horizontal)); i++;
  }

  if (  im->cP.vertical)
  {
    SetDesc(stack[i],vert,        &(mp->vertical));   i++;
  }

  if( im->cP.tilt )
  {
    SetDesc( stack[i],tiltForward,      mp   ); i++;
  }

  // Perform radial correction

  if(   im->cP.radial )
  {
    switch( im->cP.correction_mode & 3)
    {
      case correction_mode_radial:   SetDesc(stack[i],inv_radial,mp->rad);  i++; break;
      case correction_mode_vertical: SetDesc(stack[i],inv_vertical,mp->rad);  i++; break;
      case correction_mode_deregister: break;
    }
  }
  
  SetDesc(  stack[i], resize,       mp->scale   ); i++; // Scale image

  //  printf("values %d %d\n", i, im->format);

  
  if(im->format     == _rectilinear)                  // rectilinear image
  {
    SetDesc(stack[i], sphere_tp_rect,   &(mp->distance) ); i++; // Convert rectilinear to spherical
  }
  else if (im->format   == _panorama)                 //  pamoramic image
  {
    SetDesc(stack[i], sphere_tp_pano,   &(mp->distance) ); i++; // Convert panoramic to spherical
  }
  else if (im->format   == _equirectangular)          //  equirectangular image
  {
    SetDesc(stack[i], sphere_tp_erect,  &(mp->distance) ); i++; // Convert equirectangular to spherical
  }
  else if (im->format   == _mirror)                   //  Mirror image
  {
    SetDesc(stack[i],   sphere_tp_mirror,        &(mp->distance) ); i++; // Convert mirror to spherical
  }
  else if (im->format   == _equisolid)                //  Fisheye equisolid image
  {

    SetDesc(stack[i], erect_lambertazimuthal,  &(mp->distance) ); i++; // Convert lambert to equirectangular
    SetDesc(stack[i], sphere_tp_erect,  &(mp->distance) ); i++; // Convert equirectangular to spherical
    //SetDesc(stack[i], sphere_tp_equisolid,  &(mp->distance) ); i++; // Convert equisolid to spherical
  }
  else if (im->format   == _orthographic)             //  Fisheye orthographic image
  {
    SetDesc(stack[i], sphere_tp_orthographic,  &(mp->distance) ); i++; // Convert orthographic to spherical
  }
  else if (im->format   == _thoby)             //  thoby projected image
  {
    SetDesc(stack[i], sphere_tp_thoby,  &(mp->distance) ); i++; // Convert thoby to spherical
  }
  else if (im->format   == _stereographic)             //  Fisheye stereographic image
  {
    SetDesc(stack[i], erect_stereographic,  &(mp->distance) ); i++; // Convert stereographic to spherical
    SetDesc(stack[i], sphere_tp_erect,  &(mp->distance) ); i++; // Convert equirectangular to spherical
  }

  //  printf("values %d %d\n", i, im->format);  
  SetDesc(  stack[i], persp_sphere,   mp->perspect  ); i++; // Perspective Control spherical Image
  SetDesc(  stack[i], erect_sphere_tp,  &(mp->distance) ); i++; // Convert spherical image to equirect.
  SetDesc(  stack[i], rotate_erect,   mp->rot     ); i++; // Rotate equirect. image horizontally

  if( im->cP.trans )
  {
    SetDesc( stack[i], plane_transfer_from_camera,      mp   ); i++;
  }

  // THESE ARE ALL FORWARD transforms
  if(pn->format == _rectilinear)                  // rectilinear panorama
  {
    SetDesc(stack[i], rect_erect,   &(mp->distance) ); i++; // Convert equirectangular to rectilinear
  }
  else if(pn->format == _panorama)
  {
    SetDesc(stack[i], pano_erect,   &(mp->distance) ); i++; // Convert equirectangular to Cylindrical panorama
  }
  else if(pn->format == _fisheye_circ || pn->format == _fisheye_ff )
  {
    SetDesc(stack[i], sphere_tp_erect,    &(mp->distance) ); i++; // Convert equirectangular to spherical
  }
  else if(pn->format == _mercator)
  {
    SetDesc(stack[i], mercator_erect,   &(mp->distance) ); i++; // Convert equirectangular to mercator
  }
  else if(pn->format == _millercylindrical)
  {
    SetDesc(stack[i], millercylindrical_erect,    &(mp->distance) ); i++; // Convert equirectangular to miller cylindrical
  }
  else if(pn->format == _panini)
  {
    SetDesc(stack[i], panini_erect,  &(mp->distance) ); i++; // Convert panini to sphere
  }
  else if(pn->format == _equipanini)
  {
    SetDesc(stack[i], equipanini_erect,  &(mp->distance) ); i++; // Convert equi panini to sphere
  }
  else if(pn->format == _panini_general)
  {
    SetDesc(stack[i],  panini_general_erect,  mp ); i++; // Convert general panini to sphere
  }
  else if(pn->format == _architectural)
  {
    SetDesc(stack[i], arch_erect,   &(mp->distance) ); i++; // Convert arch to sphere
  }
  else if(pn->format == _lambert)
  {
    SetDesc(stack[i], lambert_erect,    &(mp->distance) ); i++; // Convert equirectangular to lambert
  }
  else if(pn->format == _lambertazimuthal)
  {
    SetDesc(stack[i], lambertazimuthal_erect,   &(mp->distance) ); i++; // Convert equirectangular to lambert azimuthal
  }
  else if(pn->format == _trans_mercator)
  {
    SetDesc(stack[i], transmercator_erect,    &(mp->distance) ); i++; // Convert equirectangular to transverse mercator
  }
  else if(pn->format == _mirror)
  {
    SetDesc(stack[i], mirror_erect,    &(mp->distance) ); i++; // Convert equirectangular to mirror
  }
  else if(pn->format == _stereographic)
  {
    SetDesc(stack[i], stereographic_erect,    &(mp->distance) ); i++; // Convert equirectangular to stereographic
  }
    else if(pn->format == _sinusoidal)
  {
    SetDesc(stack[i], sinusoidal_erect,   &(mp->distance) ); i++; // Convert equirectangular to sinusoidal
  }
  else if(pn->format == _albersequalareaconic)
  {
    SetDesc(stack[i], albersequalareaconic_erect,   mp  ); i++; // Convert equirectangular to albersequalareaconic
  }
  else if(pn->format == _equisolid )
  {
    SetDesc(stack[i], sphere_tp_erect,    &(mp->distance) ); i++; // Convert equirectangular to spherical
    SetDesc(stack[i], equisolid_sphere_tp,    &(mp->distance) ); i++; // Convert spherical to equisolid
  }
  else if(pn->format == _orthographic )
  {
    SetDesc(stack[i], sphere_tp_erect,    &(mp->distance) ); i++; // Convert equirectangular to spherical
    SetDesc(stack[i], orthographic_sphere_tp,    &(mp->distance) ); i++; // Convert spherical to orthographic
  }
  else if(pn->format == _thoby )
  {
    SetDesc(stack[i], sphere_tp_erect,    &(mp->distance) ); i++; // Convert equirectangular to spherical
    SetDesc(stack[i], thoby_sphere_tp,    &(mp->distance) ); i++; // Convert spherical to thoby
  }
  else if(pn->format == _biplane)
  {
    SetDesc(stack[i], biplane_erect, mp ); i++;  // Convert equirectangular to biplane
  }
  else if(pn->format == _triplane)
  {
    SetDesc(stack[i], triplane_erect, mp ); i++;  // Convert equirectangular to biplane
  }  else if(pn->format == _equirectangular) 
  {
    // no conversion needed   
  }
  else 
  {
    PrintError("Projection type %d not supported, using equirectangular", pn->format);
  }
  
  stack[i].func = (trfn)NULL;
}
 void panoSetMetadataDefaults_dist(pano_ImageMetadata *m)
{

    bzero(m, sizeof(*m));

    // These are "meaningful defaults 

    m->xPixelsPerResolution = PANO_DEFAULT_PIXELS_PER_RESOLUTION;
    m->yPixelsPerResolution = PANO_DEFAULT_PIXELS_PER_RESOLUTION;
    m->resolutionUnits = 2;//PANO_DEFAULT_TIFF_RESOLUTION_UNITS;


    m->rowsPerStrip =1; // THis will speed up processing of TIFFs as only one line
                        // at a time needs to be read

    m->compression.type = 32946;//PANO_DEFAULT_TIFF_COMPRESSION; 

}

void SetImageDefaults_dist(Image *im){
        im->data                = NULL;
        im->bytesPerLine        = 0;
        im->width               = 0;
        im->height              = 0;
        im->dataSize            = 0;
        im->bitsPerPixel        = 0;
        im->format              = 0;
	im->formatParamCount    = 0;
	bzero(im->formatParam, sizeof(im->formatParam));
	im->precomputedCount    = 0;
	bzero(im->precomputedValue, sizeof(im->precomputedValue));
        im->dataformat          = _RGB;
        im->hfov                = 0.0;
        im->yaw                 = 0.0;
        im->pitch               = 0.0;
        im->roll                = 0.0;
        SetCorrectDefaults( &(im->cP) );
        *(im->name)             = 0;
        im->selection.top       = 0;
        im->selection.bottom    = 0;
        im->selection.left      = 0;
        im->selection.right     = 0;
        im->cropInformation.cropped_height  = 0;
        im->cropInformation.cropped_width   = 0;
        im->cropInformation.full_height     = 0;
        im->cropInformation.full_width      = 0;
        im->cropInformation.x_offset        = 0;
        im->cropInformation.y_offset        = 0;
        panoSetMetadataDefaults_dist(&im->metadata);
}
double distSphere_dist( int num ,env* e){
        double          x, y ;  // Coordinates of control point in panorama
        double          w2, h2;
        int j;
        Image sph;
        int n[2];
        struct  MakeParams      mp;
        struct  fDesc           stack[15];
        CoordInfo b[2];
        CoordInfo cp;
        double lat[2], lon[2];  // latitude & longitude
        double dlon;
        double dangle;
        double dist;
        double radiansToPixelsFactor;

        // Factor to convert angular error in radians to equivalent in pixels
		
        radiansToPixelsFactor = e->optInfo.pano.width / (e->optInfo.pano.hfov * (PI/180.0));
        
        // Get image position in imaginary spherical image
        
        SetImageDefaults_dist( &sph );
        
        sph.width                       = 360;
        sph.height                      = 180;
        sph.format                      = _equirectangular;
        sph.hfov                        = 360.0;
        
        n[0] = e->optInfo.cpt[num].num[0];
        n[1] = e->optInfo.cpt[num].num[1];
        
        // Calculate coordinates using equirectangular mapping to get longitude/latitude

        for(j=0; j<2; j++){
                SetInvMakeParams_dist( stack, &mp, &e->optInfo.im[ n[j] ], &sph, 0 );
                
                h2      = (double)e->optInfo.im[ n[j] ].height / 2.0 - 0.5;
                w2      = (double)e->optInfo.im[ n[j] ].width  / 2.0 - 0.5;
                
                
                execute_stack_new(      (double)e->optInfo.cpt[num].x[j] - w2,            // cartesian x-coordinate src
                                                (double)e->optInfo.cpt[num].y[j] - h2,            // cartesian y-coordinate src
                                                &x, &y, stack);

                x = DEG_TO_RAD( x ); 
                y = DEG_TO_RAD( y ) + PI/2.0;

                // x is now in the range -PI to +PI, and y is 0 to PI
                lat[j] = y;
                lon[j] = x;

                b[j].x[0] =   sin(x) * sin( y );
                b[j].x[1] =   cos( y );
                b[j].x[2] = - cos(x) * sin(y);
        }

        dlon = lon[0]-lon[1];
        if (dlon < -PI) dlon += 2.0*PI;
        if (dlon > PI) dlon -= 2.0*PI;
        e->distanceComponents[0] = (dlon*sin(0.5*(lat[0]+lat[1]))) * radiansToPixelsFactor;
        e->distanceComponents[1] = (lat[0]-lat[1]) * radiansToPixelsFactor;

        // The original acos formulation (acos(SCALAR_PRODUCT(&b[0],&b[1]))
        // is inaccurate for angles near 0, because it essentially requires finding eps
        // based on the value of 1-eps^2/2.  The asin formulation is much more
        // accurate under these conditions.

        CROSS_PRODUCT(&b[0],&b[1],&cp);
        dangle = asin(ABS_VECTOR(&cp));
        if (SCALAR_PRODUCT(&b[0],&b[1]) < 0.0) dangle = PI - dangle;
        dist = dangle * radiansToPixelsFactor;
        
        // Diagnostics to help debug various calculation errors.
        // Do not delete this code --- it has been needed surprisingly often.
#if 0   
        {       double olddist;
                olddist = acos( SCALAR_PRODUCT( &b[0], &b[1] ) ) * radiansToPixelsFactor;
//              if (adjustLogFile != 0 && abs(dist-olddist) > 1.0) {
                if (adjustLogFile != 0 && num < 5) {
                        fprintf(adjustLogFile,"***** DIST ***** dCoord = %g %g, lonlat0 = %g %g, lonlat1 = %g %g, dist=%g, olddist=%g, sumDcoordSq=%g, distSq=%g\n",
                                                                  distanceComponents[0],distanceComponents[1],lon[0],lat[0],lon[1],lat[1],dist,olddist,
                                                                  distanceComponents[0]*distanceComponents[0]+distanceComponents[1]*distanceComponents[1],dist*dist);
                }
        }
#endif

        return dist;
}


double rectDistSquared_dist( int num ,env* e) 
{
        double          x[2], y[2];                             // Coordinates of control point in panorama
        double          w2, h2;
        int j, n[2];
        double result;

        struct  MakeParams      mp;
        struct  fDesc           stack[15];

        

        n[0] = e->optInfo.cpt[num].num[0];
        n[1] = e->optInfo.cpt[num].num[1];
        
        // Calculate coordinates x/y in panorama

        for(j=0; j<2; j++)
        {
                SetInvMakeParams_dist( stack, &mp, &e->optInfo.im[ n[j] ], &e->optInfo.pano, 0 );
                
                h2      = (double)e->optInfo.im[ n[j] ].height / 2.0 - 0.5;
                w2      = (double)e->optInfo.im[ n[j] ].width  / 2.0 - 0.5;
                

                execute_stack_new(      (double)e->optInfo.cpt[num].x[j] - w2,            // cartesian x-coordinate src
                                                (double)e->optInfo.cpt[num].y[j] - h2,            // cartesian y-coordinate src
                                                &x[j], &y[j], stack);
                // test to check if inverse works
#if 0
                {
                        double xt, yt;
                        struct  MakeParams      mtest;
                        struct  fDesc           stacktest[15];
                        SetMakeParams( stacktest, &mtest, &e->optInfo.im[ n[j] ], &e->optInfo.pano, 0 );
                        execute_stack_new(      x[j],           // cartesian x-coordinate src
                                                        y[j],           // cartesian y-coordinate src
                                                &xt, &yt, stacktest);
                        
                        printf("x= %lg, y= %lg,  xb = %lg, yb = %lg \n", e->optInfo.cpt[num].x[j], e->optInfo.cpt[num].y[j], xt+w2, yt+h2);  
                        
                }
#endif
        }
        
        
//      printf("Coordinates 0:   %lg:%lg        1:      %lg:%lg\n",x[0] + g->pano->width/2,y[0]+ g->pano->height/2, x[1] + g->pano->width/2,y[1]+ g->pano->height/2);


        // take care of wrapping and points at edge of panorama
        
        if( e->optInfo.pano.hfov == 360.0 )
        {
                double delta = abs( x[0] - x[1] );
                
                if( delta > e->optInfo.pano.width / 2 )
                {
                        if( x[0] < x[1] )
                                x[0] += e->optInfo.pano.width;
                        else
                                x[1] += e->optInfo.pano.width;
                }
        }


        switch( e->optInfo.cpt[num].type )                // What do we want to optimize?
        {
                case 1:                 // x difference
                        result = ( x[0] - x[1] ) * ( x[0] - x[1] );
                        break;
                case 2:                 // y-difference
                        result =  ( y[0] - y[1] ) * ( y[0] - y[1] );
                        break;
                default:
                        result = ( y[0] - y[1] ) * ( y[0] - y[1] ) + ( x[0] - x[1] ) * ( x[0] - x[1] ); // square of distance
                        e->distanceComponents[0] = y[0] - y[1];
                        e->distanceComponents[1] = x[0] - x[1];

                        break;
        }
        

        return result;
}


void pt_getXY_dist(int n, double x, double y, double *X, double *Y,env* e){
        struct  MakeParams      mp;
        struct  fDesc           stack[15];
        double h2,w2;

        SetInvMakeParams_dist( stack, &mp, &e->optInfo.im[ n ], &e->optInfo.pano, 0 );
        h2      = (double)e->optInfo.im[ n ].height / 2.0 - 0.5;
        w2      = (double)e->optInfo.im[ n ].width  / 2.0 - 0.5;


        execute_stack_new(      x - w2, y - h2, X, Y, stack);
}

// Return distance of points from a line
// The line through the two farthest apart points is calculated
// Returned is the sum distance squared of the other two points from the line
double distsqLine_dist(int N0, int N1,env* e){
        double x[4],y[4], del, delmax, A, B, C, mu, d0, d1;
        int n0, n1, n2=-1, n3=-1, i, k;

        pt_getXY_dist(e->optInfo.cpt[N0].num[0], (double)e->optInfo.cpt[N0].x[0], (double)e->optInfo.cpt[N0].y[0], &x[0], &y[0],e);
        pt_getXY_dist(e->optInfo.cpt[N0].num[1], (double)e->optInfo.cpt[N0].x[1], (double)e->optInfo.cpt[N0].y[1], &x[1], &y[1],e);
        pt_getXY_dist(e->optInfo.cpt[N1].num[0], (double)e->optInfo.cpt[N1].x[0], (double)e->optInfo.cpt[N1].y[0], &x[2], &y[2],e);
        pt_getXY_dist(e->optInfo.cpt[N1].num[1], (double)e->optInfo.cpt[N1].x[1], (double)e->optInfo.cpt[N1].y[1], &x[3], &y[3],e);

        delmax = 0.0;
        n0 = 0; n1 = 1;

        for(i=0; i<4; i++){
                for(k=i+1; k<4; k++){
                        del = (x[i]-x[k])*(x[i]-x[k])+(y[i]-y[k])*(y[i]-y[k]);
                        if(del>delmax){
                                n0=i; n1=k; delmax=del;
                        }
                }
        }
        if(delmax==0.0) return 0.0;

        for(i=0; i<4; i++){
                if(i!= n0 && i!= n1){
                        n2 = i;
                        break;
                }
        }
        for(i=0; i<4; i++){
                if(i!= n0 && i!= n1 && i!=n2){
                        n3 = i;
                }
        }


        A=y[n1]-y[n0]; B=x[n0]-x[n1]; C=y[n0]*(x[n1]-x[n0])-x[n0]*(y[n1]-y[n0]);

        mu=1.0/sqrt(A*A+B*B);

        d0 = (A*x[n2]+B*y[n2]+C)*mu;
        d1 = (A*x[n3]+B*y[n3]+C)*mu;
        e->distanceComponents[0] = d0;
        e->distanceComponents[1] = d1;

        return d0*d0 + d1*d1;

}



int     EvaluateControlPointErrorAndComponents_dist ( int num, double *errptr, double errComponent[2],env* e) {
        int j;
        int result;
        switch(e->optInfo.cpt[num].type){
                case 0: // normal control points
                        // Jim May 2004. 
                        // Optimizing cylindrical and rectilinear projection by calculating 
                        // distance error in pixel coordinates of the rendered image.
                        // When using angular (spherical) distance for these projections, 
                        // larger errors are generated the further control points are from 
                        // the center.
                        // In theory by optimizing in pixel coordinates all errors will be 
                        // distributed over the image.  This is true.
                        // In practice I have found that optimize large field of view 
                        // rectilinear projection images failed to resolve nicely if the 
                        // parameters were not very close to start with.  I leave the 
                        // code here for others to play with and maybe get better results.
                /*  if(e->optInfo.pano.format == _rectilinear || g->pano.format == _panorama)
                        {
                                *errptr = sqrt(rectDistSquared(num));
                                errComponent[0] = distanceComponents[0];
                                errComponent[1] = distanceComponents[1];
                                result = 0;
                                break;
                        }
                        else //  _equirectangular, fisheye, spherical, mirror
                        {  */
                                *errptr = distSphere_dist(num,e);
                                errComponent[0] = e->distanceComponents[0];
                                errComponent[1] = e->distanceComponents[1];
                                result = 0;
                                break;
                        //}
                case 1: // vertical
                case 2: // horizontal
                                *errptr = sqrt(rectDistSquared_dist(num,e));
                                errComponent[0] = *errptr;
                                errComponent[1] = 0.0;
                                result = 0;
                                break;
                default:// t+ controls = lines = sets of two control point pairs
                                *errptr = 0.0;  // in case there is no second pair
                                errComponent[0] = 0.0;
                                errComponent[1] = 0.0;
                                result = 1;
                                for(j=0; j<e->optInfo.numPts; j++){
                                        if(j!=num && e->optInfo.cpt[num].type == e->optInfo.cpt[j].type){
                                                *errptr = sqrt(distsqLine_dist(num,j,e));
//                                              errComponent[0] = *errptr;
//                                              errComponent[1] = 0.0;
                                                errComponent[0] = e->distanceComponents[0];
                                                errComponent[1] = e->distanceComponents[1];
                                                result = 0;
                                                break;
                                        }
                                }
                                break;
        }
        return result;
}


// Set global preferences structures using LM-params

int     SetAlignParams_dist( double *x ,env* e)
{
        // Set Parameters
        int i,j,k;
        double sumfov = 0.0;
        
        j = 0;
        for( i=0; i<e->optInfo.numIm; i++ ){

                if( (k = e->optInfo.opt[i].yaw) > 0 ){
                        if( k == 1 ){   e->optInfo.im[i].yaw  = x[j++]; NORM_ANGLE( e->optInfo.im[i].yaw );
                        }else{  e->optInfo.im[i].yaw  = e->optInfo.im[k-2].yaw; }
                }
                if( (k = e->optInfo.opt[i].pitch) > 0 ){
                        if( k == 1 ){   e->optInfo.im[i].pitch  =       x[j++]; NORM_ANGLE( e->optInfo.im[i].pitch );
                        }else{  e->optInfo.im[i].pitch  =       e->optInfo.im[k-2].pitch;       }
                }
                if( (k = e->optInfo.opt[i].roll) > 0 ){
                        if( k == 1 ){   e->optInfo.im[i].roll  =        x[j++]; NORM_ANGLE( e->optInfo.im[i].roll );
                        }else{  e->optInfo.im[i].roll  =        e->optInfo.im[k-2].roll;        }
                }
                if( (k = e->optInfo.opt[i].hfov) > 0 ){
                        if( k == 1 ){   
                                e->optInfo.im[i].hfov  =        x[j++]; 
                                if( e->optInfo.im[i].hfov < 0.0 )
                                        e->optInfo.im[i].hfov = - e->optInfo.im[i].hfov;
                        }else{  e->optInfo.im[i].hfov  = e->optInfo.im[k-2].hfov; }
                }
                sumfov += e->optInfo.im[i].hfov;
                if( (k = e->optInfo.opt[i].a) > 0 ){
                        if( k == 1 ){ e->optInfo.im[i].cP.radial_params[0][3]  =  x[j++] / C_FACTOR;
                        }else{  e->optInfo.im[i].cP.radial_params[0][3] = e->optInfo.im[k-2].cP.radial_params[0][3];}
                }
                if( (k = e->optInfo.opt[i].b) > 0 ){
                        if( k == 1 ){ 
                          e->optInfo.im[i].cP.radial_params[0][2]  =  x[j++] / C_FACTOR;
                        }else{  
                          e->optInfo.im[i].cP.radial_params[0][2] = e->optInfo.im[k-2].cP.radial_params[0][2];
                        }
                }
                if( (k = e->optInfo.opt[i].c) > 0 ){
                        if( k == 1 ){ e->optInfo.im[i].cP.radial_params[0][1]  =  x[j++] / C_FACTOR;
                        }else{  e->optInfo.im[i].cP.radial_params[0][1] = e->optInfo.im[k-2].cP.radial_params[0][1];}
                }
                if( (k = e->optInfo.opt[i].d) > 0 ){
                        if( k == 1 ){ e->optInfo.im[i].cP.horizontal_params[0]  = x[j++];
                        }else{  e->optInfo.im[i].cP.horizontal_params[0] = e->optInfo.im[k-2].cP.horizontal_params[0];}
                }
                if( (k = e->optInfo.opt[i].e) > 0 ){
                        if( k == 1 ){ e->optInfo.im[i].cP.vertical_params[0]  =   x[j++];
                        }else{  e->optInfo.im[i].cP.vertical_params[0] = e->optInfo.im[k-2].cP.vertical_params[0];}
                }
                // tilt
                if( (k = e->optInfo.opt[i].tiltXopt) > 0 ){
                    if( k == 1 ){ e->optInfo.im[i].cP.tilt_x  = x[j++]; //NORM_ANGLE_RAD(e->optInfo.im[i].cP.tilt_x);
                    }else{  e->optInfo.im[i].cP.tilt_x = e->optInfo.im[k-2].cP.tilt_x;}
                }
                if( (k = e->optInfo.opt[i].tiltYopt) > 0 ){
                    if( k == 1 ){ e->optInfo.im[i].cP.tilt_y  = x[j++]; //NORM_ANGLE_RAD(e->optInfo.im[i].cP.tilt_y);
                        }else{  e->optInfo.im[i].cP.tilt_y = e->optInfo.im[k-2].cP.tilt_y;}
                }
                if( (k = e->optInfo.opt[i].tiltZopt) > 0 ){
                    if( k == 1 ){ e->optInfo.im[i].cP.tilt_z  =x[j++]; //NORM_ANGLE_RAD(e->optInfo.im[i].cP.tilt_z);
                        }else{  e->optInfo.im[i].cP.tilt_z = e->optInfo.im[k-2].cP.tilt_z;}
                }
                if( (k = e->optInfo.opt[i].tiltScaleOpt) > 0 ){
                    if( k == 1 ) { 
                        e->optInfo.im[i].cP.tilt_scale  = x[j++]; 
                        if (e->optInfo.im[i].cP.tilt_scale == 0) {
                            e->optInfo.im[i].cP.tilt_scale = 0.001; //make sure it never becomes zero
                        }
                        e->optInfo.im[i].cP.tilt_scale = fabs(e->optInfo.im[i].cP.tilt_scale);
                        /*
                        if (e->optInfo.im[i].cP.tilt_scale > 10) {
                            e->optInfo.im[i].cP.tilt_scale = 10; //make sure it never gets out of control
                        }
                        */
                    } else{  e->optInfo.im[i].cP.tilt_scale = e->optInfo.im[k-2].cP.tilt_scale;}
                }
                // translate
                if( (k = e->optInfo.opt[i].transXopt) > 0 ){
                        if( k == 1 ){ e->optInfo.im[i].cP.trans_x  =      x[j++];
                        }else{  e->optInfo.im[i].cP.trans_x = e->optInfo.im[k-2].cP.trans_x;}
                }
                if( (k = e->optInfo.opt[i].transYopt) > 0 ){
                        if( k == 1 ){ e->optInfo.im[i].cP.trans_y  =      x[j++];
                        }else{  e->optInfo.im[i].cP.trans_y = e->optInfo.im[k-2].cP.trans_y;}
                }
                if( (k = e->optInfo.opt[i].transZopt) > 0 ){
                        if( k == 1 ){ e->optInfo.im[i].cP.trans_z  =      x[j++];
                        }else{  e->optInfo.im[i].cP.trans_z = e->optInfo.im[k-2].cP.trans_z;}
                }
                // test
                if( (k = e->optInfo.opt[i].testP0opt) > 0 ){
                        if( k == 1 ){ e->optInfo.im[i].cP.test_p0  =      x[j++];
                        }else{  e->optInfo.im[i].cP.test_p0 = e->optInfo.im[k-2].cP.test_p0;}
                }
                if( (k = e->optInfo.opt[i].testP1opt) > 0 ){
                        if( k == 1 ){ e->optInfo.im[i].cP.test_p1  =      x[j++];
                        }else{  e->optInfo.im[i].cP.test_p1 = e->optInfo.im[k-2].cP.test_p1;}
                }
                if( (k = e->optInfo.opt[i].testP2opt) > 0 ){
                        if( k == 1 ){ e->optInfo.im[i].cP.test_p2  =      x[j++];
                        }else{  e->optInfo.im[i].cP.test_p2 = e->optInfo.im[k-2].cP.test_p2;}
                }
                if( (k = e->optInfo.opt[i].testP3opt) > 0 ){
                        if( k == 1 ){ e->optInfo.im[i].cP.test_p3  =      x[j++];
                        }else{  e->optInfo.im[i].cP.test_p3 = e->optInfo.im[k-2].cP.test_p3;}
                }

                //shear
                if( (k = e->optInfo.opt[i].shear_x) > 0 ){
                        if( k == 1 ){ e->optInfo.im[i].cP.shear_x  =      x[j++];
                        }else{  e->optInfo.im[i].cP.shear_x = e->optInfo.im[k-2].cP.shear_x;}
                }

                if( (k = e->optInfo.opt[i].shear_y) > 0 ){
                        if( k == 1 ){ e->optInfo.im[i].cP.shear_y  =      x[j++];
                        }else{  e->optInfo.im[i].cP.shear_y = e->optInfo.im[k-2].cP.shear_y;}
                }

                
                e->optInfo.im[i].cP.radial_params[0][0] = 1.0 - ( e->optInfo.im[i].cP.radial_params[0][3]
                                                                                                                + e->optInfo.im[i].cP.radial_params[0][2]
                                                                                                                + e->optInfo.im[i].cP.radial_params[0][1] ) ;

        }
        e->avgfovFromSAP = sumfov / e->optInfo.numIm;
        if( j != e->optInfo.numParam )
                return -1;
        else
                return 0;

}
double huber_dist(double x, double sigma)
{
    if (abs(x) < sigma)
        return x;
    else
        return sqrt(2.0*sigma*abs(x) - sigma*sigma);
}
void optCopy(AlignInfo* dst,AlignInfo* src)
{
	
	dst->data=NULL;
	dst->numParam=src->numParam;
	dst->numIm=src->numIm;
	dst->numPts=src->numPts;
	dst->nt=src->nt;
	dst->pano=src->pano;
	dst->st=src->st;

	dst->im      = (Image*)          malloc( src->numIm   * sizeof(Image));
	dst->cpt     = (controlPoint*)   malloc( src->numPts  * sizeof(controlPoint));
	dst->opt     = (optVars*)        malloc( src->numIm   * sizeof(optVars) );
	dst->t       = (triangle*)       malloc( src->nt      * sizeof(triangle ));  
	dst->cim     = (CoordInfo*)      malloc( src->numIm   * sizeof(CoordInfo) );

	memcpy(dst->im,src->im,src->numIm   * sizeof(Image));
	memcpy(dst->cpt,src->cpt,src->numPts  * sizeof(controlPoint));
	memcpy(dst->opt,src->opt,src->numIm   * sizeof(optVars) );
	memcpy(dst->t ,src->t,src->nt      * sizeof(triangle ));
	memcpy(dst->cim  ,src->cim, src->numIm   * sizeof(CoordInfo));


	

}
void optFree(AlignInfo* g)
{
	free(g->im);
	free(g->cpt);
	free(g->opt);
	free(g->t);
	free(g->cim);
}
int fcnPano_dist(int m, int n, double x[], double fvec[], int *iflag,AlignInfo	g)
{
        int i;
        static int numIt;
        double result;
        int iresult;
        double junk;
        double junk2[2];
		env e;
		optCopy(&(e.optInfo),&g);
		e.initialAvgFov=0;
		e.fcnPanoHuberSigma=0;
		e.fcnPanoNperCP = 1;
		e.adjustLogFile=0;
		e.needInitialAvgFov=0;
		e.fcnPanoHuberSigma=0;
        
        if( *iflag == -100 ){ // reset
                numIt = 0;
                e.needInitialAvgFov = 1;
                infoDlg ( _initProgress, "Optimizing Variables" );
#if ADJUST_LOGGING_ENABLED
                if ((adjustLogFile = fopen(ADJUST_LOG_FILENAME,"a")) <= 0) {
                        PrintError("Cannot Open Log File");
                        adjustLogFile = 0;
                }
#endif
                return 0;
        }
        if( *iflag == -99 ){ // 
                infoDlg ( _disposeProgress, "" );
                if (e.adjustLogFile != 0) {
                        result = 0.0;
                        for( i=0; i < m; i++)
                        {
                                result += fvec[i]*fvec[i] ;
                        }
                        result = sqrt( result/ (double)m ) * sqrt((double)e.fcnPanoNperCP); // to approximate total distance vs dx, dy
                        fprintf(e.adjustLogFile,"At iflag=-99 (dispose), NperCP=%d, m=%d, n=%d, err = %g, x= \n",
                                              e.fcnPanoNperCP,m,n,result);
                        for (i=0; i<n; i++) {
                                fprintf(e.adjustLogFile,"\t%20.10g",x[i]);
                        }
                        fprintf(e.adjustLogFile,"\n   fvec = \n");
                        for (i=0; i<m; i++) {
                                fprintf(e.adjustLogFile,"\t%20.10g",fvec[i]);
                                if (((i+1) % e.fcnPanoNperCP) == 0) fprintf(e.adjustLogFile,"\n");
                        }
                        fprintf(e.adjustLogFile,"\n");
                        fclose(e.adjustLogFile);
                }
                return 0;
        }


        if( *iflag == 0 )
        {
                char message[256];
                
                result = 0.0;
                for( i=0; i < m; i++)
                {
                        result += fvec[i]*fvec[i] ;
                }
                result = sqrt( result/ (double)m ) * sqrt((double)e.fcnPanoNperCP); // to approximate total distance vs dx, dy

				sprintf( message,"Strategy %d\nAverage (rms) distance between Controlpoints \nafter %d iteration(s): %25.15g units", e.fcnPanoNperCP, numIt,result);//average);
                numIt += 1; // 10;
                if( !infoDlg ( _setProgress,message ) )
                        *iflag = -1;

                if (e.adjustLogFile != 0) {
                        fprintf(e.adjustLogFile,"At iteration %d, iflag=0 (print), NperCP=%d, m=%d, n=%d, err = %g, x= \n",
                                              numIt,e.fcnPanoNperCP,m,n,result);
                        for (i=0; i<n; i++) {
                                fprintf(e.adjustLogFile,"\t%20.10g",x[i]);
                        }
                        fprintf(e.adjustLogFile,"\n   fvec = \n");
                        for (i=0; i<m; i++) {
                                fprintf(e.adjustLogFile,"\t%20.10g",fvec[i]);
                                if (((i+1) % e.fcnPanoNperCP) == 0) fprintf(e.adjustLogFile,"\n");
                        }
                        fprintf(e.adjustLogFile,"\n");
                        fflush(e.adjustLogFile);
                }

                return 0;
        }

        // Set Parameters

        SetAlignParams_dist( x ,&e) ;

        if (e.needInitialAvgFov) {
                e.initialAvgFov = e.avgfovFromSAP;
                e.needInitialAvgFov = 0;
                if (e.adjustLogFile != 0) {
                        fprintf(e.adjustLogFile,"setting initialAvgFov = %g\n",e.initialAvgFov);
                        fflush(e.adjustLogFile);
                }
        }

        if (e.adjustLogFile != 0) {
                fprintf(e.adjustLogFile,"entering fcnPano, m=%d, n=%d, initialAvgFov=%g, avgfovFromSAP=%g, x = \n",
                                      m,n, e.initialAvgFov,e.avgfovFromSAP);
                for (i=0; i<n; i++) {
                        fprintf(e.adjustLogFile,"\t%20.10g",x[i]);
                }
                fprintf(e.adjustLogFile,"\n");
                fflush(e.adjustLogFile);
        }

        // Calculate distances

        iresult = 0;
        for( i=0; i < e.optInfo.numPts; i++){
                if (e.fcnPanoNperCP == 1) {
                        EvaluateControlPointErrorAndComponents_dist ( i, &fvec[iresult], &junk2[0],&e);
        } else {
                        EvaluateControlPointErrorAndComponents_dist ( i, &junk, &fvec[iresult],&e);
            if (e.fcnPanoHuberSigma) {
                fvec[iresult] = huber_dist(fvec[iresult], e.fcnPanoHuberSigma);
                fvec[iresult+1] = huber_dist(fvec[iresult+1], e.fcnPanoHuberSigma);
            }
                }
                
                // Field-of-view stabilization.  Applying here means that the
                // errors seen by the optimizer may be different from those finally
                // reported, by the same factor for all errors.  This introduces
                // the possibility of confusion for people who are paying really
                // close attention to the optimizer's periodic output versus the
                // final result.  However, it seems like the right thing to do
                // because then the final reported errors will correspond to the
                // user's settings for pano size, total fov etc. 
                
                if ((e.initialAvgFov / e.avgfovFromSAP) > 1.0) {
                        fvec[iresult] *= e.initialAvgFov / e.avgfovFromSAP;
                }
                iresult += 1;
                if (e.fcnPanoNperCP == 2) {
                        if ((e.initialAvgFov / e.avgfovFromSAP) > 1.0) {
                                fvec[iresult] *= e.initialAvgFov / e.avgfovFromSAP;
                        }
                        iresult += 1;
                }               
        }
        
        // If not enough control points are provided, then fill out
        // the function vector with copies of the average error
        // for the actual control points.

        result = 0.0;
        for (i=0; i < iresult; i++) {
                result += fvec[i]*fvec[i];
        }
        result = sqrt(result/(double)iresult);
        for (i=iresult; i < m; i++) {
                fvec[i] = result;
        }

        if (e.adjustLogFile != 0) {
                result *= sqrt((double)e.fcnPanoNperCP);
                fprintf(e.adjustLogFile,"leaving fcnPano, m=%d, n=%d, err=%g, fvec = \n",m,n,result);
                for (i=0; i<m; i++) {
                        fprintf(e.adjustLogFile,"\t%20.10g",fvec[i]);
                        if (((i+1) % e.fcnPanoNperCP) == 0) fprintf(e.adjustLogFile,"\n");
                }
                fprintf(e.adjustLogFile,"\n");
                fflush(e.adjustLogFile);
        }
		optFree(&(e.optInfo));
        return 0;
}
#endif // !1