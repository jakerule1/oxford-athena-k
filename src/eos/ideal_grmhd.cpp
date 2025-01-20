//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_grmhd.cpp
//! \brief derived class that implements ideal gas EOS in general relativistic mhd

#include <float.h>

#include "athena.hpp"
#include "mhd/mhd.hpp"
#include "eos.hpp"
#include "eos/ideal_c2p_mhd.hpp"

#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealGRMHD::IdealGRMHD(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("mhd", pp, pin) {
  eos_data.is_ideal = true;
  eos_data.gamma = pin->GetReal("mhd","gamma");
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;  // ideal gas EOS always uses internal energy
  eos_data.use_t = false;
  eos_data.gamma_max = pin->GetOrAddReal("mhd","gamma_max",(FLT_MAX));  // gamma ceiling
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrim()
//! \brief Converts conserved into primitive variables.
//! Operates over range of cells given in argument list.

void IdealGRMHD::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                            DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                            const bool only_testfloors,
                            const int il, const int iu, const int jl, const int ju,
                            const int kl, const int ku) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  auto &size = pmy_pack->pmb->mb_size;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto &fofc_ = pmy_pack->pmhd->fofc;
  auto eos = eos_data;
  Real gm1 = eos_data.gamma - 1.0;

  bool &cellavg_fix_turn_on_ = pmy_pack->pmhd->cellavg_fix_turn_on;

  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  auto &use_excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &excision_floor_ = pmy_pack->pcoord->excision_floor;
  auto &excision_flux_ = pmy_pack->pcoord->excision_flux;
  auto &dexcise_ = pmy_pack->pcoord->coord_data.dexcise;
  auto &pexcise_ = pmy_pack->pcoord->coord_data.pexcise;

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nmb*nkji;

  int nfloord_=0, nfloore_=0, nceilv_=0, nfail_=0, maxit_=0;
  Kokkos::parallel_reduce("grmhd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sumd, int &sume, int &sumv, int &sumf, int &max_it) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;

    // load single state conserved variables
    MHDCons1D u;
    u.d  = cons(m,IDN,k,j,i);
    u.mx = cons(m,IM1,k,j,i);
    u.my = cons(m,IM2,k,j,i);
    u.mz = cons(m,IM3,k,j,i);
    u.e  = cons(m,IEN,k,j,i);

    // load cell-centered fields into conserved state
    // use input CC fields if only testing floors with FOFC
    if (only_testfloors) {
      u.bx = bcc(m,IBX,k,j,i);
      u.by = bcc(m,IBY,k,j,i);
      u.bz = bcc(m,IBZ,k,j,i);
    // else use simple linear average of face-centered fields
    } else {
      u.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
      u.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
      u.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));
    }

    // Extract components of metric
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

    HydPrim1D w;
    bool dfloor_used=false, efloor_used=false;
    bool vceiling_used=false, c2p_failure=false;
    int iter_used=0;

    // Only execute cons2prim if outside excised region
    bool excised = false;
    if (use_excise) {
      if (excision_floor_(m,k,j,i)) {
        w.d = dexcise_;
        w.vx = 0.0;
        w.vy = 0.0;
        w.vz = 0.0;
        w.e = pexcise_/gm1;
        excised = true;
      }
      if (only_testfloors) {
        if (excision_flux_(m,k,j,i)) {
          excised = true;
        }
      }
    }

    if (!(excised)) {
      // calculate SR conserved quantities
      MHDCons1D u_sr;
      Real s2, b2, rpar;
      TransformToSRMHD(u,glower,gupper,s2,b2,rpar,u_sr);

      // call c2p function
      // (inline function in ideal_c2p_mhd.hpp file)
      SingleC2P_IdealSRMHD(u_sr, eos, s2, b2, rpar, w,
                           dfloor_used, efloor_used, c2p_failure, iter_used);

      // apply velocity ceiling if necessary
      Real tmp = glower[1][1]*SQR(w.vx)
               + glower[2][2]*SQR(w.vy)
               + glower[3][3]*SQR(w.vz)
               + 2.0*glower[1][2]*w.vx*w.vy + 2.0*glower[1][3]*w.vx*w.vz
               + 2.0*glower[2][3]*w.vy*w.vz;
      Real lor = sqrt(1.0+tmp);
      if (lor > eos.gamma_max) {
        vceiling_used = true;
        Real factor = sqrt((SQR(eos.gamma_max)-1.0)/(SQR(lor)-1.0));
        w.vx *= factor;
        w.vy *= factor;
        w.vz *= factor;
      }
    }

    // set FOFC flag and quit loop if this function called only to check floors
    if (only_testfloors) {
      if (dfloor_used || efloor_used || vceiling_used || c2p_failure) {
        fofc_(m,k,j,i) = true;
        sumd++;  // use dfloor as counter for when either is true
      }
    } else {
      if (dfloor_used) {sumd++;}
      if (efloor_used) {sume++;}
      if (vceiling_used) {sumv++;}
      if (c2p_failure) {sumf++;}
      max_it = (iter_used > max_it) ? iter_used : max_it;

      // store primitive state in 3D array
      prim(m,IDN,k,j,i) = w.d;
      prim(m,IVX,k,j,i) = w.vx;
      prim(m,IVY,k,j,i) = w.vy;
      prim(m,IVZ,k,j,i) = w.vz;
      prim(m,IEN,k,j,i) = w.e;

      // store cell-centered fields in 3D array
      bcc(m,IBX,k,j,i) = u.bx;
      bcc(m,IBY,k,j,i) = u.by;
      bcc(m,IBZ,k,j,i) = u.bz;

      // reset conserved variables if floor, ceiling, failure, or excision encountered
      if (dfloor_used || efloor_used || vceiling_used || c2p_failure || excised) {
        MHDPrim1D w_in;
        w_in.d  = w.d;
        w_in.vx = w.vx;
        w_in.vy = w.vy;
        w_in.vz = w.vz;
        w_in.e  = w.e;
        w_in.bx = u.bx;
        w_in.by = u.by;
        w_in.bz = u.bz;

        HydCons1D u_out;
        SingleP2C_IdealGRMHD(glower, gupper, w_in, eos.gamma, u_out);
        cons(m,IDN,k,j,i) = u_out.d;
        cons(m,IM1,k,j,i) = u_out.mx;
        cons(m,IM2,k,j,i) = u_out.my;
        cons(m,IM3,k,j,i) = u_out.mz;
        cons(m,IEN,k,j,i) = u_out.e;
        u.d = u_out.d;  // (needed if there are scalars below)
      }

      // convert scalars (if any)
      for (int n=nmhd; n<(nmhd+nscal); ++n) {
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u.d;
      }
    }
  }, Kokkos::Sum<int>(nfloord_), Kokkos::Sum<int>(nfloore_), Kokkos::Sum<int>(nceilv_),
     Kokkos::Sum<int>(nfail_), Kokkos::Max<int>(maxit_));

  // store appropriate counters
  if (only_testfloors) {
    pmy_pack->pmesh->ecounter.nfofc += nfloord_;
  } else {
    pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
    pmy_pack->pmesh->ecounter.neos_efloor += nfloore_;
    pmy_pack->pmesh->ecounter.neos_vceil  += nceilv_;
    pmy_pack->pmesh->ecounter.neos_fail   += nfail_;
    pmy_pack->pmesh->ecounter.maxit_c2p = maxit_;
  }

    // fallback for the failure of variable inversion that uses the average of the valid primitives in adjacent cells
  if ((cellavg_fix_turn_on_) && !(only_testfloors)) {
    par_for("adjacent_cellavg_fix", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // Check if the cell is in excised region
      bool excised = false;
      if (use_excise) {
        if (excision_floor_(m,k,j,i)) {
          excised = true;
        }
      }

      // Set indices around the problematic cell
      int km1 = (k-1 < kl) ? kl : k-1;
      int kp1 = (k+1 > ku) ? ku : k+1;
      int jm1 = (j-1 < jl) ? jl : j-1;
      int jp1 = (j+1 > ju) ? ju : j+1;
      int im1 = (i-1 < il) ? il : i-1;
      int ip1 = (i+1 > iu) ? iu : i+1;

      int count_jake = 0;
      if (gm1*prim(m,IEN,k,j,i) <= eos.pfloor) count_jake+=1;
      if (gm1*prim(m,IEN,km1,j,i) <= eos.pfloor) count_jake+=1;
      if (gm1*prim(m,IEN,kp1,j,i) <= eos.pfloor) count_jake+=1;
      if (gm1*prim(m,IEN,k,jm1,i) <= eos.pfloor) count_jake+=1;
      if (gm1*prim(m,IEN,k,jp1,i) <= eos.pfloor) count_jake+=1;
      if (gm1*prim(m,IEN,k,j,im1) <= eos.pfloor) count_jake+=1;
      if (gm1*prim(m,IEN,k,j,ip1) <= eos.pfloor) count_jake+=1;

      // Assign fallback state if inversion fails
      if ((!c2p_flag_(m,k,j,i) || ( (count_jake>=1) && (count_jake<=2) )) && !(excised)) { //jake

        // initialize primitive fallback
        MHDPrim1D w;
        w.d = 0.0; w.vx = 0.0; w.vy = 0.0; w.vz = 0.0; w.e = 0.0;
        // Load cell-centered fields
        // if (c2p_test_) {
        //   // use input CC fields if only testing floors with FOFC
        //   w.bx = bcc(m,IBX,k,j,i);
        //   w.by = bcc(m,IBY,k,j,i);
        //   w.bz = bcc(m,IBZ,k,j,i);
        // } else {
        //   // else use simple linear average of face-centered fields
        //   w.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
        //   w.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
        //   w.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));
        // }

        // Add the primitives of valid adjacent cells
        int n_count = 0;
        for (int kk=km1; kk<=kp1; ++kk) {
          for (int jj=jm1; jj<=jp1; ++jj) {
            for (int ii=im1; ii<=ip1; ++ii) {
              if (c2p_flag_(m,kk,jj,ii) && !(excised)) {
                w.d  = w.d  + prim(m,IDN,kk,jj,ii);
                w.vx = w.vx + prim(m,IVX,kk,jj,ii);
                w.vy = w.vy + prim(m,IVY,kk,jj,ii);
                w.vz = w.vz + prim(m,IVZ,kk,jj,ii);
                w.e  = w.e  + prim(m,IEN,kk,jj,ii);
                n_count += 1;
              } // endif c2p_flag_(m,kk,jj,ii)
            } // endfor ii
          } // endfor jj
        } // endfor kk

        // Assign the fallback state
        if (n_count == 0) {
          w.d  = w0_old_(m,IDN,k,j,i);
          w.vx = w0_old_(m,IVX,k,j,i);
          w.vy = w0_old_(m,IVY,k,j,i);
          w.vz = w0_old_(m,IVZ,k,j,i);
          w.e  = w0_old_(m,IEN,k,j,i);
        } else {
          w.d  = w.d/n_count;
          w.vx = w.vx/n_count;
          w.vy = w.vy/n_count;
          w.vz = w.vz/n_count;
          w.e  = w.e/n_count;
        }

        if (!c2p_flag_(m,k,j,i)) { // if variable inversion fails
          prim(m,IDN,k,j,i) = w.d;
          prim(m,IVX,k,j,i) = w.vx;
          prim(m,IVY,k,j,i) = w.vy;
          prim(m,IVZ,k,j,i) = w.vz;
          prim(m,IEN,k,j,i) = w.e;
        }
        // } else if (smooth_flag_(m,k,j,i)) { // if extra smooth is needed
        //   prim(m,IDN,k,j,i) = w.d;
        //   prim(m,IVX,k,j,i) = w.vx;
        //   prim(m,IVY,k,j,i) = w.vy;
        //   prim(m,IVZ,k,j,i) = w.vz;
        //   prim(m,IEN,k,j,i) = w.e;
        // }

        // Extract components of metric
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

        Real glower[4][4], gupper[4][4];
        ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);
        Real alpha = sqrt(-1.0/gupper[0][0]);
        // Reset conserved variables
        HydCons1D u;
        SingleP2C_IdealGRMHD(glower, gupper, w, eos.gamma, u);
        cons(m,IDN,k,j,i) = u.d;
        cons(m,IM1,k,j,i) = u.mx;
        cons(m,IM2,k,j,i) = u.my;
        cons(m,IM3,k,j,i) = u.mz;
        cons(m,IEN,k,j,i) = u.e;

        if (entropy_fix_){
          // compute total entropy
          Real q = glower[1][1]*w.vx*w.vx + 2.0*glower[1][2]*w.vx*w.vy + 2.0*glower[1][3]*w.vx*w.vz
                + glower[2][2]*w.vy*w.vy + 2.0*glower[2][3]*w.vy*w.vz
                + glower[3][3]*w.vz*w.vz;
          Real lor = sqrt(1.0 + q);
          Real u0 = lor/alpha;

          // assign total entropy to the first scalar
          cons(m,entropyIdx,k,j,i) = gm1*w.e / pow(w.d,gm1) * u0;
          prim(m,entropyIdx,k,j,i) = cons(m,entropyIdx,k,j,i)/u.d;
        }

      } // endif (!c2p_flag_(m,k,j,i) && !(excised))
    }); // end_par_for 'adjacent_cellavg_fix'
  } // endif (cellavg_fix_turn_on_)

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCons()
//! \brief Converts primitive into conserved variables.  Operates over range of cells
//! given in argument list.

void IdealGRMHD::PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                            DvceArray5D<Real> &cons, const int il, const int iu,
                            const int jl, const int ju, const int kl, const int ku) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real &gamma = eos_data.gamma;

  par_for("grmhd_p2c", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Extract components of metric
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

    // Load single state of primitive variables
    MHDPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);
    w.e  = prim(m,IEN,k,j,i);

    // load cell-centered fields into primitive state
    w.bx = bcc(m,IBX,k,j,i);
    w.by = bcc(m,IBY,k,j,i);
    w.bz = bcc(m,IBZ,k,j,i);

    // call p2c function
    HydCons1D u;
    SingleP2C_IdealGRMHD(glower, gupper, w, gamma, u);

    // store conserved quantities in 3D array
    cons(m,IDN,k,j,i) = u.d;
    cons(m,IM1,k,j,i) = u.mx;
    cons(m,IM2,k,j,i) = u.my;
    cons(m,IM3,k,j,i) = u.mz;
    cons(m,IEN,k,j,i) = u.e;

    // convert scalars (if any)
    for (int n=nmhd; n<(nmhd+nscal); ++n) {
      cons(m,n,k,j,i) = u.d*prim(m,n,k,j,i);
    }
  });

  return;
}
