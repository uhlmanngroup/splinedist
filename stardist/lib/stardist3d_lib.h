/*
BSD 3-Clause License

Copyright (c) 2018-2020, Uwe Schmidt, Martin Weigert
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef STARDIST3D_LIB_H
#define STARDIST3D_LIB_H


void _COMMON_non_maximum_suppression_sparse(
                    const float* scores, const float* dist, const float* points,
                    const int n_polys, const int n_params, const int n_faces, 
                    const float* verts, const int* faces,
                    const float threshold, const int use_bbox, const int verbose, 
                    bool* result);


void _COMMON_polyhedron_to_label(const float* dist, const float* points,
                                 const float* verts,const int* faces,
                                 const int n_polys, const int n_params,
                                 const int n_faces,
                                 const int* labels,
                                 const int nz, const int  ny,const int nx,
                                 const int render_mode,
                                 const int verbose,
                                 const int use_overlap_label,
                                 const int overlap_label,  
                                 int * result);

#ifdef __cplusplus
extern "C" {
#endif
 
  void _LIB_non_maximum_suppression_sparse(
                    const float*, const float*, const float*,                    
                    const int, const int, const int,
                    const float* , const int* ,
                    const float, const int, const int, 
                    bool*);


  void  _LIB_polyhedron_to_label(const float* dist, const float* points,
                                 const float* verts,const int* faces,
                                 const int n_polys, const int n_params,
                                 const int n_faces,
                                 const int* labels,
                                 const int nz, const int  ny,const int nx,
                                 const int render_mode,
                                 const int verbose,
                                 const int use_overlap_label,
                                 const int overlap_label,  
                                 int * result);
  
#ifdef __cplusplus
}
#endif

#endif

