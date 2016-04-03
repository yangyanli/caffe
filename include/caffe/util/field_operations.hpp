#ifndef CAFFE_UTIL_FIELD_OPERATIONS_H_
#define CAFFE_UTIL_FIELD_OPERATIONS_H_

namespace caffe {
  
  template<typename Dtype>
  void SnapGrid_cpu(Dtype& value, int& value_0, int& value_1, const int max) {
    if (value >= 0 && value < max) {
      value_0 = std::floor(value);
    } else if (value < 0) {
      value = 0;
      value_0 = 0;
    } else /*(value >= max)*/ {
      value = max;
      value_0 = max-1;
    }
    value_1 = value_0 + 1;
  }
  
  template<typename Dtype>
  Dtype Interpolate_cpu(const Dtype* df, const int batch_idx,
    const int x0, const int y0, const int z0,
    const int x1, const int y1, const int z1,
    const Dtype x_x0, const Dtype y_y0, const Dtype z_z0,
    const Dtype x1_x, const Dtype y1_y, const Dtype z1_z,
    const int field_dim_x, const int field_dim_y, const int field_dim_z) {
    int b_offset_000 = ((batch_idx * field_dim_x + x0) * field_dim_y + y0) * field_dim_z + z0;
    int b_offset_001 = ((batch_idx * field_dim_x + x0) * field_dim_y + y0) * field_dim_z + z1;
    int b_offset_010 = ((batch_idx * field_dim_x + x0) * field_dim_y + y1) * field_dim_z + z0;
    int b_offset_011 = ((batch_idx * field_dim_x + x0) * field_dim_y + y1) * field_dim_z + z1;
    int b_offset_100 = ((batch_idx * field_dim_x + x1) * field_dim_y + y0) * field_dim_z + z0;
    int b_offset_101 = ((batch_idx * field_dim_x + x1) * field_dim_y + y0) * field_dim_z + z1;
    int b_offset_110 = ((batch_idx * field_dim_x + x1) * field_dim_y + y1) * field_dim_z + z0;
    int b_offset_111 = ((batch_idx * field_dim_x + x1) * field_dim_y + y1) * field_dim_z + z1;
                
    Dtype v000 = df[b_offset_000];
    Dtype v001 = df[b_offset_001];
    Dtype v010 = df[b_offset_010];
    Dtype v011 = df[b_offset_011];
    Dtype v100 = df[b_offset_100];
    Dtype v101 = df[b_offset_101];
    Dtype v110 = df[b_offset_110];
    Dtype v111 = df[b_offset_111];
  
    Dtype c00 = v000*x1_x+v100*x_x0;
    Dtype c10 = v010*x1_x+v110*x_x0;
    Dtype c01 = v001*x1_x+v101*x_x0;
    Dtype c11 = v011*x1_x+v111*x_x0;
  
    Dtype c0 = c00*y1_y+c10*y_y0;
    Dtype c1 = c01*y1_y+c11*y_y0;
  
    return c0*z1_z+c1*z_z0;
  }
  
  template<typename Dtype>
  void SnapGrid_cpu(Dtype& value, int& value_0, int& value_1, Dtype& value_a, Dtype& value_m, const int max) {
    SnapGrid_cpu(value, value_0, value_1, max);
  
    const Dtype h = 0.01;
    Dtype lower_bound = value_0+h;
    Dtype upper_bound = value_1-h;
    if (value >= lower_bound && value < upper_bound) {
      value_m = value - h;
      value_a = value + h;
    } else if (value < lower_bound) {
      value_m = value_0;
      value_a = value + h;
    } else /* (value >= upper_bound) */ {
      value_m = value - h;
      value_a = value_1;
    }
  }
  
  template<typename Dtype>
  Dtype ComputeGradient_cpu(const Dtype* df, const int batch_idx,
    const int x0, const int y0, const int z0,
    const int x1, const int y1, const int z1,
    const Dtype x_a, const Dtype y_a, const Dtype z_a,
    const Dtype x_m, const Dtype y_m, const Dtype z_m,
    const Dtype x_x0, const Dtype y_y0, const Dtype z_z0,
    const Dtype x1_x, const Dtype y1_y, const Dtype z1_z,
    Dtype& dx, Dtype& dy, Dtype& dz,
    const int field_dim_x, const int field_dim_y, const int field_dim_z) {
    int b_offset_000 = ((batch_idx * field_dim_x + x0) * field_dim_y + y0) * field_dim_z + z0;
    int b_offset_001 = ((batch_idx * field_dim_x + x0) * field_dim_y + y0) * field_dim_z + z1;
    int b_offset_010 = ((batch_idx * field_dim_x + x0) * field_dim_y + y1) * field_dim_z + z0;
    int b_offset_011 = ((batch_idx * field_dim_x + x0) * field_dim_y + y1) * field_dim_z + z1;
    int b_offset_100 = ((batch_idx * field_dim_x + x1) * field_dim_y + y0) * field_dim_z + z0;
    int b_offset_101 = ((batch_idx * field_dim_x + x1) * field_dim_y + y0) * field_dim_z + z1;
    int b_offset_110 = ((batch_idx * field_dim_x + x1) * field_dim_y + y1) * field_dim_z + z0;
    int b_offset_111 = ((batch_idx * field_dim_x + x1) * field_dim_y + y1) * field_dim_z + z1;
                
    Dtype v000 = df[b_offset_000];
    Dtype v001 = df[b_offset_001];
    Dtype v010 = df[b_offset_010];
    Dtype v011 = df[b_offset_011];
    Dtype v100 = df[b_offset_100];
    Dtype v101 = df[b_offset_101];
    Dtype v110 = df[b_offset_110];
    Dtype v111 = df[b_offset_111];
  
    Dtype x_am = x_a-x_m;
    Dtype x_ma = x_m-x_a;
    Dtype y_am = y_a-y_m;
    Dtype y_ma = y_m-y_a;
    Dtype z_am = z_a-z_m;
    Dtype z_ma = z_m-z_a;
    
    dx =
    v000*x_ma*y1_y*z1_z +
    v100*x_am*y1_y*z1_z +
    v010*x_ma*y_y0*z1_z + 
    v001*x_ma*y1_y*z_z0 +
    v101*x_am*y1_y*z_z0 +
    v011*x_ma*y_y0*z_z0 +
    v110*x_am*y_y0*z1_z +
    v111*x_am*y_y0*z_z0;
    dx /= x_am;
    
    dy = 
    v000*x1_x*y_ma*z1_z +
    v100*x_x0*y_ma*z1_z +
    v010*x1_x*y_am*z1_z + 
    v001*x1_x*y_ma*z_z0 +
    v101*x_x0*y_ma*z_z0 +
    v011*x1_x*y_am*z_z0 +
    v110*x_x0*y_am*z1_z +
    v111*x_x0*y_am*z_z0;
    dy /= y_am;
    
    dz = 
    v000*x1_x*y1_y*z_ma +
    v100*x_x0*y1_y*z_ma +
    v010*x1_x*y_y0*z_ma + 
    v001*x1_x*y1_y*z_am +
    v101*x_x0*y1_y*z_am +
    v011*x1_x*y_y0*z_am +
    v110*x_x0*y_y0*z_ma +
    v111*x_x0*y_y0*z_am;
    dz /= z_am;
  
    return (
    v000*x1_x*y1_y*z1_z +
    v100*x_x0*y1_y*z1_z +
    v010*x1_x*y_y0*z1_z +
    v001*x1_x*y1_y*z_z0 +
    v101*x_x0*y1_y*z_z0 +
    v011*x1_x*y_y0*z_z0 +
    v110*x_x0*y_y0*z1_z +
    v111*x_x0*y_y0*z_z0);
  }
  
  template<typename Dtype>
  void ComputeGradient_cpu(const Dtype* df, const int batch_idx,
    Dtype& x, Dtype& y, Dtype& z,
    Dtype& dx, Dtype& dy, Dtype& dz,
    const int field_dim_x, const int field_dim_y, const int field_dim_z) {
    int x0, y0, z0, x1, y1, z1;
    Dtype x_a, y_a, z_a, x_m, y_m, z_m;
    SnapGrid_cpu(x, x0, x1, x_a, x_m, field_dim_x-1);
    SnapGrid_cpu(y, y0, y1, y_a, y_m, field_dim_y-1);
    SnapGrid_cpu(z, z0, z1, z_a, z_m, field_dim_z-1);
    Dtype x_x0 = x-x0;
    Dtype y_y0 = y-y0;
    Dtype z_z0 = z-z0;
    Dtype x1_x = x1-x;
    Dtype y1_y = y1-y;
    Dtype z1_z = z1-z;
  
    ComputeGradient_cpu(df, batch_idx, x0, y0, z0, x1, y1, z1, x_a, y_a, z_a, x_m, y_m, z_m,
                    x_x0, y_y0, z_z0, x1_x, y1_y, z1_z, dx, dy, dz, field_dim_x, field_dim_y, field_dim_z);
  }
  
  template<typename Dtype>
  void Jitter_cpu(const Dtype& value, Dtype& value_a, Dtype& value_m, const int max) {
    const Dtype h = 0.01;
    if (value >= h && value < max-h) {
      value_m = value - h;
      value_a = value + h;
    } else if (value < h) {
      value_m = 0;
      value_a = value + h;
    } else /* (value >= max-h) */ {
      value_m = value - h;
      value_a = max;
    }
  }
  
  template<typename Dtype>
  void Normalize_cpu(Dtype& nx, Dtype& ny, Dtype& nz) {
    Dtype len = std::sqrt(nx*nx+ny*ny+nz*nz);
    if (len != 0) {
      nx /= len;
      ny /= len;
      nz /= len;
    }
  }
  
  template<typename Dtype>
  void ComputeNormalGradient_cpu(const Dtype* df, const int batch_idx,
    Dtype& x, Dtype& y, Dtype& z,
    Dtype& nx_dx, Dtype& nx_dy, Dtype& nx_dz,
    Dtype& ny_dx, Dtype& ny_dy, Dtype& ny_dz,
    Dtype& nz_dx, Dtype& nz_dy, Dtype& nz_dz,
    const int field_dim_x, const int field_dim_y, const int field_dim_z) {
    Dtype x_a, y_a, z_a, x_m, y_m, z_m;
    Jitter_cpu(x, x_a, x_m, field_dim_x-1);
    Jitter_cpu(y, y_a, y_m, field_dim_y-1);
    Jitter_cpu(z, z_a, z_m, field_dim_z-1);
  
    Dtype nx_x_a, ny_x_a, nz_x_a;
    ComputeGradient_cpu(df, batch_idx, x_a, y, z, nx_x_a, ny_x_a, nz_x_a, field_dim_x, field_dim_y, field_dim_z);
    Normalize_cpu(nx_x_a, ny_x_a, nz_x_a);
  
    Dtype nx_x_m, ny_x_m, nz_x_m;
    ComputeGradient_cpu(df, batch_idx, x_m, y, z, nx_x_m, ny_x_m, nz_x_m, field_dim_x, field_dim_y, field_dim_z);
    Normalize_cpu(nx_x_m, ny_x_m, nz_x_m);
  
    Dtype nx_y_a, ny_y_a, nz_y_a;
    ComputeGradient_cpu(df, batch_idx, x, y_a, z, nx_y_a, ny_y_a, nz_y_a, field_dim_x, field_dim_y, field_dim_z);
    Normalize_cpu(nx_y_a, ny_y_a, nz_y_a);
  
    Dtype nx_y_m, ny_y_m, nz_y_m;
    ComputeGradient_cpu(df, batch_idx, x, y_m, z, nx_y_m, ny_y_m, nz_y_m, field_dim_x, field_dim_y, field_dim_z);
    Normalize_cpu(nx_y_m, ny_y_m, nz_y_m);
  
    Dtype nx_z_a, ny_z_a, nz_z_a;
    ComputeGradient_cpu(df, batch_idx, x, y, z_a, nx_z_a, ny_z_a, nz_z_a, field_dim_x, field_dim_y, field_dim_z);
    Normalize_cpu(nx_z_a, ny_z_a, nz_z_a);
  
    Dtype nx_z_m, ny_z_m, nz_z_m;
    ComputeGradient_cpu(df, batch_idx, x, y, z_m, nx_z_m, ny_z_m, nz_z_m, field_dim_x, field_dim_y, field_dim_z);
    Normalize_cpu(nx_z_m, ny_z_m, nz_z_m);
  
    Dtype x_scaler = 1.0/(x_a-x_m);
    Dtype y_scaler = 1.0/(y_a-y_m);
    Dtype z_scaler = 1.0/(z_a-z_m);
  
    nx_dx = (nx_x_a-nx_x_m)*x_scaler;
    nx_dy = (nx_y_a-nx_y_m)*y_scaler;
    nx_dz = (nx_z_a-nx_z_m)*z_scaler;
  
    ny_dx = (ny_x_a-ny_x_m)*x_scaler;
    ny_dy = (ny_y_a-ny_y_m)*y_scaler;
    ny_dz = (ny_z_a-ny_z_m)*z_scaler;
  
    nz_dx = (nz_x_a-nz_x_m)*x_scaler;
    nz_dy = (nz_y_a-nz_y_m)*y_scaler;
    nz_dz = (nz_z_a-nz_z_m)*z_scaler;
  }
  
#ifndef CPU_ONLY
  
  template<typename Dtype>
  __device__ void SnapGrid_gpu(Dtype& value, int& value_0, int& value_1, const int max) {
    if (value >= 0 && value < max) {
      value_0 = floor(value);
    } else if (value < 0) {
      value = 0;
      value_0 = 0;
    } else /*(value >= max)*/ {
      value = max;
      value_0 = max-1;
    }
    value_1 = value_0 + 1;
  }
  
  template<typename Dtype>
  __device__ Dtype Interpolate_gpu(const Dtype* df, const int batch_idx,
    const int x0, const int y0, const int z0,
    const int x1, const int y1, const int z1,
    const Dtype x_x0, const Dtype y_y0, const Dtype z_z0,
    const Dtype x1_x, const Dtype y1_y, const Dtype z1_z,
    const int field_dim_x, const int field_dim_y, const int field_dim_z) {
    int b_offset_000 = ((batch_idx * field_dim_x + x0) * field_dim_y + y0) * field_dim_z + z0;
    int b_offset_001 = ((batch_idx * field_dim_x + x0) * field_dim_y + y0) * field_dim_z + z1;
    int b_offset_010 = ((batch_idx * field_dim_x + x0) * field_dim_y + y1) * field_dim_z + z0;
    int b_offset_011 = ((batch_idx * field_dim_x + x0) * field_dim_y + y1) * field_dim_z + z1;
    int b_offset_100 = ((batch_idx * field_dim_x + x1) * field_dim_y + y0) * field_dim_z + z0;
    int b_offset_101 = ((batch_idx * field_dim_x + x1) * field_dim_y + y0) * field_dim_z + z1;
    int b_offset_110 = ((batch_idx * field_dim_x + x1) * field_dim_y + y1) * field_dim_z + z0;
    int b_offset_111 = ((batch_idx * field_dim_x + x1) * field_dim_y + y1) * field_dim_z + z1;
  
    Dtype v000 = df[b_offset_000];
    Dtype v001 = df[b_offset_001];
    Dtype v010 = df[b_offset_010];
    Dtype v011 = df[b_offset_011];
    Dtype v100 = df[b_offset_100];
    Dtype v101 = df[b_offset_101];
    Dtype v110 = df[b_offset_110];
    Dtype v111 = df[b_offset_111];
  
    Dtype c00 = v000*x1_x+v100*x_x0;
    Dtype c10 = v010*x1_x+v110*x_x0;
    Dtype c01 = v001*x1_x+v101*x_x0;
    Dtype c11 = v011*x1_x+v111*x_x0;
  
    Dtype c0 = c00*y1_y+c10*y_y0;
    Dtype c1 = c01*y1_y+c11*y_y0;
  
    return c0*z1_z+c1*z_z0;
  }
  
  template<typename Dtype>
  __device__ void SnapGrid_gpu(Dtype& value, int& value_0, int& value_1, Dtype& value_a, Dtype& value_m, const int max) {
    SnapGrid_gpu(value, value_0, value_1, max);
  
    const Dtype h = 0.01;
    Dtype lower_bound = value_0+h;
    Dtype upper_bound = value_1-h;
    if (value >= lower_bound && value < upper_bound) {
      value_m = value - h;
      value_a = value + h;
    } else if (value < lower_bound) {
      value_m = value_0;
      value_a = value + h;
    } else /* (value >= upper_bound) */ {
      value_m = value - h;
      value_a = value_1;
    }
  }
  
  template<typename Dtype>
  __device__ Dtype ComputeGradient_gpu(const Dtype* df, const int batch_idx,
    const int x0, const int y0, const int z0,
    const int x1, const int y1, const int z1,
    const Dtype x_a, const Dtype y_a, const Dtype z_a,
    const Dtype x_m, const Dtype y_m, const Dtype z_m,
    const Dtype x_x0, const Dtype y_y0, const Dtype z_z0,
    const Dtype x1_x, const Dtype y1_y, const Dtype z1_z,
    Dtype& dx, Dtype& dy, Dtype& dz,
    const int field_dim_x, const int field_dim_y, const int field_dim_z) {
    int b_offset_000 = ((batch_idx * field_dim_x + x0) * field_dim_y + y0) * field_dim_z + z0;
    int b_offset_001 = ((batch_idx * field_dim_x + x0) * field_dim_y + y0) * field_dim_z + z1;
    int b_offset_010 = ((batch_idx * field_dim_x + x0) * field_dim_y + y1) * field_dim_z + z0;
    int b_offset_011 = ((batch_idx * field_dim_x + x0) * field_dim_y + y1) * field_dim_z + z1;
    int b_offset_100 = ((batch_idx * field_dim_x + x1) * field_dim_y + y0) * field_dim_z + z0;
    int b_offset_101 = ((batch_idx * field_dim_x + x1) * field_dim_y + y0) * field_dim_z + z1;
    int b_offset_110 = ((batch_idx * field_dim_x + x1) * field_dim_y + y1) * field_dim_z + z0;
    int b_offset_111 = ((batch_idx * field_dim_x + x1) * field_dim_y + y1) * field_dim_z + z1;
                
    Dtype v000 = df[b_offset_000];
    Dtype v001 = df[b_offset_001];
    Dtype v010 = df[b_offset_010];
    Dtype v011 = df[b_offset_011];
    Dtype v100 = df[b_offset_100];
    Dtype v101 = df[b_offset_101];
    Dtype v110 = df[b_offset_110];
    Dtype v111 = df[b_offset_111];
  
    Dtype x_am = x_a-x_m;
    Dtype x_ma = x_m-x_a;
    Dtype y_am = y_a-y_m;
    Dtype y_ma = y_m-y_a;
    Dtype z_am = z_a-z_m;
    Dtype z_ma = z_m-z_a;
  
    dx =
    v000*x_ma*y1_y*z1_z +
    v100*x_am*y1_y*z1_z +
    v010*x_ma*y_y0*z1_z + 
    v001*x_ma*y1_y*z_z0 +
    v101*x_am*y1_y*z_z0 +
    v011*x_ma*y_y0*z_z0 +
    v110*x_am*y_y0*z1_z +
    v111*x_am*y_y0*z_z0;
    dx /= x_am;
    
    dy = 
    v000*x1_x*y_ma*z1_z +
    v100*x_x0*y_ma*z1_z +
    v010*x1_x*y_am*z1_z + 
    v001*x1_x*y_ma*z_z0 +
    v101*x_x0*y_ma*z_z0 +
    v011*x1_x*y_am*z_z0 +
    v110*x_x0*y_am*z1_z +
    v111*x_x0*y_am*z_z0;
    dy /= y_am;
    
    dz = 
    v000*x1_x*y1_y*z_ma +
    v100*x_x0*y1_y*z_ma +
    v010*x1_x*y_y0*z_ma + 
    v001*x1_x*y1_y*z_am +
    v101*x_x0*y1_y*z_am +
    v011*x1_x*y_y0*z_am +
    v110*x_x0*y_y0*z_ma +
    v111*x_x0*y_y0*z_am;
    dz /= z_am;
  
    return (
    v000*x1_x*y1_y*z1_z +
    v100*x_x0*y1_y*z1_z +
    v010*x1_x*y_y0*z1_z +
    v001*x1_x*y1_y*z_z0 +
    v101*x_x0*y1_y*z_z0 +
    v011*x1_x*y_y0*z_z0 +
    v110*x_x0*y_y0*z1_z +
    v111*x_x0*y_y0*z_z0);
  }
  
  template<typename Dtype>
  __device__ Dtype ComputeGradient_gpu(const Dtype* df, const int batch_idx,
    Dtype& x, Dtype& y, Dtype& z,
    Dtype& dx, Dtype& dy, Dtype& dz,
    const int field_dim_x, const int field_dim_y, const int field_dim_z) {
    int x0, y0, z0, x1, y1, z1;
    Dtype x_a, y_a, z_a, x_m, y_m, z_m;
    SnapGrid_gpu(x, x0, x1, x_a, x_m, field_dim_x-1);
    SnapGrid_gpu(y, y0, y1, y_a, y_m, field_dim_y-1);
    SnapGrid_gpu(z, z0, z1, z_a, z_m, field_dim_z-1);
    Dtype x_x0 = x-x0;
    Dtype y_y0 = y-y0;
    Dtype z_z0 = z-z0;
    Dtype x1_x = x1-x;
    Dtype y1_y = y1-y;
    Dtype z1_z = z1-z;
  
    return ComputeGradient_gpu(df, batch_idx, x0, y0, z0, x1, y1, z1, x_a, y_a, z_a, x_m, y_m, z_m,
                    x_x0, y_y0, z_z0, x1_x, y1_y, z1_z, dx, dy, dz, field_dim_x, field_dim_y, field_dim_z);
  }
  
  template<typename Dtype>
  __device__ void Jitter_gpu(const Dtype& value, Dtype& value_a, Dtype& value_m, const int max) {
    const Dtype h = 0.01;
    if (value >= h && value < max-h) {
      value_m = value - h;
      value_a = value + h;
    } else if (value < h) {
      value_m = 0;
      value_a = value + h;
    } else /* (value >= max-h) */ {
      value_m = value - h;
      value_a = max;
    }
  }
  
  template<typename Dtype>
  __device__ void Normalize_gpu(Dtype& nx, Dtype& ny, Dtype& nz) {
    Dtype len = sqrt(nx*nx+ny*ny+nz*nz);
    if (len != 0) {
      nx /= len;
      ny /= len;
      nz /= len;
    }
  }
  
  template<typename Dtype>
  __device__ void ComputeNormalGradient_gpu(const Dtype* df, const int batch_idx,
    Dtype& x, Dtype& y, Dtype& z,
    Dtype& nx_dx, Dtype& nx_dy, Dtype& nx_dz,
    Dtype& ny_dx, Dtype& ny_dy, Dtype& ny_dz,
    Dtype& nz_dx, Dtype& nz_dy, Dtype& nz_dz,
    const int field_dim_x, const int field_dim_y, const int field_dim_z) {
    Dtype x_a, y_a, z_a, x_m, y_m, z_m;
    Jitter_gpu(x, x_a, x_m, field_dim_x-1);
    Jitter_gpu(y, y_a, y_m, field_dim_y-1);
    Jitter_gpu(z, z_a, z_m, field_dim_z-1);
  
    Dtype nx_x_a, ny_x_a, nz_x_a;
    ComputeGradient_gpu(df, batch_idx, x_a, y, z, nx_x_a, ny_x_a, nz_x_a, field_dim_x, field_dim_y, field_dim_z);
    Normalize_gpu(nx_x_a, ny_x_a, nz_x_a);
  
    Dtype nx_x_m, ny_x_m, nz_x_m;
    ComputeGradient_gpu(df, batch_idx, x_m, y, z, nx_x_m, ny_x_m, nz_x_m, field_dim_x, field_dim_y, field_dim_z);
    Normalize_gpu(nx_x_m, ny_x_m, nz_x_m);
  
    Dtype nx_y_a, ny_y_a, nz_y_a;
    ComputeGradient_gpu(df, batch_idx, x, y_a, z, nx_y_a, ny_y_a, nz_y_a, field_dim_x, field_dim_y, field_dim_z);
    Normalize_gpu(nx_y_a, ny_y_a, nz_y_a);
  
    Dtype nx_y_m, ny_y_m, nz_y_m;
    ComputeGradient_gpu(df, batch_idx, x, y_m, z, nx_y_m, ny_y_m, nz_y_m, field_dim_x, field_dim_y, field_dim_z);
    Normalize_gpu(nx_y_m, ny_y_m, nz_y_m);
  
    Dtype nx_z_a, ny_z_a, nz_z_a;
    ComputeGradient_gpu(df, batch_idx, x, y, z_a, nx_z_a, ny_z_a, nz_z_a, field_dim_x, field_dim_y, field_dim_z);
    Normalize_gpu(nx_z_a, ny_z_a, nz_z_a);
  
    Dtype nx_z_m, ny_z_m, nz_z_m;
    ComputeGradient_gpu(df, batch_idx, x, y, z_m, nx_z_m, ny_z_m, nz_z_m, field_dim_x, field_dim_y, field_dim_z);
    Normalize_gpu(nx_z_m, ny_z_m, nz_z_m);
  
    Dtype x_scaler = 1.0/(x_a-x_m);
    Dtype y_scaler = 1.0/(y_a-y_m);
    Dtype z_scaler = 1.0/(z_a-z_m);
  
    nx_dx = (nx_x_a-nx_x_m)*x_scaler;
    nx_dy = (nx_y_a-nx_y_m)*y_scaler;
    nx_dz = (nx_z_a-nx_z_m)*z_scaler;
  
    ny_dx = (ny_x_a-ny_x_m)*x_scaler;
    ny_dy = (ny_y_a-ny_y_m)*y_scaler;
    ny_dz = (ny_z_a-ny_z_m)*z_scaler;
  
    nz_dx = (nz_x_a-nz_x_m)*x_scaler;
    nz_dy = (nz_y_a-nz_y_m)*y_scaler;
    nz_dz = (nz_z_a-nz_z_m)*z_scaler;
  }

#endif // !CPU_ONLY
 
}  // namespace caffe

#endif  // CAFFE_UTIL_FIELD_OPERATIONS_H_
