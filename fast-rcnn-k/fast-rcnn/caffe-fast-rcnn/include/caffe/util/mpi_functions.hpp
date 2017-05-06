#ifndef CAFFE_UTIL_MPI_FUNCTIONS_H_
#define CAFFE_UTIL_MPI_FUNCTIONS_H_

#include "caffe/util/mpi.hpp"
#include "caffe/common.hpp"

namespace caffe {

template <typename Dtype>
int caffe_Recv(Dtype* recvbuf, int icount, int isource);

template <typename Dtype>
int caffe_Send(Dtype* sendbuf, int icount, int idest);

template <typename Dtype>
int caffe_Bcast(Dtype* sendbuf, int icount, int iroot);

template <typename Dtype>
int caffe_Allreduce(Dtype* sendbuf, Dtype* recvbuf, int icount, MPI_Op iop);

template <typename Dtype>
int caffe_Ireduce(const Dtype* sendbuf, Dtype* recvbuf, int icount, MPI_Op iop, int root, MPI_Request* req);

template <typename Dtype>
int caffe_Reduce(const Dtype* sendbuf, Dtype* recvbuf, int icount, MPI_Op iop, int root);

template <typename Dtype>
int caffe_Ibcast(Dtype* sendbuf, int icount, int iroot, MPI_Request* req);

template <typename Dtype>
int caffe_Iallreduce(Dtype* sendbuf, Dtype* recvbuf, int icount, MPI_Op iop, MPI_Request* req);

template <typename Dtype>
int caffe_Allgather(Dtype* sendbuf, Dtype* recvbuf, int sendcount);

template <typename Dtype>
int caffe_Alltoall(Dtype* sendbuf, Dtype* recvbuf, int sendcount);

template <typename Dtype>
int caffe_Ialltoall(Dtype* sendbuf, int sendcount, MPI_Request* req);

int caffe_Wait(MPI_Request* req);

}


#endif
