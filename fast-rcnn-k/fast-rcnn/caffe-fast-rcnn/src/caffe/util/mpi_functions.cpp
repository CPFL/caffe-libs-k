#include "caffe/util/mpi_functions.hpp"

namespace caffe {

template<>
int caffe_Recv<float>(float* recvbuf, int icount, int isource) {
  return MPI_Recv(recvbuf, icount, MPI_FLOAT,
		  isource, 1000, MPI_COMM_WORLD, NULL);
}

template<>
int caffe_Recv<double>(double* recvbuf, int icount, int isource) {
  return MPI_Recv(recvbuf, icount, MPI_DOUBLE,
		  isource, 1000, MPI_COMM_WORLD, NULL);
}

template<>
int caffe_Send<float>(float* sendbuf, int icount, int idest) {
  return MPI_Send(sendbuf, icount, MPI_FLOAT,
		  idest, 1000, MPI_COMM_WORLD);
}

template<>
int caffe_Send<double>(double* sendbuf, int icount, int idest) {
  return MPI_Send(sendbuf, icount, MPI_DOUBLE,
		  idest, 1000, MPI_COMM_WORLD);
}

template<>
int caffe_Bcast<float>(float* sendbuf, int icount, int iroot) {
  return MPI_Bcast(sendbuf, icount, MPI_FLOAT, iroot, MPI_COMM_WORLD);
}

template<>
int caffe_Bcast<double>(double* sendbuf, int icount, int iroot) {
  return MPI_Bcast(sendbuf, icount, MPI_DOUBLE, iroot, MPI_COMM_WORLD);
}

template<>
int caffe_Bcast<int>(int* sendbuf, int icount, int iroot) {
  return MPI_Bcast(sendbuf, icount, MPI_DOUBLE, iroot, MPI_COMM_WORLD);
}
  /*
template<>
int caffe_Ibcast<float>(float* sendbuf, int icount, int iroot, MPI_Request* req) {
  return MPI_Ibcast(sendbuf, icount, MPI_FLOAT, iroot, MPI_COMM_WORLD, req);
}

template<>
int caffe_Ibcast<double>(double* sendbuf, int icount, int iroot, MPI_Request* req) {
  return MPI_Ibcast(sendbuf, icount, MPI_DOUBLE, iroot, MPI_COMM_WORLD, req);
}

template<>
int caffe_Reduce<float>(const float* sendbuf, float* recvbuf, int icount, MPI_Op iop, int root) {
  return MPI_Reduce(sendbuf, recvbuf, icount, MPI_FLOAT, iop, root, MPI_COMM_WORLD);
}

template<>
int caffe_Reduce<double>(const double* sendbuf, double* recvbuf, int icount, MPI_Op iop, int root) {
  return MPI_Reduce(sendbuf, recvbuf, icount, MPI_DOUBLE, iop, root, MPI_COMM_WORLD);
}

template<>
int caffe_Ireduce<float>(const float* sendbuf, float* recvbuf, int icount, MPI_Op iop, int root, MPI_Request* req) {
  return MPI_Ireduce(sendbuf, recvbuf, icount, MPI_FLOAT, iop, root, MPI_COMM_WORLD, req);
}

template<>
int caffe_Ireduce<double>(const double* sendbuf, double* recvbuf, int icount, MPI_Op iop, int root, MPI_Request* req) {
  return MPI_Ireduce(sendbuf, recvbuf, icount, MPI_DOUBLE, iop, root, MPI_COMM_WORLD, req);
}
  */
template<>
int caffe_Allreduce<float>(float* sendbuf, float* recvbuf, int icount, MPI_Op iop) {
  if (sendbuf == recvbuf) {
    return MPI_Allreduce(MPI_IN_PLACE, recvbuf, icount, MPI_FLOAT, iop, MPI_COMM_WORLD);
  } else {
    return MPI_Allreduce(sendbuf, recvbuf, icount, MPI_FLOAT, iop, MPI_COMM_WORLD);
  }
}

template<>
int caffe_Allreduce<double>(double* sendbuf, double* recvbuf, int icount, MPI_Op iop) {
  if (sendbuf == recvbuf) {
    return MPI_Allreduce(MPI_IN_PLACE, recvbuf, icount, MPI_DOUBLE, iop, MPI_COMM_WORLD);
  } else {
    return MPI_Allreduce(sendbuf, recvbuf, icount, MPI_DOUBLE, iop, MPI_COMM_WORLD);
  }
}
  /*
template<>
int caffe_Iallreduce<float>(float* sendbuf, float* recvbuf, int icount, MPI_Op iop, MPI_Request* req) {
  if (sendbuf == recvbuf) {
    return MPI_Iallreduce(MPI_IN_PLACE, recvbuf, icount, MPI_FLOAT, iop, MPI_COMM_WORLD, req);
  } else {
    return MPI_Iallreduce(sendbuf, recvbuf, icount, MPI_FLOAT, iop, MPI_COMM_WORLD, req);
  }
}

template<>
int caffe_Iallreduce<double>(double* sendbuf, double* recvbuf, int icount, MPI_Op iop, MPI_Request* req) {
  if (sendbuf == recvbuf) {
    return MPI_Iallreduce(MPI_IN_PLACE, recvbuf, icount, MPI_DOUBLE, iop, MPI_COMM_WORLD, req);
  } else {
    return MPI_Iallreduce(sendbuf, recvbuf, icount, MPI_DOUBLE, iop, MPI_COMM_WORLD, req);
  }
}

template<>
int caffe_Iallreduce<int>(int* sendbuf, int* recvbuf, int icount, MPI_Op iop, MPI_Request* req) {
  if (sendbuf == recvbuf) {
    return MPI_Iallreduce(MPI_IN_PLACE, recvbuf, icount, MPI_INT, iop, MPI_COMM_WORLD, req);
  } else {
    return MPI_Iallreduce(sendbuf, recvbuf, icount, MPI_INT, iop, MPI_COMM_WORLD, req);
  }
}

template<>
int caffe_Iallreduce<unsigned int>(unsigned int* sendbuf, unsigned int* recvbuf, int icount, MPI_Op iop, MPI_Request* req) {
  return MPI_Iallreduce(sendbuf, recvbuf, icount, MPI_INT, iop, MPI_COMM_WORLD, req);
}
  */
template <>
int caffe_Allgather<float>(float* sendbuf, float* recvbuf, int sendcount) {
  return MPI_Allgather(sendbuf, sendcount, MPI_FLOAT, recvbuf, sendcount, MPI_FLOAT, MPI_COMM_WORLD);
}

template <>
int caffe_Allgather<double>(double* sendbuf, double* recvbuf, int sendcount) {
  return MPI_Allgather(sendbuf, sendcount, MPI_DOUBLE, recvbuf, sendcount, MPI_DOUBLE, MPI_COMM_WORLD);
}
template <>
int caffe_Alltoall<float>(float* sendbuf, float* recvbuf, int sendcount) {
  return MPI_Alltoall(sendbuf, sendcount, MPI_FLOAT, recvbuf, sendcount, MPI_FLOAT, MPI_COMM_WORLD);
}

template <>
int caffe_Alltoall<double>(double* sendbuf, double* recvbuf, int sendcount) {
  return MPI_Alltoall(sendbuf, sendcount, MPI_DOUBLE, recvbuf, sendcount, MPI_DOUBLE, MPI_COMM_WORLD);
}
  /*
template <>
int caffe_Ialltoall<float>(float* sendbuf, int sendcount, MPI_Request* req) {
  return MPI_Ialltoall(MPI_IN_PLACE, sendcount, MPI_FLOAT, sendbuf, sendcount, MPI_FLOAT, MPI_COMM_WORLD, req);
}
  */
  /*
template <>
int caffe_Alltoall<int>(int* sendbuf, int sendcount) {
  return MPI_Alltoall(MPI_IN_PLACE, sendcount, MPI_INT, sendbuf, sendcount, MPI_INT, MPI_COMM_WORLD);
}

template <>
int caffe_Alltoall<unsigned int>(unsigned int* sendbuf, int sendcount) {
  return MPI_Alltoall(MPI_IN_PLACE, sendcount, MPI_INT, sendbuf, sendcount, MPI_INT, MPI_COMM_WORLD);
}
  */
  /*
template <>
int caffe_Ialltoall<double>(double* sendbuf, int sendcount, MPI_Request* req) {
  return MPI_Ialltoall(MPI_IN_PLACE, sendcount, MPI_DOUBLE, sendbuf, sendcount, MPI_DOUBLE, MPI_COMM_WORLD, req);
}

template <>
int caffe_Ialltoall<int>(int* sendbuf, int sendcount, MPI_Request* req) {
  return MPI_Ialltoall(MPI_IN_PLACE, sendcount, MPI_INT, sendbuf, sendcount, MPI_INT, MPI_COMM_WORLD, req);
}

template <>
int caffe_Ialltoall<unsigned int>(unsigned int* sendbuf, int sendcount, MPI_Request* req) {
  return MPI_Ialltoall(MPI_IN_PLACE, sendcount, MPI_INT, sendbuf, sendcount, MPI_INT, MPI_COMM_WORLD, req);
}

int caffe_Wait(MPI_Request* req) {
  return MPI_Wait(req, NULL);
}
  */

}
