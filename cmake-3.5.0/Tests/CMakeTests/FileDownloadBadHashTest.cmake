set(url "file:///home/ra000022/a03330/caffe/cmake-3.5.0/Tests/CMakeTests/FileDownloadInput.png")
set(dir "/home/ra000022/a03330/caffe/cmake-3.5.0/Tests/CMakeTests/downloads")

file(DOWNLOAD
  ${url}
  ${dir}/file3.png
  TIMEOUT 2
  STATUS status
  EXPECTED_HASH SHA1=5555555555555555555555555555555555555555
  )
