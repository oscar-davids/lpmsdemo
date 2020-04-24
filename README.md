
# LPMS - Livepeer media server for scene classification 

This project is the [lpms](https://github.com/livepeer/lpms) fork project for scene classification.

GPU is recommended for real-time scene classification and transcoding.

To try this project as a standalone service, follow the instructions below.

### Requirements

Project requires libavcodec (ffmpeg) and friends. See `install_ffmpeg.sh` . Running this script will install everything in `~/compiled`. In order to build LPMS, the dependent libraries need to be discoverable by pkg-config and golang. If you installed everything with `install_ffmpeg.sh` , then run `export PKG_CONFIG_PATH=~/compiled/lib/pkgconfig:$PKG_CONFIG_PATH` so the deps are picked up.
  
  remark: for use subtitle should be build ffmpeg with --enable-libass

For classification need to install tensorflow. Because currently project use the tensorflow 1.15 version,if have GPU, need to install the CUDA 10.0 version additionally. For installation of tensorflow, refer to the following URL:

 https://www.tensorflow.org/install/install_c
 https://www.tensorflow.org/install/gpu
 
For build the project need to install golang.

https://golang.org/doc/install

### Build 

```
git clone https://github.com/oscar-davids/lpmsdemo.git 

cd lpmsdemo

go build cmd/example/main.go

```

### Testing for classification

If build successed, can find main execute file in lpmsdemo folder.

For classification need to trained medel file. please change trained file(base on tensorflow)  to tmodel.pb  and copy in lpmsdemo folder.

The test server exposes a few different endpoints:
1. `rtmp://localhost:1935/stream/test` for uploading/viewing RTMP video stream.
2. `http://localhost:7935/stream/test2.m3u8` for verification classification the HLS video stream.

Do the following steps to view a live stream video:

1. Start LPMS by running `go run cmd/example/main.go`

2. Upload an RTMP video stream to `rtmp://localhost:1935/stream/test`.  We recommend using ffmpeg or [OBS](https://obsproject.com/download).

For ffmpeg on osx, run: `ffmpeg -f avfoundation -framerate 30 -pixel_format uyvy422 -i "0:0" -c:v libx264 -tune zerolatency -b:v 900k -x264-params keyint=60:min-keyint=60 -c:a aac -ac 2 -ar 44100 -f flv rtmp://localhost:1935/stream/test`

For OBS, fill in Settings->Stream->URL to be rtmp://localhost:1935

3. If you have successfully uploaded the stream, you should see something like this in the LPMS output
```
I0324 09:44:14.639405   80673 listener.go:28] RTMP server got upstream
I0324 09:44:14.639429   80673 listener.go:42] Got RTMP Stream: test
```
4. If you have successfully classification one scene, you should see something like this in the output

```
I0324 09:44:14.639405   80673 listener.go:28] RTMP server got upstream
I0324 09:44:14.639429   80673 listener.go:42] Got RTMP Stream: test
```


5. Now you have a RTMP video stream running, we can view it from the server.  Simply run `ffplay http://localhost:7935/stream/test2.m3u8`, you should see the hls video playback.

### GPU Support

Processing on Nvidia GPUs is supported. To enable this capability, FFmpeg needs
to be built with GPU support. See the
[FFmpeg guidelines](https://trac.ffmpeg.org/wiki/HWAccelIntro#NVENCNVDEC) on
this.

To execute the nvidia tests within the `ffmpeg` directory, run this command:

```
go test -tag=nvidia -run Nvidia

```

To run the tests on a particular GPU, use the GPU_DEVICE environment variable:

```
# Runs on GPU number 3
GPU_DEVICE=3 go test -tag nvidia -run Nvidia
```

Aside from the tests themselves, there is a
[sample program](https://github.com/livepeer/lpms/blob/master/cmd/transcoding/transcoding.go)
that can be used as a reference to the LPMS GPU transcoding API. The sample
program can select GPU or software processing via CLI flags. Run the sample
program via:

```
# software processing
go run cmd/transcoding/transcoding.go transcoder/test.ts P144p30fps16x9,P240p30fps16x9 sw

# nvidia processing, GPU number 2
go run cmd/transcoding/transcoding.go transcoder/test.ts P144p30fps16x9,P240p30fps16x9 nv 2
```

You can follow the development of LPMS and Livepeer @ our [forum](http://forum.livepeer.org)
